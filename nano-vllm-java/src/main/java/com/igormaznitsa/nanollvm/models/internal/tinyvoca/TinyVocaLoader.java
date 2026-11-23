package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.llm.TinyVocaSpec;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufReader;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxGraphBundle;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxTensorProto;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxWeightReader;
import com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsReader;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Loads a tiny-voca checkpoint from safetensors, GGUF, or ONNX next to {@code config.json}.
 * A lone {@code .gguf} may carry the config and the lexicon in its embedded files.
 */
public final class TinyVocaLoader {

  private TinyVocaLoader() {
  }

  public static Loaded load(final Path path) throws IOException {
    Path file = requireNonNull(path, "path").toAbsolutePath().normalize();
    if (Files.isRegularFile(file) && fileName(file).endsWith(".gguf")) {
      Path parent = file.getParent();
      if (parent == null) {
        throw new ModelLoadException("tiny-voca GGUF has no parent folder: " + file);
      }
      return loadGguf(file, parent);
    }
    if (!Files.isDirectory(file)) {
      throw new ModelLoadException(
        "tiny-voca checkpoint must be a folder or a .gguf file: " + file);
    }
    Path configFile = file.resolve("config.json");
    if (!Files.isRegularFile(configFile)) {
      throw new ModelLoadException("tiny-voca folder has no config.json: " + file);
    }
    String configJson = Files.readString(configFile, UTF_8);
    TinyVocaSpec spec = TinyVocaSpec.parse(configJson);
    List<Path> weights = weightFiles(file);
    if (weights.isEmpty()) {
      throw new ModelLoadException(
        "tiny-voca folder has no safetensors, GGUF, or ONNX weights. "
          + "A .pt pickle is not loaded. Run models/download-tiny-voca.sh");
    }
    Map<String, Tensor> tensors = null;
    IllegalArgumentException last = null;
    for (Path weight : weights) {
      try {
        Map<String, Tensor> candidate = readWeights(weight, file);
        if (candidate.containsKey("tokens.weight")
          && candidate.containsKey("encoder.blocks.0.attention.qkv.weight")) {
          tensors = candidate;
          break;
        }
      } catch (IllegalArgumentException failed) {
        last = failed;
      }
    }
    if (tensors == null) {
      throw new ModelLoadException(
        "tiny-voca weights are missing tokens.weight or encoder.blocks.0.attention.qkv.weight in "
          + file + (last == null ? "" : " (" + last.getMessage() + ")"));
    }
    TinyVocaGraph graph = new TinyVocaGraph(spec, tensors);
    String lexicon = spec.isRussian() || !DizioneFront.lexiconFileMissing(file)
      ? null
      : embeddedLexicon(weights);
    return new Loaded(spec, graph, front(spec, file, lexicon));
  }

  private static Loaded loadGguf(final Path gguf, final Path neighbor) throws IOException {
    try (GgufReader reader = GgufReader.open(gguf)) {
      String configJson = neighbor == null ? null : readIfExists(neighbor.resolve("config.json"));
      Map<String, byte[]> embedded = embedded(reader);
      if (configJson == null) {
        byte[] config = embedded.get("config.json");
        if (config == null) {
          throw new ModelLoadException(
            "tiny-voca GGUF has no config.json beside it or inside it: " + gguf);
        }
        configJson = new String(config, UTF_8);
      }
      TinyVocaSpec spec = TinyVocaSpec.parse(configJson);
      Map<String, Tensor> tensors = new LinkedHashMap<>();
      for (String name : reader.tensorNames()) {
        tensors.put(canonical(name), reader.getTensor(name));
      }
      TinyVocaGraph graph = new TinyVocaGraph(spec, tensors);
      String lexicon = text(embedded.get("phonemizer/lexicon.tsv"));
      if (lexicon == null) {
        lexicon = text(embedded.get("lexicon.tsv"));
      }
      return new Loaded(spec, graph,
        front(spec, neighbor == null ? gguf.getParent() : neighbor, lexicon));
    }
  }

  private static TinyVocaFront front(
    final TinyVocaSpec spec,
    final Path modelDir,
    final String embeddedLexicon
  ) throws IOException {
    if (spec.isRussian()) {
      return new RussianVocaFront(spec, modelDir);
    }
    if (embeddedLexicon != null && DizioneFront.lexiconFileMissing(modelDir)) {
      return DizioneFront.fromLexicon(spec, embeddedLexicon);
    }
    return DizioneFront.load(spec, modelDir);
  }

  private static String embeddedLexicon(final List<Path> weights) throws IOException {
    for (Path weight : weights) {
      if (!fileName(weight).endsWith(".gguf")) {
        continue;
      }
      try (GgufReader reader = GgufReader.open(weight)) {
        Map<String, byte[]> embedded = embedded(reader);
        String lexicon = text(embedded.get("phonemizer/lexicon.tsv"));
        if (lexicon == null) {
          lexicon = text(embedded.get("lexicon.tsv"));
        }
        if (lexicon != null) {
          return lexicon;
        }
      }
    }
    return null;
  }

  private static Map<String, byte[]> embedded(final GgufReader reader) {
    Object names = reader.metadata().get("audiocpp.embedded_files.names");
    Object data = reader.metadata().get("audiocpp.embedded_files.data");
    Object offsets = reader.metadata().get("audiocpp.embedded_files.offsets");
    if (!(names instanceof List<?> nameList) || !(offsets instanceof List<?> offsetList)) {
      return Map.of();
    }
    byte[] bytes = asBytes(data);
    if (bytes == null) {
      return Map.of();
    }
    Map<String, byte[]> files = new LinkedHashMap<>();
    for (int i = 0; i < nameList.size(); i++) {
      int from = number(offsetList, i);
      int to = i + 1 < offsetList.size() ? number(offsetList, i + 1) : bytes.length;
      if (from < 0 || to < from || to > bytes.length) {
        continue;
      }
      byte[] slice = new byte[to - from];
      System.arraycopy(bytes, from, slice, 0, slice.length);
      files.put(String.valueOf(nameList.get(i)), slice);
    }
    return files;
  }

  private static List<Path> weightFiles(final Path folder) throws IOException {
    List<Path> found = new ArrayList<>();
    try (var walk = Files.walk(folder, 2)) {
      walk.filter(Files::isRegularFile).forEach(path -> {
        String name = fileName(path);
        if (name.endsWith(".safetensors") || name.endsWith(".gguf") || name.endsWith(".onnx")) {
          found.add(path);
        }
      });
    }
    found.sort(Comparator.comparingInt(TinyVocaLoader::rank));
    return found;
  }

  private static int rank(final Path path) {
    String name = fileName(path);
    if (name.endsWith(".safetensors")) {
      return 0;
    }
    if (name.contains("f16")) {
      return 1;
    }
    if (name.contains("orig")) {
      return 2;
    }
    if (name.contains("q8")) {
      return 3;
    }
    if (name.endsWith(".gguf")) {
      return 4;
    }
    return 5;
  }

  private static String fileName(final Path path) {
    Path name = path.getFileName();
    if (name == null) {
      throw new ModelLoadException("tiny-voca path has no file name: " + path);
    }
    return name.toString().toLowerCase(Locale.ROOT);
  }

  private static Map<String, Tensor> readWeights(final Path weight, final Path modelDir)
    throws IOException {
    String name = fileName(weight);
    if (name.endsWith(".safetensors")) {
      return readSafetensors(weight);
    }
    if (name.endsWith(".gguf")) {
      return readGgufTensors(weight);
    }
    return readOnnx(weight, modelDir);
  }

  private static Map<String, Tensor> readSafetensors(final Path weight) throws IOException {
    Map<String, Tensor> tensors = new LinkedHashMap<>();
    try (SafetensorsReader reader = SafetensorsReader.open(weight)) {
      for (String name : reader.keys()) {
        tensors.put(canonical(name), reader.getTensor(name));
      }
    }
    return tensors;
  }

  private static Map<String, Tensor> readGgufTensors(final Path weight) throws IOException {
    Map<String, Tensor> tensors = new LinkedHashMap<>();
    try (GgufReader reader = GgufReader.open(weight)) {
      for (String name : reader.tensorNames()) {
        tensors.put(canonical(name), reader.getTensor(name));
      }
    }
    return tensors;
  }

  private static Map<String, Tensor> readOnnx(final Path weight, final Path modelDir)
    throws IOException {
    OnnxGraphBundle bundle = OnnxProtoReader.readGraph(weight);
    Map<String, String> rename = new LinkedHashMap<>();
    bundle.matMulWeightAliases().forEach((from, to) -> rename.put(from, canonical(to)));
    bundle.identityWeightAliases().forEach((from, to) -> rename.putIfAbsent(from, canonical(to)));
    Map<String, Tensor> tensors = new LinkedHashMap<>();
    for (OnnxTensorProto proto : bundle.initializers()) {
      if (!OnnxWeightReader.isFloatingWeightType(proto.dataType())) {
        continue;
      }
      String raw = proto.name() == null ? "" : proto.name();
      String key = rename.getOrDefault(raw, canonical(raw));
      if (key.isBlank() || key.startsWith("onnx::")) {
        continue;
      }
      Tensor tensor = OnnxWeightReader.toTensor(proto, modelDir);
      if (bundle.matMulWeightAliases().containsKey(raw) && tensor.shape().length == 2) {
        tensor = transpose(tensor);
      }
      tensors.put(key, tensor);
    }
    return tensors;
  }

  static String canonical(final String raw) {
    String name = raw.replace('\\', '/');
    while (name.startsWith("/")) {
      name = name.substring(1);
    }
    name = name.replace('/', '.');
    if (name.startsWith("model.")) {
      name = name.substring("model.".length());
    }
    return name;
  }

  private static Tensor transpose(final Tensor tensor) {
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    float[] source = tensor.toFloatArray();
    float[] dest = new float[source.length];
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        dest[col * rows + row] = source[row * cols + col];
      }
    }
    return Tensor.of(dest, cols, rows);
  }

  private static String readIfExists(final Path path) throws IOException {
    return Files.isRegularFile(path) ? Files.readString(path, UTF_8) : null;
  }

  private static String text(final byte[] bytes) {
    return bytes == null || bytes.length == 0 ? null : new String(bytes, UTF_8);
  }

  private static byte[] asBytes(final Object value) {
    if (value instanceof byte[] bytes) {
      return bytes;
    }
    if (value instanceof String text) {
      return text.getBytes(UTF_8);
    }
    if (!(value instanceof List<?> list)) {
      return null;
    }
    byte[] bytes = new byte[list.size()];
    for (int i = 0; i < list.size(); i++) {
      if (!(list.get(i) instanceof Number number)) {
        return null;
      }
      bytes[i] = (byte) number.intValue();
    }
    return bytes;
  }

  private static int number(final List<?> values, final int index) {
    Object value = values.get(index);
    return value instanceof Number number ? number.intValue() : 0;
  }

  public record Loaded(TinyVocaSpec spec, TinyVocaGraph graph, TinyVocaFront front) {
  }
}
