package com.igormaznitsa.nanollvm.models.llmcontainer;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxGraphBundle;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxTensorProto;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * ONNX protobuf transport: file pick, graph initializers, and container dtype decode to float32.
 * Does not remap names or transpose MatMul weights — that is the architecture processor.
 *
 * @since 1.1.0
 */
public final class OnnxTransport implements ContainerTransport {

  private static final List<String> PREFERRED_NAMES = List.of(
    "model.onnx",
    "model_fp16.onnx",
    "decoder_model_merged.onnx",
    "decoder_model.onnx",
    "encoder_model.onnx");

  private final String label;
  private final String configJson;
  private final OnnxGraphBundle graph;
  private final Path externalBase;
  private final ContainerCatalog catalog;

  private OnnxTransport(
    final String label,
    final String configJson,
    final OnnxGraphBundle graph,
    final Path externalBase
  ) {
    this.label = requireNonNull(label, "label");
    this.configJson = requireNonNull(configJson, "configJson");
    this.graph = requireNonNull(graph, "graph");
    this.externalBase = externalBase;
    Set<String> names = new LinkedHashSet<>();
    this.graph.initializers().stream()
      .map(OnnxTensorProto::name)
      .filter(name -> name != null && !name.isBlank())
      .forEach(names::add);
    this.catalog = ContainerCatalog.ofHf(
      ModelSupport.Source.ONNX, this.label, this.configJson, names);
  }

  /**
   * {@code true} when the folder (or {@code onnx/} subfolder) has a supported {@code .onnx} file.
   *
   * @since 1.1.0
   */
  public static boolean present(final Path modelDir) throws IOException {
    return selectFile(modelDir) != null;
  }

  public static Path selectFile(final Path modelDir) throws IOException {
    requireNonNull(modelDir, "modelDir");
    for (Path dir : List.of(modelDir, modelDir.resolve("onnx"))) {
      if (!Files.isDirectory(dir)) {
        continue;
      }
      for (String name : PREFERRED_NAMES) {
        Path candidate = dir.resolve(name);
        if (Files.isRegularFile(candidate) && isAllowedName(name)) {
          return candidate;
        }
      }
      try (Stream<Path> stream = Files.list(dir)) {
        List<Path> others = stream
          .filter(Files::isRegularFile)
          .filter(OnnxTransport::isCandidate)
          .sorted()
          .toList();
        if (!others.isEmpty()) {
          return others.getFirst();
        }
      }
    }
    return null;
  }

  public static boolean isAllowedName(final String fileName) {
    String lower = fileName.toLowerCase(ROOT);
    if (!lower.endsWith(".onnx")) {
      return false;
    }
    return !(lower.contains("_q4")
      || lower.contains("_int8")
      || lower.contains("_uint8")
      || lower.contains("_bnb4")
      || lower.contains("_quantized")
      || lower.contains("with_past"));
  }

  public static List<Path> listCandidateFiles(final Path modelDir) throws IOException {
    List<Path> found = new ArrayList<>();
    for (Path dir : List.of(modelDir, modelDir.resolve("onnx"))) {
      if (!Files.isDirectory(dir)) {
        continue;
      }
      try (Stream<Path> stream = Files.list(dir)) {
        stream.filter(Files::isRegularFile)
          .filter(OnnxTransport::isCandidate)
          .sorted()
          .forEach(found::add);
      }
    }
    return found;
  }

  /**
   * Opens an HF folder: reads {@code config.json} and a preferred {@code .onnx} file.
   *
   * @since 1.1.0
   */
  public static OnnxTransport open(final Path modelDir) throws IOException {
    Path dir = requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
    return open(dir, readConfigJson(dir));
  }

  /**
   * Opens an HF folder with a caller-supplied {@code config.json} body.
   *
   * @since 1.1.0
   */
  public static OnnxTransport open(final Path modelDir, final String configJson)
    throws IOException {
    Path dir = requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
    Path onnxFile = selectFile(dir);
    if (onnxFile == null) {
      throw new ModelLoadException("no supported .onnx weight file in " + dir);
    }
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(onnxFile);
    if (graph.initializers().isEmpty()) {
      throw new ModelLoadException("ONNX file has no initializers: " + onnxFile);
    }
    Path externalBase = onnxFile.getParent() == null ? dir : onnxFile.getParent();
    return new OnnxTransport(onnxFile.toString(), configJson, graph, externalBase);
  }

  /**
   * Opens ONNX protobuf already in heap (no {@code external_data} sidecars).
   *
   * @since 1.1.0
   */
  public static OnnxTransport open(
    final byte[] onnxBytes,
    final String label,
    final String configJson
  ) throws IOException {
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(
      ByteBuffer.wrap(requireNonNull(onnxBytes, "onnxBytes")).order(ByteOrder.LITTLE_ENDIAN),
      requireNonNull(label, "label"));
    if (graph.initializers().isEmpty()) {
      throw new ModelLoadException("ONNX file has no initializers: " + label);
    }
    return new OnnxTransport(label, configJson, graph, null);
  }

  private static String readConfigJson(final Path modelDir) throws IOException {
    Path configPath = modelDir.resolve(CONFIG_JSON);
    if (!Files.isRegularFile(configPath)) {
      throw new ModelLoadException("missing config.json in " + modelDir);
    }
    return Files.readString(configPath, UTF_8);
  }

  private static boolean isCandidate(final Path path) {
    String name = PathNames.of(path);
    return name.toLowerCase(ROOT).endsWith(".onnx") && isAllowedName(name);
  }

  public String label() {
    return this.label;
  }

  public String configJson() {
    return this.configJson;
  }

  public OnnxGraphBundle graph() {
    return this.graph;
  }

  public Map<String, String> matMulAliases() {
    return this.graph.matMulWeightAliases();
  }

  public Map<String, Tensor> readFloatingTensors(final LlmListener io) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Map<String, Tensor> named = new LinkedHashMap<>();
    int total = (int) this.graph.initializers().stream()
      .filter(OnnxDataTypes::shouldLoadAsWeight)
      .count();
    LoadProgress progress = new LoadProgress("ONNX weights", total, streams);
    try {
      for (OnnxTensorProto proto : this.graph.initializers()) {
        if (!OnnxDataTypes.shouldLoadAsWeight(proto)) {
          OnnxDataTypes.requireHandledOrSkip(proto);
          continue;
        }
        this.requireInlinePayload(proto);
        Tensor tensor = OnnxWeightReader.toTensor(proto, this.externalBase);
        String name = proto.name() == null ? "" : proto.name();
        named.put(name, tensor);
        progress.step(name);
      }
      progress.finish("%d tensors".formatted(named.size()));
    } catch (RuntimeException | IOException e) {
      progress.finish("failed");
      throw e;
    }
    return named;
  }

  private void requireInlinePayload(final OnnxTensorProto proto) {
    if (this.externalBase != null) {
      return;
    }
    if (proto.hasExternalData()
      && proto.name() != null
      && !proto.name().isBlank()
      && OnnxWeightReader.isFloatingWeightType(proto.dataType())) {
      throw new ModelLoadException(
        "ONNX external_data is not supported for ModelFileSource loads; use make(Path): "
          + proto.name());
    }
  }

  @Override
  public ContainerCatalog catalog() {
    return this.catalog;
  }

  @Override
  public void close() throws IOException {
  }
}
