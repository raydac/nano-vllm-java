package com.igormaznitsa.nanollvm.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.GemmaQatWeight;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads Gemma 4 QAT language weights from safetensors (skips vision/audio towers; keeps int2/4/8 packed).
 */
public final class Gemma4QatLoader {

  private static final String LANGUAGE_PREFIX = "model.language_model.";

  private Gemma4QatLoader() {
  }

  public static WeightBag loadWeights(
    final Path modelDir,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    List<Path> files = SafetensorsReader.listSafetensors(modelDir);
    if (files.isEmpty()) {
      throw new IllegalArgumentException("no .safetensors files in " + modelDir);
    }
    long fileBytes = 0L;
    for (Path file : files) {
      fileBytes += Files.size(file);
    }
    LlmListeners.infof(streams, null, "Loading Gemma 4 QAT language weights from %s (%.2f GiB)%n",
      modelDir, fileBytes / (1024.0 * 1024.0 * 1024.0));
    List<WeightSource> sources = files.stream()
      .map(
        file -> new WeightSource(file.getFileName().toString(), () -> SafetensorsReader.open(file)))
      .toList();
    return assemble(sources, modelDir.toString(), hfConfig, schema, streams);
  }

  public static WeightBag loadWeights(
    final List<ModelFileBundle.NamedBytes> blobs,
    final String label,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    if (blobs.isEmpty()) {
      throw new IllegalArgumentException("no .safetensors blobs for " + label);
    }
    List<WeightSource> sources = blobs.stream()
      .map(blob -> new WeightSource(
        blob.name(),
        () -> SafetensorsReader.open(blob.buffer(), blob.name())))
      .toList();
    return assemble(sources, label, hfConfig, schema, streams);
  }

  private static WeightBag assemble(
    final List<WeightSource> sources,
    final String label,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener streams
  ) throws IOException {
    List<Located> located = new ArrayList<>();
    for (WeightSource source : sources) {
      try (SafetensorsReader probe = source.open()) {
        for (String rawName : probe.keys()) {
          if (shouldSkip(rawName)) {
            continue;
          }
          String canonical = canonicalize(rawName);
          located.add(new Located(
            source, rawName, canonical, probe.dtype(rawName), probe.shape(rawName),
            probe.byteSize(rawName)));
        }
      }
    }
    if (located.isEmpty()) {
      throw new IllegalStateException("no Gemma 4 language tensors found in " + label);
    }

    Map<String, ModuleParts> modules = new LinkedHashMap<>();
    Map<String, Located> dense = new LinkedHashMap<>();
    for (Located item : located) {
      String role = tensorRole(item);
      if (role == null) {
        dense.put(item.canonical(), item);
        continue;
      }
      String param = paramName(item.canonical(), role);
      modules.computeIfAbsent(param, k -> new ModuleParts()).accept(role, item);
    }

    Map<String, Object> bag = new LinkedHashMap<>();
    List<LoadItem> plan = new ArrayList<>();
    for (Located item : dense.values()) {
      plan.add(new LoadItem(item, null));
    }
    for (Map.Entry<String, ModuleParts> entry : modules.entrySet()) {
      plan.add(new LoadItem(entry.getValue().payload(), entry.getKey()));
    }

    LoadProgress progress = new LoadProgress("Gemma 4 QAT", plan.size(), streams);
    WeightSource current = null;
    SafetensorsReader reader = null;
    long loadedBytes = 0L;
    try {
      for (LoadItem item : plan) {
        Located locatedItem = item.located();
        if (reader == null || locatedItem.source() != current) {
          if (reader != null) {
            reader.close();
          }
          current = locatedItem.source();
          reader = current.open();
        }
        if (item.qatParam() == null) {
          bag.put(locatedItem.canonical(), reader.getTensor(locatedItem.rawName()));
          loadedBytes += locatedItem.byteSize();
        } else {
          ModuleParts parts = modules.get(item.qatParam());
          bag.put(item.qatParam(), loadQat(reader, parts, hfConfig, item.qatParam()));
          loadedBytes += parts.payload().byteSize();
        }
        progress.step("%s | %s (%.0f MiB)".formatted(
          locatedItem.source().label(),
          locatedItem.rawName(),
          loadedBytes / (1024.0 * 1024.0)));
      }
      progress.finish("%.0f MiB language tensors".formatted(loadedBytes / (1024.0 * 1024.0)));
    } catch (RuntimeException | IOException e) {
      progress.finish("failed");
      throw e;
    } finally {
      if (reader != null) {
        reader.close();
      }
    }

    WeightBag weights = new WeightBag(bag);
    for (String required : schema.expectedParameters()) {
      if (!weights.has(required)) {
        throw new IllegalStateException("missing required Gemma 4 weight after load: " + required);
      }
    }
    return weights;
  }

  private static GemmaQatWeight loadQat(
    final SafetensorsReader reader,
    final ModuleParts parts,
    final Config.HfConfig hf,
    final String param
  ) {
    Located payload = parts.payload();
    byte[] packed = reader.getRaw(payload.rawName());
    Tensor scaleTensor = readAssociated(reader, parts.scale());
    float[] scales = scaleTensor.toFloatArray();
    int scaleCols = scaleTensor.ndim() >= 2 ? scaleTensor.size(scaleTensor.ndim() - 1) : 1;
    int rows = payload.shape()[0];
    int packedWidth = payload.shape()[payload.shape().length - 1];
    int bits = inferBits(param, payload.dtype(), hf);
    int cols = bits == 8 ? packedWidth : packedWidth * (8 / bits);
    float inScale = scalar(reader, parts.inputScale());
    float outScale = scalar(reader, parts.outputScale());
    return new GemmaQatWeight(param, packed, scales, rows, cols, bits, scaleCols, inScale,
      outScale);
  }

  private static Tensor readAssociated(final SafetensorsReader reader, final Located located) {
    if (located == null) {
      throw new IllegalStateException("missing QAT scale tensor");
    }
    if (located.source() != null && !located.source().label().equals(reader.label())
      && !reader.contains(located.rawName())) {
      throw new IllegalStateException("QAT scale not in the same shard as packed weight: "
        + located.rawName());
    }
    return reader.getTensor(located.rawName());
  }

  private static float scalar(final SafetensorsReader reader, final Located located) {
    if (located == null) {
      return 0f;
    }
    Tensor tensor = reader.getTensor(located.rawName());
    return tensor.numel() == 0 ? 0f : tensor.data()[tensor.offset()];
  }

  private static int inferBits(final String param, final String dtype, final Config.HfConfig hf) {
    if ("I8".equals(dtype)) {
      return 8;
    }
    if (param.contains("lm_head") || param.contains("embed_tokens.weight")
      && !param.contains("per_layer")) {
      return 2;
    }
    Integer layer = parseLayerIndex(param);
    if (layer != null && param.contains(".mlp.") && hf.isKvSharedLayer(layer)) {
      return 2;
    }
    return 4;
  }

  private static Integer parseLayerIndex(final String param) {
    String marker = "model.layers.";
    int start = param.indexOf(marker);
    if (start < 0) {
      return null;
    }
    int from = start + marker.length();
    int to = param.indexOf('.', from);
    if (to < 0) {
      return null;
    }
    try {
      return Integer.parseInt(param.substring(from, to));
    } catch (NumberFormatException ignored) {
      return null;
    }
  }

  private static boolean shouldSkip(final String rawName) {
    return rawName.contains("vision_tower")
      || rawName.contains("audio_tower")
      || rawName.contains("embed_vision")
      || rawName.contains("embed_audio")
      || rawName.endsWith("k_cache_scale")
      || rawName.endsWith("v_cache_scale");
  }

  private static String canonicalize(final String rawName) {
    if (rawName.startsWith(LANGUAGE_PREFIX)) {
      return "model." + rawName.substring(LANGUAGE_PREFIX.length());
    }
    return rawName;
  }

  private static String tensorRole(final Located item) {
    String canonical = item.canonical();
    if (canonical.endsWith(".embedding_quantized")) {
      return "quant";
    }
    if (canonical.endsWith(".embedding_scale")) {
      return "scale";
    }
    if (canonical.endsWith(".weight_scale")) {
      return "scale";
    }
    if (canonical.endsWith(".input_activation_scale")) {
      return "in_scale";
    }
    if (canonical.endsWith(".output_activation_scale")) {
      return "out_scale";
    }
    if (canonical.endsWith(".weight") && isPackedDtype(item.dtype())) {
      return "quant";
    }
    return null;
  }

  private static boolean isPackedDtype(final String dtype) {
    return "U8".equals(dtype) || "I8".equals(dtype);
  }

  private static String paramName(final String canonical, final String role) {
    return switch (role) {
      case "quant" -> canonical.endsWith(".embedding_quantized")
        ? canonical.substring(0, canonical.length() - ".embedding_quantized".length()) + ".weight"
        : canonical;
      case "scale" -> canonical.endsWith(".embedding_scale")
        ? canonical.substring(0, canonical.length() - ".embedding_scale".length()) + ".weight"
        : canonical.substring(0, canonical.length() - ".weight_scale".length()) + ".weight";
      case "in_scale" ->
        canonical.substring(0, canonical.length() - ".input_activation_scale".length()) + ".weight";
      case "out_scale" ->
        canonical.substring(0, canonical.length() - ".output_activation_scale".length()) +
          ".weight";
      default -> canonical;
    };
  }

  @FunctionalInterface
  private interface ReaderOpen {
    SafetensorsReader open() throws IOException;
  }

  private record WeightSource(String label, ReaderOpen opener) {
    SafetensorsReader open() throws IOException {
      return this.opener.open();
    }
  }

  private record Located(
    WeightSource source,
    String rawName,
    String canonical,
    String dtype,
    int[] shape,
    long byteSize
  ) {
  }

  private record LoadItem(Located located, String qatParam) {
  }

  private static final class ModuleParts {
    private Located quant;
    private Located scale;
    private Located inputScale;
    private Located outputScale;

    void accept(final String role, final Located located) {
      switch (role) {
        case "quant" -> this.quant = located;
        case "scale" -> this.scale = located;
        case "in_scale" -> this.inputScale = located;
        case "out_scale" -> this.outputScale = located;
        default -> throw new IllegalArgumentException(role);
      }
    }

    Located payload() {
      return requireNonNull(this.quant, "missing packed QAT payload");
    }

    Located scale() {
      return this.scale;
    }

    Located inputScale() {
      return this.inputScale;
    }

    Located outputScale() {
      return this.outputScale;
    }
  }
}
