package com.igormaznitsa.nanollvm.internal;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.OnnxProtoReader.OnnxGraphBundle;
import com.igormaznitsa.nanollvm.internal.OnnxProtoReader.OnnxTensorProto;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoderFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Tier A ONNX weight import: graph <em>initializers</em> → {@link WeightBag} for existing causal /
 * embedding Java graphs (no ONNX Runtime, no op execution).
 *
 * <p>Looks under the model folder root and {@code onnx/}. Prefers
 * {@code model.onnx} / {@code model_fp16.onnx} / Optimum decoder names; rejects community quant and
 * {@code with_past} filenames via {@link #isAllowedOnnxName(String)}.
 *
 * @since 1.1.0
 */
public final class OnnxModelLoader {

  private static final List<String> PREFERRED_NAMES = List.of(
    "model.onnx",
    "model_fp16.onnx",
    "decoder_model_merged.onnx",
    "decoder_model.onnx",
    "encoder_model.onnx");

  private OnnxModelLoader() {
  }

  /**
   * {@code true} when {@link #selectOnnxFile(Path)} finds an allowed {@code .onnx} file.
   *
   * @since 1.1.0
   */
  public static boolean hasOnnxWeights(final Path modelDir) throws IOException {
    return selectOnnxFile(modelDir) != null;
  }

  /**
   * Picks the preferred allowed ONNX file under {@code modelDir} or {@code modelDir/onnx}, or
   * {@code null} if none.
   *
   * @since 1.1.0
   */
  public static Path selectOnnxFile(final Path modelDir) throws IOException {
    requireNonNull(modelDir, "modelDir");
    for (Path dir : List.of(modelDir, modelDir.resolve("onnx"))) {
      if (!Files.isDirectory(dir)) {
        continue;
      }
      for (String name : PREFERRED_NAMES) {
        Path candidate = dir.resolve(name);
        if (Files.isRegularFile(candidate) && isAllowedOnnxName(name)) {
          return candidate;
        }
      }
      try (Stream<Path> stream = Files.list(dir)) {
        List<Path> others = stream
          .filter(Files::isRegularFile)
          .filter(p -> p.getFileName().toString().toLowerCase(ROOT).endsWith(".onnx"))
          .filter(p -> isAllowedOnnxName(p.getFileName().toString()))
          .sorted()
          .toList();
        if (!others.isEmpty()) {
          return others.getFirst();
        }
      }
    }
    return null;
  }

  /**
   * Loads weights from {@link #selectOnnxFile(Path)} into a bag matching {@code schema}.
   *
   * @since 1.1.0
   */
  public static WeightBag loadWeights(
    final Path modelDir,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) throws IOException {
    Path onnxFile = selectOnnxFile(modelDir);
    if (onnxFile == null) {
      throw new ModelLoadException("no supported .onnx weight file in " + modelDir);
    }
    return loadWeights(onnxFile, modelDir, hfConfig, schema, io);
  }

  /**
   * Loads weights from an explicit ONNX file; {@code modelDir} is the external_data base when the
   * file sits beside sidecars.
   *
   * @since 1.1.0
   */
  public static WeightBag loadWeights(
    final Path onnxFile,
    final Path modelDir,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) throws IOException {
    requireNonNull(onnxFile, "onnxFile");
    requireNonNull(modelDir, "modelDir");
    requireNonNull(hfConfig, "hfConfig");
    requireNonNull(schema, "schema");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    long bytes = Files.size(onnxFile);
    LlmListeners.infof(streams, null, "Loading ONNX weights from %s (%.2f MiB)%n",
      onnxFile, bytes / (1024.0 * 1024.0));

    OnnxGraphBundle graph = OnnxProtoReader.readGraph(onnxFile);
    if (graph.initializers().isEmpty()) {
      throw new ModelLoadException("ONNX file has no initializers: " + onnxFile);
    }

    boolean embedding = EmbeddingEncoderFactory.isEmbeddingArchitecture(hfConfig);
    Path externalBase = onnxFile.getParent() == null ? modelDir : onnxFile.getParent();
    Map<String, Tensor> named = decodeFloatingInitializers(graph, externalBase, embedding, streams);
    return ModelLoader.assembleFromNamedTensors(
      named, onnxFile.toString(), hfConfig, schema, streams);
  }

  private static Map<String, Tensor> decodeFloatingInitializers(
    final OnnxGraphBundle graph,
    final Path externalBase,
    final boolean embedding,
    final LlmListener streams
  ) throws IOException {
    Map<String, Tensor> named = new LinkedHashMap<>();
    int total = (int) graph.initializers().stream()
      .filter(OnnxDataTypes::shouldLoadAsWeight)
      .count();
    LoadProgress progress = new LoadProgress("ONNX weights", total, streams);
    try {
      for (OnnxTensorProto proto : graph.initializers()) {
        if (putFloatingInitializer(named, graph, proto, externalBase, embedding)) {
          progress.step(proto.name() == null ? "" : proto.name());
        }
      }
      progress.finish("%d tensors".formatted(named.size()));
    } catch (RuntimeException | IOException e) {
      progress.finish("failed");
      throw e;
    }
    return named;
  }

  /**
   * ONNX MatMul stores {@code B} as {@code [in, out]}; PyTorch Linear weights are {@code [out, in]}.
   *
   * @since 1.1.0
   */
  static Tensor transposeIfMatrix(final Tensor tensor) {
    if (tensor.shape().length != 2) {
      return tensor;
    }
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    float[] src = tensor.toFloatArray();
    float[] dst = new float[src.length];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        dst[c * rows + r] = src[r * cols + c];
      }
    }
    return Tensor.of(dst, cols, rows);
  }

  private static boolean putFloatingInitializer(
    final Map<String, Tensor> named,
    final OnnxGraphBundle graph,
    final OnnxTensorProto proto,
    final Path externalBase,
    final boolean embedding
  ) throws IOException {
    if (!OnnxDataTypes.shouldLoadAsWeight(proto)) {
      OnnxDataTypes.requireHandledOrSkip(proto);
      return false;
    }
    Tensor tensor = OnnxWeightReader.toTensor(proto, externalBase);
    String alias = graph.matMulWeightAliases().get(proto.name());
    if (alias != null) {
      named.put(OnnxWeightNames.normalizeChatName(alias), transposeIfMatrix(tensor));
      return true;
    }
    String mapped = embedding
      ? OnnxWeightNames.normalizeBertName(proto.name())
      : OnnxWeightNames.normalizeChatName(proto.name());
    named.put(mapped, tensor);
    return true;
  }

  /**
   * Filename gate for Tier A: must end with {@code .onnx} and must not look like a community quant
   * or KV-{@code with_past} export ({@code _q4}, {@code _int8}, {@code _uint8}, {@code _bnb4},
   * {@code _quantized}, {@code with_past}).
   *
   * @since 1.1.0
   */
  public static boolean isAllowedOnnxName(final String fileName) {
    String lower = fileName.toLowerCase(Locale.ROOT);
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

  /**
   * All allowed {@code .onnx} files under the folder root and {@code onnx/}, sorted.
   *
   * @since 1.1.0
   */
  public static List<Path> listCandidateOnnxFiles(final Path modelDir) throws IOException {
    List<Path> found = new ArrayList<>();
    for (Path dir : List.of(modelDir, modelDir.resolve("onnx"))) {
      if (!Files.isDirectory(dir)) {
        continue;
      }
      try (Stream<Path> stream = Files.list(dir)) {
        stream.filter(Files::isRegularFile)
          .filter(p -> p.getFileName().toString().toLowerCase(ROOT).endsWith(".onnx"))
          .filter(p -> isAllowedOnnxName(p.getFileName().toString()))
          .sorted()
          .forEach(found::add);
      }
    }
    return found;
  }

  /**
   * In-memory ONNX load for {@code ModelFileSource} / classpath paths. Rejects float initializers
   * that use {@code external_data} — use {@link #loadWeights(Path, Path, Config.HfConfig,
   * WeightSchema, LlmListener)} / {@code LlmModelFactory.make(Path)} instead.
   *
   * @since 1.1.0
   */
  public static WeightBag loadWeightsFromBytes(
    final byte[] onnxBytes,
    final String label,
    final Config.HfConfig hfConfig,
    final WeightSchema schema,
    final LlmListener io
  ) throws IOException {
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(
      ByteBuffer.wrap(onnxBytes).order(ByteOrder.LITTLE_ENDIAN), label);
    boolean embedding = EmbeddingEncoderFactory.isEmbeddingArchitecture(hfConfig);
    for (OnnxTensorProto proto : graph.initializers()) {
      if (proto.hasExternalData()
        && proto.name() != null
        && !proto.name().isBlank()
        && OnnxWeightReader.isFloatingWeightType(proto.dataType())) {
        throw new ModelLoadException(
          "ONNX external_data is not supported for ModelFileSource loads; use make(Path): "
            + proto.name());
      }
    }
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Map<String, Tensor> named =
      decodeFloatingInitializers(graph, Path.of("/"), embedding, streams);
    return ModelLoader.assembleFromNamedTensors(named, label, hfConfig, schema, streams);
  }
}
