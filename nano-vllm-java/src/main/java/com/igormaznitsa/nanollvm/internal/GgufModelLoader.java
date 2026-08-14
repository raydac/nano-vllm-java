package com.igormaznitsa.nanollvm.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Loads GGUF checkpoints ({@code lfm2} causal, {@code bert} embedding) into {@link Config.HfConfig} +
 * {@link WeightBag}. Default keeps large matrices GGML-packed; {@code allowUnpackParameters}
 * dequantizes each tensor to float32 from the mmap during load (no packed heap copy).
 */
public final class GgufModelLoader {

  private GgufModelLoader() {
  }

  public static LoadedGguf load(final Path ggufPath, final LlmListener io) throws IOException {
    return load(ggufPath, io, false);
  }

  public static LoadedGguf load(
    final Path ggufPath,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Path path = requireNonNull(ggufPath, "ggufPath").toAbsolutePath().normalize();
    LlmListeners.infof(streams, null, "Loading GGUF from %s%n", path);

    GgufReader reader = GgufReader.open(path);
    return load(reader, path.toString(), streams, allowUnpackParameters);
  }

  /**
   * @since 1.1.0
   */
  public static LoadedGguf load(
    final ByteBuffer data,
    final Path virtualPath,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) throws IOException {
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Path path = requireNonNull(virtualPath, "virtualPath").toAbsolutePath().normalize();
    LlmListeners.infof(streams, null, "Loading GGUF from memory (%s)%n", path);
    GgufReader reader = GgufReader.open(requireNonNull(data, "data"), path);
    return load(reader, path.toString(), streams, allowUnpackParameters);
  }

  private static LoadedGguf load(
    final GgufReader reader,
    final String label,
    final LlmListener streams,
    final boolean allowUnpackParameters
  ) throws IOException {
    String arch = reader.metaString("general.architecture", "");
    try {
      ModelSupport.Selection selected = ModelSupport.requireGguf(arch);
      if (selected.isEmbedding()) {
        return loadWeights(reader, buildBertConfig(reader), streams, allowUnpackParameters, arch);
      }
      return loadWeights(reader, buildLfm2Config(reader, arch), streams, allowUnpackParameters,
        arch);
    } catch (UnsupportedModelException e) {
      reader.close();
      throw new UnsupportedModelException(
        "Cannot load GGUF '" + label + "'."
          + System.lineSeparator() + System.lineSeparator() + e.getMessage(),
        e.modelType(),
        e.architectures());
    } catch (RuntimeException e) {
      reader.close();
      throw e;
    }
  }

  private static LoadedGguf loadWeights(
    final GgufReader reader,
    final Config.HfConfig config,
    final LlmListener streams,
    final boolean allowUnpackParameters,
    final String arch
  ) throws IOException {
    LlmListeners.infof(streams, null,
      "GGUF %s: layers=%d hidden=%d ff=%d heads=%d/%d%n",
      arch,
      config.numHiddenLayers(),
      config.hiddenSize(),
      config.intermediateSize(),
      config.numAttentionHeads(),
      config.numKeyValueHeads());

    Map<String, Object> weights = new LinkedHashMap<>();
    LoadProgress progress = new LoadProgress("GGUF weights", reader.tensorCount(), streams);
    long accountedBytes = 0L;
    try {
      for (String name : reader.tensorNames()) {
        if (allowUnpackParameters) {
          var tensor = reader.getTensor(name);
          weights.put(name, tensor);
          accountedBytes += (long) tensor.numel() * Float.BYTES;
          progress.step(
            "%s (%.0f MiB float32)".formatted(name, accountedBytes / (1024.0 * 1024.0)));
        } else {
          PackedWeight weight = reader.getPackedWeight(name);
          weights.put(name, weight);
          accountedBytes += weight.packedBytes();
          progress.step("%s (%.0f MiB packed)".formatted(name, accountedBytes / (1024.0 * 1024.0)));
        }
      }
      progress.finish(allowUnpackParameters
        ? "%.0f MiB float32 (unpacked at load)".formatted(accountedBytes / (1024.0 * 1024.0))
        : "%.0f MiB packed (dequant on matmul)".formatted(accountedBytes / (1024.0 * 1024.0)));
    } catch (RuntimeException e) {
      progress.finish("failed");
      reader.close();
      throw e;
    }

    return new LoadedGguf(config, new WeightBag(weights), reader);
  }

  private static Config.HfConfig buildBertConfig(final GgufReader reader) {
    String prefix = "bert";
    int hidden = reader.metaInt(prefix + ".embedding_length", 0);
    int layers = reader.metaInt(prefix + ".block_count", 0);
    int intermediate = reader.metaInt(prefix + ".feed_forward_length", 0);
    int heads = reader.metaInt(prefix + ".attention.head_count", 0);
    float normEps = reader.metaFloat(prefix + ".attention.layer_norm_epsilon", 1e-12f);
    int context = reader.metaInt(prefix + ".context_length", 512);
    int vocab = reader.metaStringArray("tokenizer.ggml.tokens").size();
    if (vocab == 0) {
      vocab = reader.metaInt(prefix + ".vocab_size", 0);
    }
    if (hidden <= 0 || layers <= 0 || heads <= 0 || intermediate <= 0 || vocab <= 0) {
      throw new IllegalStateException(
        "incomplete BERT GGUF metadata (hidden/layers/heads/ff/vocab)");
    }
    if (hidden % heads != 0) {
      throw new IllegalStateException(
        "BERT hiddenSize %d not divisible by heads %d".formatted(hidden, heads));
    }
    int headDim = hidden / heads;
    return new Config.HfConfig(
      vocab,
      hidden,
      intermediate,
      layers,
      heads,
      heads,
      headDim,
      context,
      normEps,
      "gelu",
      false,
      true,
      10_000f,
      null,
      "float32",
      WeightNames.ARCH_BERT,
      List.of("BertForEmbedding"),
      "gelu",
      0,
      List.of(),
      10_000f,
      0f,
      0,
      false,
      false,
      null
    );
  }

  private static Config.HfConfig buildLfm2Config(final GgufReader reader, final String arch) {
    String prefix = arch.contains("lfm2moe") ? "lfm2moe" : "lfm2";
    int hidden = reader.metaInt(prefix + ".embedding_length", 0);
    int layers = reader.metaInt(prefix + ".block_count", 0);
    int intermediate = reader.metaInt(prefix + ".feed_forward_length", 0);
    int heads = reader.metaInt(prefix + ".attention.head_count", 0);
    int kvHeads = resolveKvHeadCount(reader, prefix, heads);
    float rmsEps = resolveRmsEps(reader, prefix);
    float ropeTheta = reader.metaFloat(prefix + ".rope.freq_base", 10_000_000f);
    int context = reader.metaInt(prefix + ".context_length", 131_072);
    int convL = reader.metaInt(prefix + ".shortconv.l_cache", 3);
    int vocab = reader.metaStringArray("tokenizer.ggml.tokens").size();
    if (vocab == 0) {
      vocab = reader.metaInt(prefix + ".vocab_size", 0);
    }
    if (hidden <= 0 || layers <= 0 || heads <= 0 || intermediate <= 0 || vocab <= 0) {
      throw new IllegalStateException(
        "incomplete lfm2 GGUF metadata (hidden/layers/heads/ff/vocab)");
    }
    int headDim = hidden / heads;
    List<String> layerTypes = resolveLayerTypes(reader, prefix, layers, kvHeads);
    boolean tie = !reader.hasTensor("output.weight");

    return new Config.HfConfig(
      vocab,
      hidden,
      intermediate,
      layers,
      heads,
      kvHeads > 0 ? kvHeads : heads,
      headDim,
      context,
      rmsEps,
      "silu",
      tie,
      false,
      ropeTheta,
      null,
      "float32",
      "lfm2",
      List.of("Lfm2ForCausalLM"),
      "silu",
      0,
      layerTypes,
      10_000f,
      0f,
      convL,
      false,
      false,
      null
    );
  }

  private static float resolveRmsEps(final GgufReader reader, final String prefix) {
    return Stream.of(
        prefix + ".attention.layer_norm_rms_epsilon",
        prefix + ".attention.layer_norm_rms_eps",
        prefix + ".attention.layernorm_rms_eps")
      .filter(key -> reader.metadata().containsKey(key))
      .map(key -> reader.metaFloat(key, 1e-5f))
      .findFirst()
      .orElse(1e-5f);
  }

  private static int resolveKvHeadCount(
    final GgufReader reader,
    final String prefix,
    final int defaultHeads
  ) {
    List<Number> perLayer = reader.metaNumberArray(prefix + ".attention.head_count_kv");
    if (!perLayer.isEmpty()) {
      return perLayer.stream()
        .mapToInt(Number::intValue)
        .filter(v -> v > 0)
        .max()
        .orElse(defaultHeads);
    }
    return reader.metaInt(prefix + ".attention.head_count_kv", defaultHeads);
  }

  private static List<String> resolveLayerTypes(
    final GgufReader reader,
    final String prefix,
    final int layers,
    final int defaultKvHeads
  ) {
    List<Number> perLayerKv = reader.metaNumberArray(prefix + ".attention.head_count_kv");
    return IntStream.range(0, layers)
      .mapToObj(i -> {
        int kv = defaultKvHeads;
        if (i < perLayerKv.size()) {
          kv = perLayerKv.get(i).intValue();
        } else if (reader.hasTensor("blk." + i + ".shortconv.in_proj.weight")
          || reader.hasTensor("blk." + i + ".shortconv.conv.weight")
          || !reader.hasTensor("blk." + i + ".attn_q.weight")) {
          kv = 0;
        }
        return kv == 0 ? "conv" : "full_attention";
      })
      .toList();
  }

  public record LoadedGguf(Config.HfConfig config, WeightBag weights, GgufReader reader) {
  }
}
