package com.igormaznitsa.nanollvm.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.EngineIo;
import com.igormaznitsa.nanollvm.models.WeightBag;
import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Loads an LFM2 (or compatible) GGUF file into {@link Config.HfConfig} + float32 {@link WeightBag}.
 */
public final class GgufModelLoader {

  private GgufModelLoader() {
  }

  public static LoadedGguf load(final Path ggufPath, final EngineIo io) throws IOException {
    EngineIo streams = io == null ? EngineIo.silent() : io;
    Path path = requireNonNull(ggufPath, "ggufPath").toAbsolutePath().normalize();
    streams.infof("Loading GGUF from %s%n", path);

    GgufReader reader = GgufReader.open(path);
    String arch = reader.metaString("general.architecture", "").toLowerCase(Locale.ROOT);
    if (!arch.contains("lfm2")) {
      reader.close();
      throw new IllegalArgumentException(
        "unsupported GGUF architecture '" + arch + "' (expected lfm2)");
    }

    Config.HfConfig config = buildConfig(reader, arch);
    streams.infof(
      "GGUF %s: layers=%d hidden=%d ff=%d heads=%d/%d convL=%d%n",
      arch,
      config.numHiddenLayers(),
      config.hiddenSize(),
      config.intermediateSize(),
      config.numAttentionHeads(),
      config.numKeyValueHeads(),
      config.convLCache());

    Map<String, Tensor> weights = new LinkedHashMap<>();
    Progress progress = new Progress("GGUF weights", reader.tensorCount(), streams);
    long loaded = 0L;
    try {
      for (String name : reader.tensorNames()) {
        Tensor tensor = reader.getTensor(name);
        weights.put(name, tensor);
        loaded += (long) tensor.numel() * Float.BYTES;
        progress.step("%s (%.0f MiB fp32)".formatted(name, loaded / (1024.0 * 1024.0)));
      }
      progress.finish("%.0f MiB dequantized to float32".formatted(loaded / (1024.0 * 1024.0)));
    } catch (RuntimeException e) {
      progress.finish("failed");
      reader.close();
      throw e;
    }

    return new LoadedGguf(config, new WeightBag(weights), reader);
  }

  private static Config.HfConfig buildConfig(final GgufReader reader, final String arch) {
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
    if (vocab <= 0) {
      vocab = reader.metaInt(prefix + ".vocab_size", 0);
    }
    if (hidden <= 0 || layers <= 0 || heads <= 0 || intermediate <= 0 || vocab <= 0) {
      throw new IllegalStateException(
        "incomplete LFM2 GGUF metadata (hidden/layers/heads/ff/vocab)");
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
      convL
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

  /**
   * Tiny progress helper (same UX as {@link ModelLoader}).
   */
  private static final class Progress {
    private final String label;
    private final int total;
    private final EngineIo io;
    private int done;

    Progress(final String label, final int total, final EngineIo io) {
      this.label = label;
      this.total = Math.max(1, total);
      this.io = io;
    }

    void step(final String detail) {
      this.done++;
      if (this.done == 1 || this.done == this.total || this.done % 8 == 0) {
        this.io.infof("%s [%d/%d] %s%n", this.label, this.done, this.total, detail);
      }
    }

    void finish(final String detail) {
      this.io.infof("%s done: %s%n", this.label, detail);
    }
  }
}
