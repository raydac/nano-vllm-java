package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_OUTPUT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Maps GGUF metadata keys onto {@link Config.HfConfig} for architectures this library can run.
 *
 * @since 1.1.0
 */
final class GgufConfigs {

  private GgufConfigs() {
  }

  /**
   * Qwen3 GGUF {@code general.architecture=qwen3} → Hugging Face-shaped config.
   *
   * @since 1.1.0
   */
  static Config.HfConfig qwen3(final ContainerCatalog catalog) {
    requireNonNull(catalog, "catalog");
    String prefix = metadataPrefix(catalog, "qwen3");
    int hidden = catalog.metaInt(prefix + ".embedding_length", 0);
    int layers = catalog.metaInt(prefix + ".block_count", 0);
    int intermediate = catalog.metaInt(prefix + ".feed_forward_length", 0);
    int heads = catalog.metaInt(prefix + ".attention.head_count", 0);
    int kvHeads = catalog.metaInt(prefix + ".attention.head_count_kv", heads);
    int headDim = resolveQwen3HeadDim(catalog, prefix);
    float rmsEps = catalog.metaFloat(prefix + ".attention.layer_norm_rms_epsilon", 1e-6f);
    float ropeTheta = catalog.metaFloat(prefix + ".rope.freq_base", 1_000_000f);
    int context = catalog.metaInt(prefix + ".context_length", 32_768);
    int vocab = resolveVocab(catalog, prefix);
    if (hidden <= 0 || layers <= 0 || heads <= 0 || intermediate <= 0 || vocab <= 0
      || headDim <= 0) {
      throw new IllegalStateException(
        "incomplete qwen3 GGUF metadata (hidden/layers/heads/ff/vocab/head_dim)");
    }
    boolean tie = !catalog.hasTensor(GGUF_OUTPUT);
    boolean attnBias = catalog.hasTensor("blk.0.attn_q.bias");
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
      attnBias,
      ropeTheta,
      null,
      "float32",
      ARCH_QWEN3,
      List.of("Qwen3ForCausalLM"),
      "silu",
      0,
      List.of(),
      10_000f,
      0f,
      0,
      false,
      false,
      false,
      false,
      false,
      null,
      null
    );
  }

  /**
   * LFM2 GGUF {@code general.architecture=lfm2} → Hugging Face-shaped config.
   *
   * @since 1.1.0
   */
  static Config.HfConfig lfm2(final ContainerCatalog catalog) {
    requireNonNull(catalog, "catalog");
    String arch = catalog.architectureHint();
    String prefix = arch.contains("lfm2moe") ? "lfm2moe" : "lfm2";
    int hidden = catalog.metaInt(prefix + ".embedding_length", 0);
    int layers = catalog.metaInt(prefix + ".block_count", 0);
    int intermediate = catalog.metaInt(prefix + ".feed_forward_length", 0);
    int heads = catalog.metaInt(prefix + ".attention.head_count", 0);
    int kvHeads = resolveKvHeadCount(catalog, prefix, heads);
    float rmsEps = resolveRmsEps(catalog, prefix);
    float ropeTheta = catalog.metaFloat(prefix + ".rope.freq_base", 10_000_000f);
    int context = catalog.metaInt(prefix + ".context_length", 131_072);
    int convL = catalog.metaInt(prefix + ".shortconv.l_cache", 3);
    int vocab = resolveVocab(catalog, prefix);
    if (hidden <= 0 || layers <= 0 || heads <= 0 || intermediate <= 0 || vocab <= 0) {
      throw new IllegalStateException(
        "incomplete lfm2 GGUF metadata (hidden/layers/heads/ff/vocab)");
    }
    int headDim = hidden / heads;
    List<String> layerTypes = resolveLayerTypes(catalog, prefix, layers, kvHeads);
    boolean tie = !catalog.hasTensor(GGUF_OUTPUT);
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
      ARCH_LFM2,
      List.of("Lfm2ForCausalLM"),
      "silu",
      0,
      layerTypes,
      10_000f,
      0f,
      convL,
      false,
      false,
      false,
      false,
      false,
      null,
      null
    );
  }

  /**
   * BERT GGUF {@code general.architecture=bert} → Hugging Face-shaped config.
   *
   * @since 1.1.0
   */
  static Config.HfConfig bert(final ContainerCatalog catalog) {
    requireNonNull(catalog, "catalog");
    String prefix = "bert";
    int hidden = catalog.metaInt(prefix + ".embedding_length", 0);
    int layers = catalog.metaInt(prefix + ".block_count", 0);
    int intermediate = catalog.metaInt(prefix + ".feed_forward_length", 0);
    int heads = catalog.metaInt(prefix + ".attention.head_count", 0);
    float normEps = catalog.metaFloat(prefix + ".attention.layer_norm_epsilon", 1e-12f);
    int context = catalog.metaInt(prefix + ".context_length", 512);
    int vocab = resolveVocab(catalog, prefix);
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
      ARCH_BERT,
      List.of("BertForEmbedding"),
      "gelu",
      0,
      List.of(),
      10_000f,
      0f,
      0,
      false,
      false,
      false,
      false,
      false,
      null,
      null
    );
  }

  private static String metadataPrefix(final ContainerCatalog catalog, final String canonical) {
    String hint = catalog.architectureHint();
    if (!hint.isBlank() && catalog.hasMeta(hint + ".embedding_length")) {
      return hint;
    }
    return canonical;
  }

  private static int resolveQwen3HeadDim(final ContainerCatalog catalog, final String prefix) {
    int keyLength = catalog.metaInt(prefix + ".attention.key_length", 0);
    if (keyLength > 0) {
      return keyLength;
    }
    int valueLength = catalog.metaInt(prefix + ".attention.value_length", 0);
    if (valueLength > 0) {
      return valueLength;
    }
    return catalog.metaInt(prefix + ".rope.dimension_count", 0);
  }

  private static int resolveVocab(final ContainerCatalog catalog, final String prefix) {
    int fromTokens = catalog.metaStringArray("tokenizer.ggml.tokens").size();
    return fromTokens > 0 ? fromTokens : catalog.metaInt(prefix + ".vocab_size", 0);
  }

  private static float resolveRmsEps(final ContainerCatalog catalog, final String prefix) {
    return Stream.of(
        prefix + ".attention.layer_norm_rms_epsilon",
        prefix + ".attention.layer_norm_rms_eps",
        prefix + ".attention.layernorm_rms_eps")
      .filter(catalog::hasMeta)
      .map(key -> catalog.metaFloat(key, 1e-5f))
      .findFirst()
      .orElse(1e-5f);
  }

  private static int resolveKvHeadCount(
    final ContainerCatalog catalog,
    final String prefix,
    final int defaultHeads
  ) {
    List<Number> perLayer = catalog.metaNumberArray(prefix + ".attention.head_count_kv");
    if (!perLayer.isEmpty()) {
      return perLayer.stream()
        .mapToInt(Number::intValue)
        .filter(v -> v > 0)
        .max()
        .orElse(defaultHeads);
    }
    return catalog.metaInt(prefix + ".attention.head_count_kv", defaultHeads);
  }

  private static List<String> resolveLayerTypes(
    final ContainerCatalog catalog,
    final String prefix,
    final int layers,
    final int defaultKvHeads
  ) {
    List<Number> perLayerKv = catalog.metaNumberArray(prefix + ".attention.head_count_kv");
    return IntStream.range(0, layers)
      .mapToObj(i -> {
        int kv = defaultKvHeads;
        if (i < perLayerKv.size()) {
          kv = perLayerKv.get(i).intValue();
        } else if (catalog.hasTensor("blk." + i + ".shortconv.in_proj.weight")
          || catalog.hasTensor("blk." + i + ".shortconv.conv.weight")
          || !catalog.hasTensor("blk." + i + ".attn_q.weight")) {
          kv = 0;
        }
        return kv == 0 ? "conv" : "full_attention";
      })
      .toList();
  }
}
