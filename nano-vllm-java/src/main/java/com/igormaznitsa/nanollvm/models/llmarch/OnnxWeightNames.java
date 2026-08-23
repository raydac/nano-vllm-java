package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_POSITION_EMBD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD_NORM_BIAS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_TYPES;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ggufBlk;
import static java.util.Locale.ROOT;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Maps ONNX / Hugging Face initializer names onto internal {@link
 * com.igormaznitsa.nanollvm.models.internal.WeightSchema} keys used by causal and BERT bags.
 *
 * @since 1.1.0
 */
public final class OnnxWeightNames {

  private static final Pattern BERT_LAYER = Pattern.compile(
    "^encoder\\.layer\\.(\\d+)\\.(.+)$");

  private OnnxWeightNames() {
  }

  /**
   * Causal decoder HF path cleanup (strip leading noise, collapse {@code model.model.} →
   * {@code model.}).
   *
   * @since 1.1.0
   */
  public static String normalizeChatName(final String rawName) {
    String name = stripNoise(rawName);
    if (name.startsWith("model.model.")) {
      name = name.substring("model.".length());
    }
    return name;
  }

  /**
   * BERT HF names → GGUF-style schema keys ({@code token_embd.weight}, {@code blk.N.…}).
   *
   * @since 1.1.0
   */
  public static String normalizeBertName(final String rawName) {
    String name = stripEncoderPrefix(stripNoise(rawName));
    return switch (name) {
      case "embeddings.word_embeddings.weight" -> GGUF_TOKEN_EMBD;
      case "embeddings.position_embeddings.weight" -> GGUF_POSITION_EMBD;
      case "embeddings.token_type_embeddings.weight" -> GGUF_TOKEN_TYPES;
      case "embeddings.LayerNorm.weight", "embeddings.LayerNorm.gamma" -> GGUF_TOKEN_EMBD_NORM;
      case "embeddings.LayerNorm.bias", "embeddings.LayerNorm.beta" -> GGUF_TOKEN_EMBD_NORM_BIAS;
      default -> mapBertEncoder(name);
    };
  }

  private static String mapBertEncoder(final String name) {
    Matcher m = BERT_LAYER.matcher(name);
    if (!m.matches()) {
      return name;
    }
    int layer = Integer.parseInt(m.group(1));
    String rest = m.group(2);
    String blk = ggufBlk(layer);
    return switch (rest) {
      case "attention.self.query.weight" -> blk + "attn_q.weight";
      case "attention.self.query.bias" -> blk + "attn_q.bias";
      case "attention.self.key.weight" -> blk + "attn_k.weight";
      case "attention.self.key.bias" -> blk + "attn_k.bias";
      case "attention.self.value.weight" -> blk + "attn_v.weight";
      case "attention.self.value.bias" -> blk + "attn_v.bias";
      case "attention.output.dense.weight" -> blk + "attn_output.weight";
      case "attention.output.dense.bias" -> blk + "attn_output.bias";
      case "attention.output.LayerNorm.weight", "attention.output.LayerNorm.gamma" ->
        blk + "attn_output_norm.weight";
      case "attention.output.LayerNorm.bias", "attention.output.LayerNorm.beta" ->
        blk + "attn_output_norm.bias";
      case "intermediate.dense.weight" -> blk + "ffn_up.weight";
      case "intermediate.dense.bias" -> blk + "ffn_up.bias";
      case "output.dense.weight" -> blk + "ffn_down.weight";
      case "output.dense.bias" -> blk + "ffn_down.bias";
      case "output.LayerNorm.weight", "output.LayerNorm.gamma" -> blk + "layer_output_norm.weight";
      case "output.LayerNorm.bias", "output.LayerNorm.beta" -> blk + "layer_output_norm.bias";
      default -> name;
    };
  }

  private static String stripNoise(final String rawName) {
    String name = requireNonBlank(rawName).strip();
    while (name.startsWith("/")) {
      name = name.substring(1);
    }
    if (name.toLowerCase(ROOT).contains("onnx::")) {
      return name;
    }
    return name.replace('/', '.');
  }

  private static String stripEncoderPrefix(final String name) {
    String embeddings = dropPrefixBefore(name, "embeddings.");
    if (embeddings != null) {
      return embeddings;
    }
    String encoder = dropPrefixBefore(name, "encoder.");
    return encoder != null ? encoder : name;
  }

  private static String dropPrefixBefore(final String name, final String root) {
    int nested = name.indexOf("." + root);
    if (nested >= 0) {
      return name.substring(nested + 1);
    }
    return name.startsWith(root) ? name : null;
  }

  private static String requireNonBlank(final String rawName) {
    if (rawName == null || rawName.isBlank()) {
      throw new IllegalArgumentException("ONNX tensor name must not be blank");
    }
    return rawName;
  }
}
