package com.igormaznitsa.nanollvm.models.internal;

import java.util.Map;

/**
 * Shared HF weight-path fragments, architecture ids, and packed-module rewrite rules.
 */
public final class WeightNames {

  public static final String ARCH_GEMMA3 = "gemma3";
  public static final String ARCH_QWEN3 = "qwen3";
  public static final String ARCH_LFM2 = "lfm2";
  /**
   * @since 1.1.0
   */
  public static final String ARCH_LLAMA = "llama";
  /**
   * @since 1.1.0
   */
  public static final String ARCH_BERT = "bert";

  public static final String EMBED_TOKENS = "model.embed_tokens.weight";
  public static final String MODEL_NORM = "model.norm.weight";
  public static final String LM_HEAD = "lm_head.weight";

  public static final String GGUF_TOKEN_EMBD = "token_embd.weight";
  public static final String GGUF_TOKEN_EMBD_NORM = "token_embd_norm.weight";
  /**
   * @since 1.1.0
   */
  public static final String GGUF_TOKEN_EMBD_NORM_BIAS = "token_embd_norm.bias";
  /**
   * @since 1.1.0
   */
  public static final String GGUF_POSITION_EMBD = "position_embd.weight";
  /**
   * @since 1.1.0
   */
  public static final String GGUF_TOKEN_TYPES = "token_types.weight";
  public static final String GGUF_OUTPUT = "output.weight";

  public static final String QKV_PROJ = "qkv_proj";
  public static final String GATE_UP_PROJ = "gate_up_proj";

  public static final String INPUT_LAYERNORM = "input_layernorm.weight";
  public static final String POST_ATTENTION_LAYERNORM = "post_attention_layernorm.weight";
  public static final String PRE_FEEDFORWARD_LAYERNORM = "pre_feedforward_layernorm.weight";
  public static final String POST_FEEDFORWARD_LAYERNORM = "post_feedforward_layernorm.weight";

  public static final String QKV_PROJ_WEIGHT = "qkv_proj.weight";
  public static final String O_PROJ_WEIGHT = "o_proj.weight";
  public static final String Q_NORM_WEIGHT = "q_norm.weight";
  public static final String K_NORM_WEIGHT = "k_norm.weight";
  public static final String GATE_UP_PROJ_WEIGHT = "gate_up_proj.weight";
  public static final String DOWN_PROJ_WEIGHT = "down_proj.weight";

  public static final Map<String, Object[]> PACKED_MODULES_MAPPING = Map.of(
    "q_proj", new Object[] {QKV_PROJ, "q"},
    "k_proj", new Object[] {QKV_PROJ, "k"},
    "v_proj", new Object[] {QKV_PROJ, "v"},
    "gate_proj", new Object[] {GATE_UP_PROJ, 0},
    "up_proj", new Object[] {GATE_UP_PROJ, 1}
  );

  private WeightNames() {
  }

  public static String layer(final int layerIndex) {
    return "model.layers." + layerIndex + ".";
  }

  public static String selfAttn(final int layerIndex) {
    return layer(layerIndex) + "self_attn.";
  }

  public static String mlp(final int layerIndex) {
    return layer(layerIndex) + "mlp.";
  }

  public static String ggufBlk(final int layerIndex) {
    return "blk." + layerIndex + ".";
  }
}
