package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA4;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_WHISPER;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_OUTPUT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_OUTPUT_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_POSITION_EMBD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD_NORM_BIAS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_TYPES;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.PACKED_MODULES_MAPPING;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.PRE_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.QKV_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ggufBlk;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.selfAttn;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.Config;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Expected / optional HF weight names and packed-module rewrite rules for one architecture.
 */
public final class WeightSchema {

  private final Map<String, Object[]> packedModulesMapping;
  private final Set<String> expectedParameters;
  private final Set<String> optionalParameters;

  private WeightSchema(
    final Map<String, Object[]> packedModulesMapping,
    final Set<String> expectedParameters,
    final Set<String> optionalParameters
  ) {
    this.packedModulesMapping = Map.copyOf(requireNonNull(packedModulesMapping));
    this.expectedParameters = Set.copyOf(requireNonNull(expectedParameters));
    this.optionalParameters = Set.copyOf(requireNonNull(optionalParameters));
  }

  public static WeightSchema forArchitecture(final String arch, final Config.HfConfig config) {
    return switch (arch) {
      case ARCH_GEMMA3 -> gemma3(config);
      case ARCH_GEMMA4 -> gemma4(config);
      case ARCH_QWEN3 -> qwen3(config);
      case ARCH_LLAMA -> llama(config);
      case ARCH_LFM2 -> lfm2(config);
      case ARCH_BERT -> bert(config);
      case ARCH_WHISPER -> whisper(config);
      default -> throw new IllegalArgumentException("unsupported architecture '" + arch + "'");
    };
  }

  public static WeightSchema forGguf(final String arch, final Config.HfConfig config) {
    return switch (arch) {
      case ARCH_QWEN3 -> qwen3Gguf(config);
      case ARCH_LFM2 -> lfm2(config);
      case ARCH_BERT -> bert(config);
      default -> throw new IllegalArgumentException("unsupported GGUF architecture '" + arch + "'");
    };
  }

  /**
   * {@code llama} architecture schema (RMSNorm, RoPE, GQA, SiLU MLP; no Q/K head norms).
   *
   * @since 1.1.0
   */
  public static WeightSchema llama(final Config.HfConfig config) {
    Set<String> expected = new LinkedHashSet<>();
    Set<String> optional = new LinkedHashSet<>();
    expected.add(EMBED_TOKENS);
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String p = layer(i);
      String attn = selfAttn(i);
      String mlpPrefix = mlp(i);
      expected.add(p + INPUT_LAYERNORM);
      expected.add(p + POST_ATTENTION_LAYERNORM);
      expected.add(attn + QKV_PROJ_WEIGHT);
      expected.add(attn + O_PROJ_WEIGHT);
      expected.add(mlpPrefix + GATE_UP_PROJ_WEIGHT);
      expected.add(mlpPrefix + DOWN_PROJ_WEIGHT);
    }
    expected.add(MODEL_NORM);
    if (config.tieWordEmbeddings()) {
      optional.add(LM_HEAD);
    } else {
      expected.add(LM_HEAD);
    }
    return new WeightSchema(PACKED_MODULES_MAPPING, expected, optional);
  }

  public static WeightSchema qwen3(final Config.HfConfig config) {
    Set<String> expected = new LinkedHashSet<>();
    Set<String> optional = new LinkedHashSet<>();
    expected.add(EMBED_TOKENS);
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String p = layer(i);
      String attn = selfAttn(i);
      String mlpPrefix = mlp(i);
      expected.add(p + INPUT_LAYERNORM);
      expected.add(p + POST_ATTENTION_LAYERNORM);
      expected.add(attn + QKV_PROJ_WEIGHT);
      expected.add(attn + O_PROJ_WEIGHT);
      if (!config.attentionBias()) {
        expected.add(attn + Q_NORM_WEIGHT);
        expected.add(attn + K_NORM_WEIGHT);
      }
      expected.add(mlpPrefix + GATE_UP_PROJ_WEIGHT);
      expected.add(mlpPrefix + DOWN_PROJ_WEIGHT);
    }
    expected.add(MODEL_NORM);
    if (config.tieWordEmbeddings()) {
      optional.add(LM_HEAD);
    } else {
      expected.add(LM_HEAD);
    }
    return new WeightSchema(PACKED_MODULES_MAPPING, expected, optional);
  }

  public static WeightSchema qwen3Gguf(final Config.HfConfig config) {
    Set<String> expected = new LinkedHashSet<>();
    Set<String> optional = new LinkedHashSet<>();
    expected.add(GGUF_TOKEN_EMBD);
    expected.add(GGUF_OUTPUT_NORM);
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String blk = ggufBlk(i);
      expected.add(blk + "attn_norm.weight");
      expected.add(blk + "ffn_norm.weight");
      expected.add(blk + "attn_q.weight");
      expected.add(blk + "attn_k.weight");
      expected.add(blk + "attn_v.weight");
      expected.add(blk + "attn_output.weight");
      if (!config.attentionBias()) {
        expected.add(blk + "attn_q_norm.weight");
        expected.add(blk + "attn_k_norm.weight");
      }
      expected.add(blk + "ffn_gate.weight");
      expected.add(blk + "ffn_up.weight");
      expected.add(blk + "ffn_down.weight");
    }
    if (config.tieWordEmbeddings()) {
      optional.add(GGUF_OUTPUT);
    } else {
      expected.add(GGUF_OUTPUT);
    }
    return new WeightSchema(Map.of(), expected, optional);
  }

  public static WeightSchema gemma3(final Config.HfConfig config) {
    Set<String> expected = new LinkedHashSet<>();
    Set<String> optional = new LinkedHashSet<>();
    expected.add(EMBED_TOKENS);
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String p = layer(i);
      String attn = selfAttn(i);
      String mlpPrefix = mlp(i);
      expected.add(p + INPUT_LAYERNORM);
      expected.add(p + POST_ATTENTION_LAYERNORM);
      expected.add(p + PRE_FEEDFORWARD_LAYERNORM);
      expected.add(p + POST_FEEDFORWARD_LAYERNORM);
      expected.add(attn + QKV_PROJ_WEIGHT);
      expected.add(attn + O_PROJ_WEIGHT);
      expected.add(attn + Q_NORM_WEIGHT);
      expected.add(attn + K_NORM_WEIGHT);
      expected.add(mlpPrefix + GATE_UP_PROJ_WEIGHT);
      expected.add(mlpPrefix + DOWN_PROJ_WEIGHT);
    }
    expected.add(MODEL_NORM);
    optional.add(LM_HEAD);
    return new WeightSchema(PACKED_MODULES_MAPPING, expected, optional);
  }

  public static WeightSchema gemma4(final Config.HfConfig config) {
    Set<String> expected = new LinkedHashSet<>();
    Set<String> optional = new LinkedHashSet<>();
    expected.add(EMBED_TOKENS);
    expected.add("model.embed_tokens_per_layer.weight");
    expected.add("model.per_layer_model_projection.weight");
    expected.add("model.per_layer_projection_norm.weight");
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String p = layer(i);
      String attn = selfAttn(i);
      String mlpPrefix = mlp(i);
      expected.add(p + INPUT_LAYERNORM);
      expected.add(p + POST_ATTENTION_LAYERNORM);
      expected.add(p + PRE_FEEDFORWARD_LAYERNORM);
      expected.add(p + POST_FEEDFORWARD_LAYERNORM);
      expected.add(p + "post_per_layer_input_norm.weight");
      expected.add(p + "layer_scalar");
      expected.add(p + "per_layer_input_gate.weight");
      expected.add(p + "per_layer_projection.weight");
      expected.add(attn + "q_proj.weight");
      expected.add(attn + O_PROJ_WEIGHT);
      expected.add(attn + Q_NORM_WEIGHT);
      expected.add(mlpPrefix + "gate_proj.weight");
      expected.add(mlpPrefix + "up_proj.weight");
      expected.add(mlpPrefix + DOWN_PROJ_WEIGHT);
      if (config.isKvSharedLayer(i)) {
        optional.add(attn + "k_proj.weight");
        optional.add(attn + "v_proj.weight");
        optional.add(attn + K_NORM_WEIGHT);
      } else {
        expected.add(attn + "k_proj.weight");
        expected.add(attn + "v_proj.weight");
        expected.add(attn + K_NORM_WEIGHT);
      }
    }
    expected.add(MODEL_NORM);
    expected.add(LM_HEAD);
    return new WeightSchema(Map.of(), expected, optional);
  }

  public static WeightSchema lfm2(final Config.HfConfig config) {
    Set<String> expected = new LinkedHashSet<>();
    Set<String> optional = new LinkedHashSet<>();
    expected.add(GGUF_TOKEN_EMBD);
    expected.add(GGUF_TOKEN_EMBD_NORM);
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String blk = ggufBlk(i);
      expected.add(blk + "attn_norm.weight");
      expected.add(blk + "ffn_norm.weight");
      expected.add(blk + "ffn_gate.weight");
      expected.add(blk + "ffn_up.weight");
      expected.add(blk + "ffn_down.weight");
      if (config.isConvLayer(i)) {
        expected.add(blk + "shortconv.conv.weight");
        expected.add(blk + "shortconv.in_proj.weight");
        expected.add(blk + "shortconv.out_proj.weight");
      } else {
        expected.add(blk + "attn_q.weight");
        expected.add(blk + "attn_k.weight");
        expected.add(blk + "attn_v.weight");
        expected.add(blk + "attn_output.weight");
        expected.add(blk + "attn_q_norm.weight");
        expected.add(blk + "attn_k_norm.weight");
      }
    }
    if (config.tieWordEmbeddings()) {
      optional.add(GGUF_OUTPUT);
    } else {
      expected.add(GGUF_OUTPUT);
    }
    return new WeightSchema(Map.of(), expected, optional);
  }

  /**
   * {@code bert} embedding encoder schema (token/position/type embeddings + post-LN blocks).
   *
   * @since 1.1.0
   */
  public static WeightSchema bert(final Config.HfConfig config) {
    Set<String> expected = new LinkedHashSet<>();
    expected.add(GGUF_TOKEN_EMBD);
    expected.add(GGUF_TOKEN_EMBD_NORM);
    expected.add(GGUF_TOKEN_EMBD_NORM_BIAS);
    expected.add(GGUF_POSITION_EMBD);
    expected.add(GGUF_TOKEN_TYPES);
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String blk = ggufBlk(i);
      expected.add(blk + "attn_q.weight");
      expected.add(blk + "attn_q.bias");
      expected.add(blk + "attn_k.weight");
      expected.add(blk + "attn_k.bias");
      expected.add(blk + "attn_v.weight");
      expected.add(blk + "attn_v.bias");
      expected.add(blk + "attn_output.weight");
      expected.add(blk + "attn_output.bias");
      expected.add(blk + "attn_output_norm.weight");
      expected.add(blk + "attn_output_norm.bias");
      expected.add(blk + "ffn_up.weight");
      expected.add(blk + "ffn_up.bias");
      expected.add(blk + "ffn_down.weight");
      expected.add(blk + "ffn_down.bias");
      expected.add(blk + "layer_output_norm.weight");
      expected.add(blk + "layer_output_norm.bias");
    }
    return new WeightSchema(Map.of(), expected, Set.of());
  }

  /**
   * OpenAI Whisper encoder-decoder ASR (Hugging Face {@code model.encoder.*} / {@code model.decoder.*}).
   *
   * @since 1.3.0
   */
  public static WeightSchema whisper(final Config.HfConfig config) {
    Config.WhisperSpec spec = requireNonNull(config.whisper(), "whisper spec");
    Set<String> expected = new LinkedHashSet<>();
    Set<String> optional = new LinkedHashSet<>();
    expected.add("model.encoder.conv1.weight");
    expected.add("model.encoder.conv1.bias");
    expected.add("model.encoder.conv2.weight");
    expected.add("model.encoder.conv2.bias");
    expected.add("model.encoder.embed_positions.weight");
    expected.add("model.encoder.layer_norm.weight");
    expected.add("model.encoder.layer_norm.bias");
    expected.add("model.decoder.embed_tokens.weight");
    expected.add("model.decoder.embed_positions.weight");
    expected.add("model.decoder.layer_norm.weight");
    expected.add("model.decoder.layer_norm.bias");
    optional.add("proj_out.weight");
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      addWhisperAttn(expected, optional, "model.encoder.layers." + i + ".self_attn");
      addWhisperMlp(expected, "model.encoder.layers." + i);
    }
    for (int i = 0; i < spec.decoderLayers(); i++) {
      addWhisperAttn(expected, optional, "model.decoder.layers." + i + ".self_attn");
      addWhisperAttn(expected, optional, "model.decoder.layers." + i + ".encoder_attn");
      addWhisperMlp(expected, "model.decoder.layers." + i);
    }
    return new WeightSchema(Map.of(), expected, optional);
  }

  private static void addWhisperAttn(
    final Set<String> expected,
    final Set<String> optional,
    final String prefix
  ) {
    expected.add(prefix + ".q_proj.weight");
    expected.add(prefix + ".q_proj.bias");
    expected.add(prefix + ".k_proj.weight");
    optional.add(prefix + ".k_proj.bias");
    expected.add(prefix + ".v_proj.weight");
    expected.add(prefix + ".v_proj.bias");
    expected.add(prefix + ".out_proj.weight");
    expected.add(prefix + ".out_proj.bias");
  }

  private static void addWhisperMlp(final Set<String> expected, final String prefix) {
    expected.add(prefix + ".self_attn_layer_norm.weight");
    expected.add(prefix + ".self_attn_layer_norm.bias");
    expected.add(prefix + ".fc1.weight");
    expected.add(prefix + ".fc1.bias");
    expected.add(prefix + ".fc2.weight");
    expected.add(prefix + ".fc2.bias");
    expected.add(prefix + ".final_layer_norm.weight");
    expected.add(prefix + ".final_layer_norm.bias");
    if (prefix.contains("decoder.layers")) {
      expected.add(prefix + ".encoder_attn_layer_norm.weight");
      expected.add(prefix + ".encoder_attn_layer_norm.bias");
    }
  }

  public Map<String, Object[]> packedModulesMapping() {
    return this.packedModulesMapping;
  }

  public boolean accepts(final String paramName) {
    return this.expectedParameters.contains(paramName)
      || this.optionalParameters.contains(paramName);
  }

  public boolean expects(final String paramName) {
    return this.expectedParameters.contains(paramName);
  }

  public Set<String> expectedParameters() {
    return this.expectedParameters;
  }

  public Set<String> optionalParameters() {
    return this.optionalParameters;
  }
}
