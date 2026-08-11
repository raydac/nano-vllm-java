package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_OUTPUT;
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
      case ARCH_QWEN3 -> qwen3(config);
      case ARCH_LFM2 -> lfm2(config);
      case ARCH_BERT -> bert(config);
      default -> throw new IllegalArgumentException("unsupported architecture '" + arch + "'");
    };
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
