package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.models.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.PACKED_MODULES_MAPPING;
import static com.igormaznitsa.nanollvm.models.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.POST_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.PRE_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.QKV_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.WeightNames.selfAttn;
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
