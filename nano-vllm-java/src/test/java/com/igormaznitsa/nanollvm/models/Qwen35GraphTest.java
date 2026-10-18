package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.linearAttn;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.selfAttn;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.engine.DeltaNetStateArena;
import com.igormaznitsa.nanollvm.engine.KvCacheArena;
import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.Qwen35ForCausalLM;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class Qwen35GraphTest {

  private static float[] logits(
    final Qwen35ForCausalLM graph,
    final Context ctx,
    final float[] tokenIds,
    final float[] positions,
    final int seqId
  ) {
    int n = tokenIds.length;
    int[] seqIds = new int[n];
    Arrays.fill(seqIds, seqId);
    ctx.set(true, new int[] {0, n}, new int[] {0, n}, n, n, null, null, null, seqIds);
    Tensor hidden = graph.forward(Tensor.of(tokenIds, n), Tensor.of(positions, n), ctx);
    return graph.computeLogits(hidden, ctx).toFloatArray();
  }

  private static boolean finite(final float[] values) {
    if (values.length != 16) {
      return false;
    }
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static WeightBag weights() {
    Map<String, Tensor> bag = new LinkedHashMap<>();
    bag.put(EMBED_TOKENS, constant(0.02f, 16, 8));
    bag.put(MODEL_NORM, constant(0.02f, 8));
    putLayerNorms(bag, 0);
    putMlp(bag, 0);
    String linear = linearAttn(0);
    bag.put(linear + "in_proj_qkv.weight", constant(0.02f, 12, 8));
    bag.put(linear + "in_proj_z.weight", constant(0.02f, 4, 8));
    bag.put(linear + "in_proj_b.weight", constant(0.02f, 2, 8));
    bag.put(linear + "in_proj_a.weight", constant(0.02f, 2, 8));
    bag.put(linear + "conv1d.weight", constant(0.02f, 12, 2));
    bag.put(linear + "A_log", constant(0f, 2));
    bag.put(linear + "dt_bias", constant(0f, 2));
    bag.put(linear + "norm.weight", constant(1f, 2));
    bag.put(linear + "out_proj.weight", constant(0.02f, 8, 4));
    putLayerNorms(bag, 1);
    putMlp(bag, 1);
    String attn = selfAttn(1);
    bag.put(attn + "q_proj.weight", constant(0.02f, 16, 8));
    bag.put(attn + "k_proj.weight", constant(0.02f, 4, 8));
    bag.put(attn + "v_proj.weight", constant(0.02f, 4, 8));
    bag.put(attn + O_PROJ_WEIGHT, constant(0.02f, 8, 8));
    bag.put(attn + Q_NORM_WEIGHT, constant(0f, 4));
    bag.put(attn + K_NORM_WEIGHT, constant(0f, 4));
    return new WeightBag(bag);
  }

  private static void putLayerNorms(final Map<String, Tensor> bag, final int index) {
    bag.put(layer(index) + INPUT_LAYERNORM, constant(0f, 8));
    bag.put(layer(index) + POST_ATTENTION_LAYERNORM, constant(0f, 8));
  }

  private static void putMlp(final Map<String, Tensor> bag, final int index) {
    bag.put(mlp(index) + GATE_UP_PROJ_WEIGHT, constant(0.02f, 16, 8));
    bag.put(mlp(index) + DOWN_PROJ_WEIGHT, constant(0.02f, 8, 8));
  }

  private static Tensor constant(final float value, final int... shape) {
    int n = 1;
    for (int dim : shape) {
      n *= dim;
    }
    float[] data = new float[n];
    Arrays.fill(data, value);
    return Tensor.of(data, shape);
  }

  @Test
  void tinyHybridGraphProducesFiniteLogits() {
    Config.HfConfig config = Config.HfConfig.parse("""
      {
        "model_type": "qwen3_5_text",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "vocab_size": 16,
        "max_position_embeddings": 32,
        "tie_word_embeddings": true,
        "partial_rotary_factor": 0.5,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 2,
        "layer_types": ["linear_attention", "full_attention"]
      }
      """);
    Qwen35ForCausalLM graph = new Qwen35ForCausalLM(config, weights());
    assertEquals(WeightNames.ARCH_QWEN3_5, graph.architectureName());
    assertEquals(1, graph.attentionLayers().size());

    Context ctx = new Context();
    try (
      KvCacheArena kv = new KvCacheArena(1, 16, 1, new int[] {4, 4}, new boolean[] {false, true});
      DeltaNetStateArena delta = new DeltaNetStateArena(
        new boolean[] {true, false}, config.qwen35().convChannels(), 2,
        config.qwen35().recurrentFloats())) {
      ctx.bindKvCache(kv);
      ctx.bindDeltaCache(delta);
      assertTrue(finite(logits(graph, ctx, new float[] {1f, 2f}, new float[] {0f, 1f}, 7)));
      assertTrue(finite(logits(graph, ctx, new float[] {3f}, new float[] {2f}, 7)));
    } finally {
      ctx.clear();
    }
  }
}
