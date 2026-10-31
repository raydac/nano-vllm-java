package com.igormaznitsa.nanollvm.models.llmarch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ModelLoaderTest {

  private static final int HIDDEN = 6;
  private static final int INTERMEDIATE = 5;
  private static final int QUERY_ROWS = 4;
  private static final int KEY_ROWS = 2;

  private static WeightBag load(
    final Map<String, Tensor> tensors,
    final Config.HfConfig config,
    final WeightSchema schema
  ) {
    return ModelLoader.assembleFromNamedTensors(tensors, "test", config, schema, null);
  }

  private static Map<String, Tensor> shards(final Tensor query) {
    Map<String, Tensor> tensors = plain(WeightSchema.llama(tinyLlama()));
    tensors.put("model.layers.0.self_attn.q_proj.weight", query);
    tensors.put("model.layers.0.self_attn.k_proj.weight", filled(2f, KEY_ROWS, HIDDEN));
    tensors.put("model.layers.0.self_attn.v_proj.weight", filled(3f, KEY_ROWS, HIDDEN));
    tensors.put("model.layers.0.mlp.gate_proj.weight", filled(4f, INTERMEDIATE, HIDDEN));
    tensors.put("model.layers.0.mlp.up_proj.weight", filled(5f, INTERMEDIATE, HIDDEN));
    return tensors;
  }

  private static Map<String, Tensor> plain(final WeightSchema schema) {
    Map<String, Tensor> tensors = new LinkedHashMap<>();
    for (String name : schema.expectedParameters()) {
      if (name.endsWith(".qkv_proj.weight") || name.endsWith(".gate_up_proj.weight")) {
        continue;
      }
      tensors.put(name, marker(1f, 1));
    }
    return tensors;
  }

  private static Tensor filled(final float value, final int rows, final int cols) {
    float[] data = new float[rows * cols];
    Arrays.fill(data, value);
    return Tensor.of(data, rows, cols);
  }

  private static Tensor marker(final float value, final int length) {
    float[] data = new float[length];
    Arrays.fill(data, value);
    return Tensor.of(data, length);
  }

  private static Config.HfConfig tinyLlama() {
    return Config.HfConfig.parse("""
      {
        "model_type": "llama",
        "hidden_size": 6,
        "intermediate_size": 5,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "max_position_embeddings": 16,
        "tie_word_embeddings": false
      }
      """);
  }

  private static Config.HfConfig tinyQwen35() {
    return Config.HfConfig.parse("""
      {
        "model_type": "qwen3_5",
        "hidden_size": 6,
        "intermediate_size": 5,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "max_position_embeddings": 16,
        "tie_word_embeddings": true,
        "layer_types": ["linear_attention", "full_attention"]
      }
      """);
  }

  @Test
  void wrapperPrefixesMergeIntoConfigShapes() {
    Map<String, Tensor> tensors = plain(WeightSchema.llama(tinyLlama()));
    tensors.remove("lm_head.weight");
    tensors.put("language_model.lm_head.weight", marker(9f, 1));
    tensors.put(
      "base_model.model.language_model.layers.0.self_attn.q_proj.weight",
      filled(1f, QUERY_ROWS, HIDDEN));
    tensors.put("model/layers/0/self_attn/k_proj/weight", filled(2f, KEY_ROWS, HIDDEN));
    tensors.put(
      "language_model.layers.0.self_attn.v_proj.weight",
      filled(3f, KEY_ROWS, HIDDEN));
    tensors.put("model.model.layers.0.mlp.gate_proj.weight", filled(4f, INTERMEDIATE, HIDDEN));
    tensors.put("text_model.layers.0.mlp.up_proj.weight", filled(5f, INTERMEDIATE, HIDDEN));
    tensors.put("model.visual.blocks.0.attn.q_proj.weight", filled(99f, 3, 3));

    WeightBag bag = load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama()));

    Tensor qkv = bag.require("model.layers.0.self_attn.qkv_proj.weight");
    assertEquals(QUERY_ROWS + 2 * KEY_ROWS, qkv.size(0));
    assertEquals(HIDDEN, qkv.size(1));
    assertEquals(1f, qkv.get(0));
    assertEquals(2f, qkv.get(QUERY_ROWS * HIDDEN));
    assertEquals(3f, qkv.get((QUERY_ROWS + KEY_ROWS) * HIDDEN));
    Tensor gateUp = bag.require("model.layers.0.mlp.gate_up_proj.weight");
    assertEquals(INTERMEDIATE * 2, gateUp.size(0));
    assertEquals(4f, gateUp.get(0));
    assertEquals(5f, gateUp.get(INTERMEDIATE * HIDDEN));
    assertEquals(9f, bag.require("lm_head.weight").get(0));
    assertFalse(bag.has("model.visual.blocks.0.attn.q_proj.weight"));
  }

  @Test
  void attentionAndFeedForwardNamesMergeLikeSelfAttnAndMlp() {
    Map<String, Tensor> tensors = plain(WeightSchema.llama(tinyLlama()));
    tensors.put(
      "model.layers.0.attention.q_proj.weight",
      filled(1f, QUERY_ROWS, HIDDEN));
    tensors.put("model.layers.0.attention.k_proj.weight", filled(2f, KEY_ROWS, HIDDEN));
    tensors.put("model.layers.0.attention.v_proj.weight", filled(3f, KEY_ROWS, HIDDEN));
    tensors.put("model.layers.0.feed_forward.gate_proj.weight", filled(4f, INTERMEDIATE, HIDDEN));
    tensors.put("model.layers.0.ffn.up_proj.weight", filled(5f, INTERMEDIATE, HIDDEN));

    WeightBag bag = load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama()));

    assertEquals(1f, bag.require("model.layers.0.self_attn.qkv_proj.weight").get(0));
    assertEquals(4f, bag.require("model.layers.0.mlp.gate_up_proj.weight").get(0));
    assertEquals(5f,
      bag.require("model.layers.0.mlp.gate_up_proj.weight").get(INTERMEDIATE * HIDDEN));
  }

  @Test
  void swappedQueryMatrixIsTurnedIntoOutIn() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS, HIDDEN));
    float[] stored = new float[HIDDEN * QUERY_ROWS];
    for (int i = 0; i < stored.length; i++) {
      stored[i] = i;
    }
    tensors.put(
      "model.layers.0.self_attn.q_proj.weight",
      Tensor.of(stored, HIDDEN, QUERY_ROWS));

    Tensor qkv = load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama()))
      .require("model.layers.0.self_attn.qkv_proj.weight");

    assertEquals(4f, qkv.get(1));
  }

  @Test
  void alreadyFusedGateUpStaysOneMatrix() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS, HIDDEN));
    tensors.remove("model.layers.0.mlp.gate_proj.weight");
    tensors.remove("model.layers.0.mlp.up_proj.weight");
    tensors.put(
      "model.layers.0.mlp.gate_up_proj.weight",
      filled(7.5f, INTERMEDIATE * 2, HIDDEN));

    Tensor gateUp = load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama()))
      .require("model.layers.0.mlp.gate_up_proj.weight");

    assertEquals(7.5f, gateUp.get(0));
    assertEquals(INTERMEDIATE * 2, gateUp.size(0));
  }

  @Test
  void swappedFusedQkvIsTurnedIntoOutIn() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS, HIDDEN));
    tensors.remove("model.layers.0.self_attn.q_proj.weight");
    tensors.remove("model.layers.0.self_attn.k_proj.weight");
    tensors.remove("model.layers.0.self_attn.v_proj.weight");
    int rows = QUERY_ROWS + 2 * KEY_ROWS;
    float[] stored = new float[HIDDEN * rows];
    for (int in = 0; in < HIDDEN; in++) {
      for (int out = 0; out < rows; out++) {
        stored[in * rows + out] = 1000f + in * 10f + out;
      }
    }
    tensors.put("model.layers.0.self_attn.qkv_proj.weight", Tensor.of(stored, HIDDEN, rows));

    Tensor qkv = load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama()))
      .require("model.layers.0.self_attn.qkv_proj.weight");

    assertEquals(rows, qkv.size(0));
    assertEquals(HIDDEN, qkv.size(1));
    assertEquals(1010f, qkv.get(1));
  }

  @Test
  void qwen35KeepsSeparateQueryAndLinearQkv() {
    Config.HfConfig config = tinyQwen35();
    WeightSchema schema = WeightSchema.qwen35(config);
    Map<String, Tensor> tensors = new LinkedHashMap<>();
    for (String name : schema.expectedParameters()) {
      if (name.endsWith(".gate_up_proj.weight")) {
        continue;
      }
      tensors.put(name, marker(name.endsWith("in_proj_qkv.weight") ? 9f : 1f, 1));
    }
    tensors.put("model.layers.1.self_attn.q_proj.weight", marker(3f, 1));
    tensors.put("model.layers.0.mlp.gate_proj.weight", filled(4f, INTERMEDIATE, HIDDEN));
    tensors.put("model.layers.0.mlp.up_proj.weight", filled(5f, INTERMEDIATE, HIDDEN));
    tensors.put("model.layers.1.mlp.gate_proj.weight", filled(4f, INTERMEDIATE, HIDDEN));
    tensors.put("model.layers.1.mlp.up_proj.weight", filled(5f, INTERMEDIATE, HIDDEN));

    WeightBag bag = load(tensors, config, schema);

    assertEquals(3f, bag.require("model.layers.1.self_attn.q_proj.weight").get(0));
    assertEquals(9f, bag.require("model.layers.0.linear_attn.in_proj_qkv.weight").get(0));
    assertFalse(bag.has("model.layers.1.self_attn.qkv_proj.weight"));
    assertTrue(bag.has("model.layers.0.mlp.gate_up_proj.weight"));
  }

  @Test
  void gemma4KeepsUnfusedProjectionsUnderLanguagePrefix() {
    Config.HfConfig config = tinyLlama();
    WeightSchema schema = WeightSchema.gemma4(config);
    Map<String, Tensor> tensors = new LinkedHashMap<>();
    for (String name : schema.expectedParameters()) {
      String stored = name.startsWith("model.")
        ? "model.language_model." + name.substring("model.".length())
        : "language_model." + name;
      tensors.put(stored, marker(name.endsWith("q_proj.weight") ? 3.5f : 1f, 1));
    }

    WeightBag bag = load(tensors, config, schema);

    assertEquals(3.5f, bag.require("model.layers.0.self_attn.q_proj.weight").get(0));
    assertTrue(bag.has("model.layers.0.self_attn.k_proj.weight"));
    assertTrue(bag.has("model.layers.0.mlp.gate_proj.weight"));
    assertTrue(bag.has("lm_head.weight"));
    assertFalse(bag.has("model.layers.0.self_attn.qkv_proj.weight"));
    assertFalse(bag.has("model.layers.0.mlp.gate_up_proj.weight"));
  }

  @Test
  void duplicateTensorFails() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS, HIDDEN));
    tensors.put("language_model.embed_tokens.weight", marker(2f, 1));

    IllegalStateException error = assertThrows(
      IllegalStateException.class,
      () -> load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama())));

    assertTrue(error.getMessage().contains("duplicate weight"));
  }

  @Test
  void duplicateQueryShardFails() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS, HIDDEN));
    tensors.put(
      "language_model.layers.0.self_attn.q_proj.weight",
      filled(8f, QUERY_ROWS, HIDDEN));

    IllegalStateException error = assertThrows(
      IllegalStateException.class,
      () -> load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama())));

    assertTrue(error.getMessage().contains("duplicate qkv shard"));
  }

  @Test
  void fusedAndSplitQkvTogetherFail() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS, HIDDEN));
    tensors.put(
      "model.layers.0.self_attn.qkv_proj.weight",
      filled(1f, QUERY_ROWS + 2 * KEY_ROWS, HIDDEN));

    IllegalStateException error = assertThrows(
      IllegalStateException.class,
      () -> load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama())));

    assertTrue(error.getMessage().contains("duplicate weight"));
  }

  @Test
  void oversizedQueryShardFails() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS + 1, HIDDEN));

    IllegalStateException error = assertThrows(
      IllegalStateException.class,
      () -> load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama())));

    assertTrue(error.getMessage().contains("expected [4, 6]"));
  }

  @Test
  void rank3QueryShardFails() {
    Map<String, Tensor> tensors = shards(Tensor.of(new float[QUERY_ROWS * HIDDEN], 2, 2, HIDDEN));

    assertThrows(
      IllegalStateException.class,
      () -> load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama())));
  }

  @Test
  void missingValueShardFails() {
    Map<String, Tensor> tensors = shards(filled(1f, QUERY_ROWS, HIDDEN));
    tensors.remove("model.layers.0.self_attn.v_proj.weight");

    IllegalStateException error = assertThrows(
      IllegalStateException.class,
      () -> load(tensors, tinyLlama(), WeightSchema.llama(tinyLlama())));

    assertTrue(error.getMessage().contains("incomplete qkv_proj"));
  }

  @Test
  void zeroHeadDimFailsBeforeABagIsReturned() {
    Config.HfConfig config = Config.HfConfig.parse("""
      {
        "model_type": "llama",
        "hidden_size": 6,
        "intermediate_size": 5,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 0,
        "vocab_size": 8,
        "max_position_embeddings": 16,
        "tie_word_embeddings": true
      }
      """);
    Map<String, Tensor> tensors = plain(WeightSchema.llama(config));
    tensors.put("model.layers.0.self_attn.q_proj.weight", filled(1f, 1, HIDDEN));

    IllegalStateException error = assertThrows(
      IllegalStateException.class,
      () -> load(tensors, config, WeightSchema.llama(config)));

    assertTrue(error.getMessage().contains("positive shape"));
  }

  @Test
  void blankNameAndEmptyCatalogFail() {
    Config.HfConfig config = tinyLlama();
    WeightSchema schema = WeightSchema.llama(config);
    assertThrows(
      IllegalArgumentException.class,
      () -> ModelLoader.assembleFromNamedTensors(Map.of(), "empty", config, schema, null));
    Map<String, Tensor> blank = new LinkedHashMap<>();
    blank.put("  ", marker(1f, 1));
    assertThrows(IllegalArgumentException.class, () -> load(blank, config, schema));
    Map<String, Tensor> missing = new LinkedHashMap<>();
    missing.put("model.embed_tokens.weight", null);
    assertThrows(NullPointerException.class, () -> load(missing, config, schema));
  }
}
