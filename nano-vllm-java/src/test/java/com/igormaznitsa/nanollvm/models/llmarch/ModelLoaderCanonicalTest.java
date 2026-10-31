package com.igormaznitsa.nanollvm.models.llmarch;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class ModelLoaderCanonicalTest {

  @Test
  void languageTowerPrefixBecomesTextName() {
    assertEquals(
      "model.layers.0.self_attn.q_proj.weight",
      ModelLoader.canonicalWeightName(
        "model.language_model.layers.0.self_attn.q_proj.weight"));
    assertEquals(
      "model.embed_tokens.weight",
      ModelLoader.canonicalWeightName("model.embed_tokens.weight"));
    assertEquals(
      "model.layers.0.linear_attn.in_proj_qkv.weight",
      ModelLoader.canonicalWeightName(
        "language_model.layers.0.linear_attn.in_proj_qkv.weight"));
    assertEquals(
      "model.embed_tokens.weight",
      ModelLoader.canonicalWeightName("language_model.embed_tokens.weight"));
    assertEquals(
      "model.layers.0.self_attn.q_proj.weight",
      ModelLoader.canonicalWeightName(
        "base_model.model.model.layers.0.self_attn.q_proj.weight"));
    assertEquals(
      "model.layers.0.self_attn.q_proj.weight",
      ModelLoader.canonicalWeightName(
        "model/language_model/layers/0/self_attn/q_proj/weight"));
    assertEquals(
      "model.embed_tokens.weight",
      ModelLoader.canonicalWeightName("text_model.embed_tokens.weight"));
    assertEquals(
      "lm_head.weight",
      ModelLoader.canonicalWeightName("language_model.lm_head.weight"));
    assertEquals(
      "lm_head.weight",
      ModelLoader.canonicalWeightName("model.lm_head.weight"));
    assertEquals(
      "model.layers.0.self_attn.q_proj.weight",
      ModelLoader.canonicalWeightName(
        ModelLoader.canonicalWeightName(
          "model.language_model.layers.0.self_attn.q_proj.weight")));
  }
}
