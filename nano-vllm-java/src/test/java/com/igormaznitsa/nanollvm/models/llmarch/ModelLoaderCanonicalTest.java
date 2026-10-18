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
  }
}
