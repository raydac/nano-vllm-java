package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SamplingParamsTest {

  @Test
  void builderDefaultsMatchNeutral() {
    SamplingParams params = SamplingParams.builder().build();
    assertEquals(SamplingDefaults.DEFAULT_TEMPERATURE, params.temperature());
    assertEquals(SamplingDefaults.DEFAULT_MAX_TOKENS, params.maxTokens());
    assertFalse(params.ignoreEos());
    assertEquals(0, params.topK());
    assertEquals(SamplingDefaults.DEFAULT_TOP_P, params.topP());
    assertEquals(params, SamplingDefaults.neutral());
  }

  @Test
  void builderAndWithersCopyOtherFields() {
    SamplingParams base = SamplingParams.builder()
      .temperature(0.2f)
      .maxTokens(128)
      .ignoreEos(true)
      .topK(64)
      .topP(0.8f)
      .build();
    assertEquals(0.2f, base.temperature());
    assertEquals(128, base.maxTokens());
    assertTrue(base.ignoreEos());
    assertEquals(64, base.topK());
    assertEquals(0.8f, base.topP());

    assertEquals(0.1f, base.withTemperature(0.1f).temperature());
    assertEquals(128, base.withTemperature(0.1f).maxTokens());
    assertEquals(32, base.withMaxTokens(32).maxTokens());
    assertEquals(0.2f, base.withMaxTokens(32).temperature());
    assertFalse(base.withIgnoreEos(false).ignoreEos());
    assertEquals(8, base.withTopK(8).topK());
    assertEquals(0.5f, base.withTopP(0.5f).topP());
  }

  @Test
  void builderValidates() {
    assertThrows(IllegalArgumentException.class,
      () -> SamplingParams.builder().temperature(0f).build());
    assertThrows(IllegalArgumentException.class,
      () -> SamplingParams.builder().maxTokens(0).build());
    assertThrows(IllegalArgumentException.class,
      () -> SamplingParams.builder().topK(-1).build());
    assertThrows(IllegalArgumentException.class,
      () -> SamplingParams.builder().topP(0f).build());
    assertThrows(IllegalArgumentException.class,
      () -> SamplingParams.builder().topP(1.1f).build());
  }

  @Test
  void forTokenizerIsNeutralAlias() {
    assertEquals(SamplingDefaults.neutral(), SamplingDefaults.forTokenizer(null));
    assertEquals(SamplingDefaults.neutral(100), SamplingDefaults.forTokenizer(null, 100));
  }
}
