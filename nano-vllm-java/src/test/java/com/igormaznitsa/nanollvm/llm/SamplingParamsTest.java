package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
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

  @Test
  void deterministicKeepsArgmaxKnobs() {
    SamplingParams params = SamplingParams.deterministic(64);
    assertTrue(params.isDeterministic());
    assertEquals(1, params.topK());
    assertEquals(1f, params.topP());
    assertEquals(64, params.maxTokens());
    assertEquals(SamplingDefaults.DEFAULT_TEMPERATURE, params.temperature());
    assertEquals(params, SamplingDefaults.deterministic(64));
    assertEquals(SamplingParams.deterministic(), SamplingDefaults.deterministic());

    SamplingParams hotter = SamplingParams.builder()
      .temperature(0.9f)
      .maxTokens(32)
      .topK(64)
      .topP(0.8f)
      .ignoreEos(true)
      .build();
    SamplingParams greedy = hotter.asDeterministic();
    assertTrue(greedy.isDeterministic());
    assertEquals(0.9f, greedy.temperature());
    assertEquals(32, greedy.maxTokens());
    assertTrue(greedy.ignoreEos());
    assertEquals(1, greedy.topK());
    assertEquals(1f, greedy.topP());
    assertSame(greedy, greedy.asDeterministic());
  }
}
