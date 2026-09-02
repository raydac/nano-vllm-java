package com.igormaznitsa.nanollvm.models;

/**
 * Typed result from {@link LlmModel#generate(LlmInput, LlmModality)} /
 * {@link com.igormaznitsa.nanollvm.llm.LLM#generate(LlmInput, LlmModality)}.
 *
 * @since 1.3.0
 */
public sealed interface LlmOutput
  permits LlmOutText, LlmOutSoundData, LlmOutEmbedding, LlmOutLabels {

  /**
   * Content type of this result.
   */
  LlmModality modality();
}
