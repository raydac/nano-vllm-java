package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

/**
 * Neutral default {@link SamplingParams} for chat / completion helpers.
 * Model-family knobs (e.g. turn-based top-k) belong in the application or samples.
 */
public final class SamplingDefaults {

  public static final int DEFAULT_MAX_TOKENS = 256;

  private SamplingDefaults() {
  }

  public static SamplingParams forTokenizer(final Tokenizer tokenizer) {
    return forTokenizer(tokenizer, DEFAULT_MAX_TOKENS);
  }

  public static SamplingParams forTokenizer(final Tokenizer tokenizer, final int maxTokens) {
    return new SamplingParams(0.6f, maxTokens, false, 0, 0.95f);
  }
}
