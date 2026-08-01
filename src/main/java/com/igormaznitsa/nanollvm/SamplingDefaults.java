package com.igormaznitsa.nanollvm;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

/**
 * Model-aware default {@link SamplingParams} for chat / completion helpers.
 */
public final class SamplingDefaults {

  public static final int DEFAULT_MAX_TOKENS = 256;

  private SamplingDefaults() {
  }

  public static SamplingParams forTokenizer(Tokenizer tokenizer) {
    return forTokenizer(tokenizer, DEFAULT_MAX_TOKENS);
  }

  public static SamplingParams forTokenizer(Tokenizer tokenizer, int maxTokens) {
    if (tokenizer != null && tokenizer.isGemmaChat()) {
      return new SamplingParams(0.6f, maxTokens, false, 64, 0.95f);
    }
    return new SamplingParams(0.6f, maxTokens, false, 0, 0.95f);
  }
}
