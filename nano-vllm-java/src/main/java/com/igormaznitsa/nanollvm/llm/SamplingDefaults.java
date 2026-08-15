package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

/**
 * Neutral default {@link SamplingParams} for chat / completion helpers.
 * Model-family knobs (e.g. turn-based top-k) belong in the application or samples.
 */
public final class SamplingDefaults {

  /**
   * Neutral softmax temperature ({@code 0.6}).
   *
   * @since 1.1.0
   */
  public static final float DEFAULT_TEMPERATURE = 0.6f;
  public static final int DEFAULT_MAX_TOKENS = 256;

  /**
   * Neutral nucleus probability ({@code 0.95}).
   *
   * @since 1.1.0
   */
  public static final float DEFAULT_TOP_P = 0.95f;

  private SamplingDefaults() {
  }

  /**
   * Architecture-neutral defaults: temperature {@code 0.6}, {@code maxTokens 256}, EOS honored,
   * top-k off, top-p {@code 0.95}.
   *
   * @since 1.1.0
   */
  public static SamplingParams neutral() {
    return SamplingParams.builder().build();
  }

  /**
   * {@link #neutral()} with a caller-chosen {@code maxTokens} ({@code >= 1}).
   *
   * @since 1.1.0
   */
  public static SamplingParams neutral(final int maxTokens) {
    return SamplingParams.builder().maxTokens(maxTokens).build();
  }

  /**
   * Alias of {@link #neutral()}. The tokenizer is unused; family knobs stay in the application.
   */
  public static SamplingParams forTokenizer(final Tokenizer tokenizer) {
    return neutral();
  }

  /**
   * Alias of {@link #neutral(int)}. The tokenizer is unused; family knobs stay in the application.
   */
  public static SamplingParams forTokenizer(final Tokenizer tokenizer, final int maxTokens) {
    return neutral(maxTokens);
  }
}
