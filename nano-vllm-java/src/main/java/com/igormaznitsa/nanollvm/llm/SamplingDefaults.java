package com.igormaznitsa.nanollvm.llm;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

/**
 * Starting {@link SamplingParams} for chat / completion when you do not set
 * {@link LLM.Builder#sampling(SamplingParams)}.
 *
 * <p>These are architecture-agnostic starting points, not a quality preset: temperature
 * {@code 0.6} (how random the next word-piece is), {@code maxTokens} {@code 256} (how long the
 * reply may be), end-of-text honored, top-k off, top-p {@code 0.95}. Lower temperature (and
 * optionally a tighter top-p) for factual / RAG answers; raise temperature for more variety.
 * Same answer every run: {@link #deterministic()} or {@link LLM.Builder#deterministic()}.
 * Prefer {@link #neutral()} or {@link SamplingParams#builder()} — {@link #forTokenizer(Tokenizer)}
 * ignores its argument. Model-family extras (turn-based top-k, …) belong in the application.
 *
 * @see SamplingParams
 */
public final class SamplingDefaults {

  /**
   * Neutral softmax temperature ({@code 0.6}). Lower values (for example {@code 0.1}) make
   * next-token draws less random; {@code 0} is rejected.
   *
   * @since 1.1.0
   */
  public static final float DEFAULT_TEMPERATURE = 0.6f;

  /**
   * Neutral new-token cap ({@code 256}). The engine also clamps this to remaining context.
   *
   * @since 1.1.0
   */
  public static final int DEFAULT_MAX_TOKENS = 256;

  /**
   * Neutral nucleus probability ({@code 0.95}). {@code 1} disables nucleus; smaller values cut
   * a longer low-probability tail.
   *
   * @since 1.1.0
   */
  public static final float DEFAULT_TOP_P = 0.95f;

  private SamplingDefaults() {
  }

  /**
   * Architecture-neutral starting knobs: temperature {@code 0.6}, {@code maxTokens 256}, EOS
   * honored, top-k off, top-p {@code 0.95}. Not tuned per model family — lower temperature for
   * grounded / RAG answers; add top-k in the application when a checkpoint expects it.
   *
   * @return a new immutable {@link SamplingParams}
   * @see SamplingParams
   * @since 1.1.0
   */
  public static SamplingParams neutral() {
    return SamplingParams.builder().build();
  }

  /**
   * Same as {@link #neutral()} except {@code maxTokens} is the caller budget. Other knobs stay
   * at the architecture-neutral defaults. The engine still clamps this to remaining context.
   *
   * @param maxTokens new-token cap; must be {@code >= 1}
   * @return a new immutable {@link SamplingParams}
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   * @since 1.1.0
   */
  public static SamplingParams neutral(final int maxTokens) {
    return SamplingParams.builder().maxTokens(maxTokens).build();
  }

  /**
   * Repeatable knobs: highest-logit token every step ({@code topK = 1}, nucleus off). Other
   * fields match {@link #neutral()}. Prefer {@link LLM.Builder#deterministic()} to seal this on
   * chat / typed generate / RAG.
   *
   * @return a new immutable {@link SamplingParams}
   * @see SamplingParams#deterministic()
   * @since 1.2.0
   */
  public static SamplingParams deterministic() {
    return SamplingParams.deterministic();
  }

  /**
   * {@link #deterministic()} with a custom {@code maxTokens} budget.
   *
   * @param maxTokens new-token cap; must be {@code >= 1}
   * @return a new immutable {@link SamplingParams}
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   * @since 1.2.0
   */
  public static SamplingParams deterministic(final int maxTokens) {
    return SamplingParams.deterministic(maxTokens);
  }

  /**
   * Prefer {@link #neutral()}. The tokenizer argument is unused: family-specific knobs stay in
   * the application or samples, not in this library default.
   *
   * @param tokenizer ignored; may be {@code null}
   * @return {@link #neutral()}
   */
  public static SamplingParams forTokenizer(final Tokenizer tokenizer) {
    return neutral();
  }

  /**
   * Alias of {@link #neutral(int)}. The tokenizer argument is unused; see
   * {@link #forTokenizer(Tokenizer)}.
   *
   * @param tokenizer ignored; may be {@code null}
   * @param maxTokens new-token cap; must be {@code >= 1}
   * @return {@link #neutral(int)}
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   */
  public static SamplingParams forTokenizer(final Tokenizer tokenizer, final int maxTokens) {
    return neutral(maxTokens);
  }
}
