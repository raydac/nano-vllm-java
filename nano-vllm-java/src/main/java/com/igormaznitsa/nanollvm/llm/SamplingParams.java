package com.igormaznitsa.nanollvm.llm;

/**
 * Immutable sampling knobs for {@link LLM#generate} and chat turns.
 *
 * <p>Greedy sampling is rejected ({@code temperature} must be greater than {@code 1e-10}).
 * {@code topK == 0} disables top-k; {@code topP} must be in {@code (0, 1]}. Prefer
 * {@link SamplingDefaults#forTokenizer} for chat rather than constructing these by hand.
 * Safe to share across threads and across prompts in a batch.
 *
 * @param temperature softmax temperature; must be {@code > 1e-10} (greedy not supported). Lower
 *                    values make the next-token distribution more peaked.
 * @param maxTokens   maximum newly generated tokens per sequence; must be {@code >= 1}. The
 *                    engine also clamps this to remaining context ({@code maxModelLen}).
 * @param ignoreEos   when {@code true}, end-of-sequence does not finish the sequence
 *                    ({@code maxTokens} still applies)
 * @param topK        keep only the top-{@code k} logits before nucleus; {@code 0} disables top-k
 * @param topP        nucleus sampling cumulative probability in {@code (0, 1]}
 * @see SamplingDefaults
 */
public record SamplingParams(
  float temperature,
  int maxTokens,
  boolean ignoreEos,
  int topK,
  float topP
) {

  public SamplingParams {
    if (temperature <= 1e-10f) {
      throw new IllegalArgumentException("greedy sampling is not permitted");
    }
    if (maxTokens < 1) {
      throw new IllegalArgumentException("maxTokens must be >= 1");
    }
    if (topK < 0) {
      throw new IllegalArgumentException("topK must be >= 0 (0 = disabled)");
    }
    if (topP <= 0f || topP > 1f) {
      throw new IllegalArgumentException("topP must be in (0, 1]");
    }
  }

  /**
   * Defaults: temperature {@code 0.7}, {@code maxTokens 64}, EOS honored, top-k off, top-p {@code 0.9}.
   */
  public SamplingParams() {
    this(0.7f, 64, false, 0, 0.9f);
  }

  /**
   * Convenience with EOS honored, top-k off, top-p {@code 0.9}.
   *
   * @param temperature must be {@code > 1e-10}
   * @param maxTokens   must be {@code >= 1}
   */
  public SamplingParams(final float temperature, final int maxTokens) {
    this(temperature, maxTokens, false, 0, 0.9f);
  }

  /**
   * Convenience with top-k off and top-p {@code 0.9}.
   *
   * @param temperature must be {@code > 1e-10}
   * @param maxTokens   must be {@code >= 1}
   * @param ignoreEos   see record component
   */
  public SamplingParams(final float temperature, final int maxTokens, final boolean ignoreEos) {
    this(temperature, maxTokens, ignoreEos, 0, 0.9f);
  }
}
