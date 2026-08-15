package com.igormaznitsa.nanollvm.llm;

/**
 * Immutable sampling knobs for {@link LLM#generate} and chat turns.
 *
 * <p>Greedy sampling is rejected ({@code temperature} must be greater than {@code 1e-10}).
 * {@code topK == 0} disables top-k; {@code topP} must be in {@code (0, 1]}. Prefer
 * {@link #builder()} or {@link SamplingDefaults#neutral()} rather than constructing these by hand.
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
   * Neutral defaults: temperature {@code 0.6}, {@code maxTokens 256}, EOS honored, top-k off,
   * top-p {@code 0.95}. Same table as {@link SamplingDefaults#neutral()}.
   */
  public SamplingParams() {
    this(
      SamplingDefaults.DEFAULT_TEMPERATURE,
      SamplingDefaults.DEFAULT_MAX_TOKENS,
      false,
      0,
      SamplingDefaults.DEFAULT_TOP_P);
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

  /**
   * Fluent configurator. Defaults match {@link SamplingDefaults#neutral()}.
   *
   * @since 1.1.0
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Copy with a new temperature ({@code > 1e-10}).
   *
   * @since 1.1.0
   */
  public SamplingParams withTemperature(final float temperature) {
    return new SamplingParams(temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
  }

  /**
   * Copy with a new {@code maxTokens} ({@code >= 1}).
   *
   * @since 1.1.0
   */
  public SamplingParams withMaxTokens(final int maxTokens) {
    return new SamplingParams(this.temperature, maxTokens, this.ignoreEos, this.topK, this.topP);
  }

  /**
   * Copy that honors or ignores end-of-sequence.
   *
   * @since 1.1.0
   */
  public SamplingParams withIgnoreEos(final boolean ignoreEos) {
    return new SamplingParams(this.temperature, this.maxTokens, ignoreEos, this.topK, this.topP);
  }

  /**
   * Copy with a new top-k ({@code 0} disables).
   *
   * @since 1.1.0
   */
  public SamplingParams withTopK(final int topK) {
    return new SamplingParams(this.temperature, this.maxTokens, this.ignoreEos, topK, this.topP);
  }

  /**
   * Copy with a new nucleus probability in {@code (0, 1]}.
   *
   * @since 1.1.0
   */
  public SamplingParams withTopP(final float topP) {
    return new SamplingParams(this.temperature, this.maxTokens, this.ignoreEos, this.topK, topP);
  }

  /**
   * Fluent configurator. Defaults match {@link SamplingDefaults#neutral()}.
   *
   * @since 1.1.0
   */
  public static final class Builder {

    private float temperature = SamplingDefaults.DEFAULT_TEMPERATURE;
    private int maxTokens = SamplingDefaults.DEFAULT_MAX_TOKENS;
    private boolean ignoreEos;
    private int topK;
    private float topP = SamplingDefaults.DEFAULT_TOP_P;

    private Builder() {
    }

    /**
     * Softmax temperature; must be {@code > 1e-10}.
     *
     * @since 1.1.0
     */
    public Builder temperature(final float temperature) {
      this.temperature = temperature;
      return this;
    }

    /**
     * Maximum newly generated tokens; must be {@code >= 1}.
     *
     * @since 1.1.0
     */
    public Builder maxTokens(final int maxTokens) {
      this.maxTokens = maxTokens;
      return this;
    }

    /**
     * When {@code true}, end-of-sequence does not finish the sequence.
     *
     * @since 1.1.0
     */
    public Builder ignoreEos(final boolean ignoreEos) {
      this.ignoreEos = ignoreEos;
      return this;
    }

    /**
     * Keep only the top-{@code k} logits; {@code 0} disables top-k.
     *
     * @since 1.1.0
     */
    public Builder topK(final int topK) {
      this.topK = topK;
      return this;
    }

    /**
     * Nucleus sampling cumulative probability in {@code (0, 1]}.
     *
     * @since 1.1.0
     */
    public Builder topP(final float topP) {
      this.topP = topP;
      return this;
    }

    /**
     * Seals this configurator into an immutable {@link SamplingParams}.
     *
     * @since 1.1.0
     */
    public SamplingParams build() {
      return new SamplingParams(
        this.temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
    }
  }
}
