package com.igormaznitsa.nanollvm.llm;

import java.util.Objects;

/**
 * Immutable sampling knobs for {@link LLM#generate} and chat turns.
 *
 * <p>Construct only via {@link #builder()} or {@link SamplingDefaults}. Copies use {@code with*}
 * methods. Greedy sampling is rejected ({@code temperature} must be greater than {@code 1e-10}).
 * {@code topK == 0} disables top-k; {@code topP} must be in {@code (0, 1]}. Safe to share across
 * threads and across prompts in a batch.
 *
 * @see SamplingDefaults
 */
public final class SamplingParams {

  private final float temperature;
  private final int maxTokens;
  private final boolean ignoreEos;
  private final int topK;
  private final float topP;

  private SamplingParams(
    final float temperature,
    final int maxTokens,
    final boolean ignoreEos,
    final int topK,
    final float topP
  ) {
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
    this.temperature = temperature;
    this.maxTokens = maxTokens;
    this.ignoreEos = ignoreEos;
    this.topK = topK;
    this.topP = topP;
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
   * Softmax temperature; must be {@code > 1e-10}. Lower values make the next-token distribution
   * more peaked.
   */
  public float temperature() {
    return this.temperature;
  }

  /**
   * Maximum newly generated tokens per sequence; must be {@code >= 1}. The engine also clamps this
   * to remaining context ({@code maxModelLen}).
   */
  public int maxTokens() {
    return this.maxTokens;
  }

  /**
   * When {@code true}, end-of-sequence does not finish the sequence ({@code maxTokens} still
   * applies).
   */
  public boolean ignoreEos() {
    return this.ignoreEos;
  }

  /**
   * Keep only the top-{@code k} logits before nucleus; {@code 0} disables top-k.
   */
  public int topK() {
    return this.topK;
  }

  /**
   * Nucleus sampling cumulative probability in {@code (0, 1]}.
   */
  public float topP() {
    return this.topP;
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

  @Override
  public boolean equals(final Object other) {
    return other instanceof SamplingParams that
      && Float.compare(this.temperature, that.temperature) == 0
      && this.maxTokens == that.maxTokens
      && this.ignoreEos == that.ignoreEos
      && this.topK == that.topK
      && Float.compare(this.topP, that.topP) == 0;
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
  }

  @Override
  public String toString() {
    return "SamplingParams[temperature=%s, maxTokens=%d, ignoreEos=%s, topK=%d, topP=%s]"
      .formatted(this.temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
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
