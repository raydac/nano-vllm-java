package com.igormaznitsa.nanollvm.llm;

import java.util.Objects;

/**
 * Immutable next-token sampling knobs for {@link LLM#generate} and chat / RAG turns.
 *
 * <p>Construct only via {@link #builder()} or {@link SamplingDefaults}. Copies use {@code with*}
 * methods. The built object is safe to share across threads and across prompts in a batch. There
 * is no RNG seed: repeated runs can differ unless the knobs are {@linkplain #deterministic()
 * deterministic} ({@code topK == 1}), which always picks the highest-logit token.
 *
 * <p>Each decode step turns logits into a distribution, then draws one token, in this order:
 * <ol>
 *   <li><b>Temperature</b> — softmax uses {@code logits / temperature}. Lower values peak on the
 *       highest-logit token (more repeatable, less random). Higher values flatten the distribution
 *       (more variety). Must be {@code > 1e-10}; {@code 0} is rejected. Repeatable argmax is
 *       {@link #deterministic()} / {@link LLM.Builder#deterministic()}, not temperature zero.
 *       Neutral chat is {@link SamplingDefaults#DEFAULT_TEMPERATURE} ({@code 0.6}). Factoid / RAG
 *       answers usually want {@code 0.1}–{@code 0.2}.</li>
 *   <li><b>Top-k</b> — keep only the {@code k} most probable tokens and renormalize.
 *       {@code 0} leaves the full vocabulary (disabled). Typical chat values are {@code 20}–{@code 64}
 *       (some turn-based models sit near {@code 64}).</li>
 *   <li><b>Top-p</b> (nucleus) — keep the smallest prefix of remaining tokens whose probabilities
 *       sum to at least {@code p}, then renormalize. Must be in {@code (0, 1]}; {@code 1} disables
 *       nucleus. {@link SamplingDefaults#DEFAULT_TOP_P} is {@code 0.95}. Tighter values
 *       ({@code 0.8}–{@code 0.85}) cut a long low-probability tail.</li>
 * </ol>
 * Top-k runs first when enabled, then top-p. Both may be combined.
 *
 * <p>{@link #maxTokens()} caps <em>new</em> tokens; the engine also clamps it to remaining context
 * ({@code maxModelLen}). {@link #ignoreEos()} {@code true} keeps generating through end-of-sequence
 * until that cap.
 *
 * <p>{@link com.igormaznitsa.nanollvm.rag.RagSession} may further lower temperature on turns that
 * retrieved passages, even if these knobs are hotter.
 *
 * @see SamplingDefaults
 * @see LLM.Builder#sampling(SamplingParams)
 * @see com.igormaznitsa.nanollvm.chat.ChatSession#sampling(SamplingParams)
 * @see com.igormaznitsa.nanollvm.rag.RagSession#sampling(SamplingParams)
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
   * Fluent configurator. Unset fields match {@link SamplingDefaults#neutral()} ({@code temperature}
   * {@code 0.6}, {@code maxTokens} {@code 256}, EOS honored, top-k off, top-p {@code 0.95}).
   * Validation runs in {@link Builder#build()}.
   *
   * @return a new builder
   * @since 1.1.0
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Repeatable next-token picks: keep only the highest-logit token ({@code topK = 1}) and turn
   * nucleus off ({@code topP = 1}). Temperature stays at {@link SamplingDefaults#DEFAULT_TEMPERATURE}
   * (ranking is unchanged). Same prompt → same token ids.
   *
   * @return a new immutable instance
   * @since 1.1.1
   */
  public static SamplingParams deterministic() {
    return builder().deterministic().build();
  }

  /**
   * {@link #deterministic()} with a custom {@link #maxTokens()} cap.
   *
   * @param maxTokens must be {@code >= 1}
   * @return a new immutable instance
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   * @since 1.1.1
   */
  public static SamplingParams deterministic(final int maxTokens) {
    return builder().deterministic().maxTokens(maxTokens).build();
  }

  /**
   * Softmax temperature {@code τ}: logits are divided by this value before softmax.
   * Lower {@code τ} → more deterministic; higher {@code τ} → more random. Always {@code > 1e-10}.
   *
   * @return temperature in {@code (1e-10, +∞)}
   */
  public float temperature() {
    return this.temperature;
  }

  /**
   * Maximum newly generated tokens per sequence. The engine also clamps this to remaining context
   * ({@code maxModelLen}).
   *
   * @return cap {@code >= 1}
   */
  public int maxTokens() {
    return this.maxTokens;
  }

  /**
   * When {@code true}, an end-of-sequence token does not finish the sequence; generation continues
   * until {@link #maxTokens()} (or the context clamp).
   *
   * @return {@code true} if EOS is ignored
   */
  public boolean ignoreEos() {
    return this.ignoreEos;
  }

  /**
   * After softmax, keep only this many highest-probability tokens. {@code 0} disables top-k
   * (full vocabulary, then top-p if it is below {@code 1}).
   *
   * @return {@code k >= 0}; {@code 0} means top-k is off
   */
  public int topK() {
    return this.topK;
  }

  /**
   * Nucleus (top-p) cumulative probability. Applied after top-k. {@code 1} keeps the whole
   * remaining mass (nucleus off).
   *
   * @return {@code p} in {@code (0, 1]}
   */
  public float topP() {
    return this.topP;
  }

  /**
   * Copy with a new temperature. Lower values make answers less random; must stay {@code > 1e-10}
   * (greedy {@code 0} is rejected).
   *
   * @param temperature softmax {@code τ}; must be {@code > 1e-10}
   * @return a new instance; this object is unchanged
   * @throws IllegalArgumentException if {@code temperature <= 1e-10}
   * @since 1.1.0
   */
  public SamplingParams withTemperature(final float temperature) {
    return new SamplingParams(temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
  }

  /**
   * Copy with a different {@link #maxTokens()} cap. The engine still clamps this to remaining
   * context ({@code maxModelLen - promptLen}) at generate time.
   *
   * @param maxTokens must be {@code >= 1}
   * @return a new instance; this object is unchanged
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   * @since 1.1.0
   */
  public SamplingParams withMaxTokens(final int maxTokens) {
    return new SamplingParams(this.temperature, maxTokens, this.ignoreEos, this.topK, this.topP);
  }

  /**
   * Copy that honors or ignores end-of-sequence.
   *
   * @param ignoreEos {@code true} to keep generating through EOS until {@link #maxTokens()}
   * @return a new instance; this object is unchanged
   * @since 1.1.0
   */
  public SamplingParams withIgnoreEos(final boolean ignoreEos) {
    return new SamplingParams(this.temperature, this.maxTokens, ignoreEos, this.topK, this.topP);
  }

  /**
   * Copy with a new top-k. {@code 0} disables the cap.
   *
   * @param topK must be {@code >= 0}
   * @return a new instance; this object is unchanged
   * @throws IllegalArgumentException if {@code topK < 0}
   * @since 1.1.0
   */
  public SamplingParams withTopK(final int topK) {
    return new SamplingParams(this.temperature, this.maxTokens, this.ignoreEos, topK, this.topP);
  }

  /**
   * Copy with a new nucleus probability. {@code 1} disables nucleus; smaller values (for example
   * {@code 0.8}) cut a longer tail.
   *
   * @param topP must be in {@code (0, 1]}
   * @return a new instance; this object is unchanged
   * @throws IllegalArgumentException if {@code topP} is outside {@code (0, 1]}
   * @since 1.1.0
   */
  public SamplingParams withTopP(final float topP) {
    return new SamplingParams(this.temperature, this.maxTokens, this.ignoreEos, this.topK, topP);
  }

  /**
   * {@code true} when next-token choice cannot use the RNG ({@link #topK()} is {@code 1}).
   *
   * @return whether these knobs are greedy argmax
   * @since 1.1.1
   */
  public boolean isDeterministic() {
    return this.topK == 1;
  }

  /**
   * Copy that always picks the highest-logit token ({@code topK = 1}, {@code topP = 1}).
   * Temperature, {@link #maxTokens()}, and {@link #ignoreEos()} stay.
   *
   * @return this instance when already deterministic; otherwise a new copy
   * @since 1.1.1
   */
  public SamplingParams asDeterministic() {
    return this.isDeterministic() && this.topP == 1f
      ? this
      : new SamplingParams(this.temperature, this.maxTokens, this.ignoreEos, 1, 1f);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean equals(final Object other) {
    return other instanceof SamplingParams that
      && Float.compare(this.temperature, that.temperature) == 0
      && this.maxTokens == that.maxTokens
      && this.ignoreEos == that.ignoreEos
      && this.topK == that.topK
      && Float.compare(this.topP, that.topP) == 0;
  }

  /** {@inheritDoc} */
  @Override
  public int hashCode() {
    return Objects.hash(this.temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return "SamplingParams[temperature=%s, maxTokens=%d, ignoreEos=%s, topK=%d, topP=%s]"
      .formatted(this.temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
  }

  /**
   * Fluent configurator. Unset fields match {@link SamplingDefaults#neutral()}.
   *
   * @see SamplingParams
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
     * Softmax temperature. Lower → less random; must be {@code > 1e-10} at {@link #build()}.
     *
     * @param temperature softmax {@code τ}
     * @return {@code this}
     * @since 1.1.0
     */
    public Builder temperature(final float temperature) {
      this.temperature = temperature;
      return this;
    }

    /**
     * Maximum newly generated tokens per sequence.
     *
     * @param maxTokens must be {@code >= 1} at {@link #build()}
     * @return {@code this}
     * @since 1.1.0
     */
    public Builder maxTokens(final int maxTokens) {
      this.maxTokens = maxTokens;
      return this;
    }

    /**
     * When {@code true}, end-of-sequence does not finish the sequence.
     *
     * @param ignoreEos {@code true} to ignore EOS
     * @return {@code this}
     * @since 1.1.0
     */
    public Builder ignoreEos(final boolean ignoreEos) {
      this.ignoreEos = ignoreEos;
      return this;
    }

    /**
     * Keep only the {@code k} most probable tokens after softmax. {@code 0} disables top-k.
     *
     * @param topK must be {@code >= 0} at {@link #build()}
     * @return {@code this}
     * @since 1.1.0
     */
    public Builder topK(final int topK) {
      this.topK = topK;
      return this;
    }

    /**
     * Nucleus cumulative probability after top-k. {@code 1} disables nucleus.
     *
     * @param topP must be in {@code (0, 1]} at {@link #build()}
     * @return {@code this}
     * @since 1.1.0
     */
    public Builder topP(final float topP) {
      this.topP = topP;
      return this;
    }

    /**
     * Greedy argmax: {@code topK(1)} and {@code topP(1)}. Temperature and {@link #maxTokens(int)}
     * stay as set. Prefer {@link LLM.Builder#deterministic()} to seal this on an engine.
     *
     * @return {@code this}
     * @since 1.1.1
     */
    public Builder deterministic() {
      this.topK = 1;
      this.topP = 1f;
      return this;
    }

    /**
     * Seals this configurator into an immutable {@link SamplingParams}.
     *
     * @return a new validated instance
     * @throws IllegalArgumentException if temperature is greedy ({@code <= 1e-10}),
     *                                  {@code maxTokens < 1}, {@code topK < 0}, or {@code topP}
     *                                  is outside {@code (0, 1]}
     * @since 1.1.0
     */
    public SamplingParams build() {
      return new SamplingParams(
        this.temperature, this.maxTokens, this.ignoreEos, this.topK, this.topP);
    }
  }
}
