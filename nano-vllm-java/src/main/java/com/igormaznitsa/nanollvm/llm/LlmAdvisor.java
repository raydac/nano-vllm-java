package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

/**
 * Named pre-answer role attached to an {@link LLM} via
 * {@link LLM.Builder#advisors(LlmAdvisorMixer, LlmAdvisor...)}.
 *
 * <p>On each chat / RAG turn the engine runs every configured advisor as <em>one batched</em>
 * {@link LLM#generate} (same lock as the main reply). Each advisor sees the user turn (and prior
 * user turns) plus this object's {@link #prompt()} as its system instruction. The library ships
 * no default names or wording — the application owns both. Names must be unique on one engine
 * (case-insensitive).
 *
 * <p>Replies are mixed into the main user prompt by {@link LlmAdvisorMixer}, then the primary
 * generate runs. When the main answer is unusable,
 * {@link com.igormaznitsa.nanollvm.chat.ChatSession} salvage may reuse grounded
 * advisor notes. Advisors add latency and tokens; skip them for tiny models or tight max-token
 * budgets. Immutable; construct only via {@link #builder()}.
 *
 * @see LlmAdvisorMixer
 * @see AdvisorEnrichment
 * @see LLM.Builder#advisors(LlmAdvisorMixer, LlmAdvisor...)
 */
public final class LlmAdvisor {

  private final String name;
  private final String prompt;

  private LlmAdvisor(final String name, final String prompt) {
    this.name = name;
    this.prompt = prompt;
  }

  /**
   * Starts a fluent builder. {@link Builder#name(String)} and {@link Builder#prompt(String)} are
   * both required before {@link Builder#build()}; omitting either fails at build time.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Role label used in mix / salvage and uniqueness checks. Must be unique among advisors on one
   * {@link LLM} (compared case-insensitively after strip).
   *
   * @return non-blank name; never {@code null}
   */
  public String name() {
    return this.name;
  }

  /**
   * System-side instruction this advisor receives on every turn. Not a user-visible reply; it
   * tells the model how to write the advisor note (facts, risks, style, …).
   *
   * @return non-blank instruction; never {@code null}
   */
  public String prompt() {
    return this.prompt;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public String toString() {
    return "LlmAdvisor[name=%s]".formatted(this.name);
  }

  /**
   * Fluent configurator. {@link #name(String)} and {@link #prompt(String)} must both be set to
   * non-blank text before {@link #build()}.
   */
  public static final class Builder {

    private String name;
    private String prompt;

    private Builder() {
    }

    /**
     * Role label shown in mixed notes and uniqueness checks. Leading/trailing whitespace is
     * stripped at {@link #build()}.
     *
     * @param name non-{@code null} name; must be non-blank after strip
     * @return {@code this}
     * @throws NullPointerException if {@code name} is {@code null}
     */
    public Builder name(final String name) {
      this.name = requireNonNull(name, "name");
      return this;
    }

    /**
     * Instruction text sent as this advisor's system prompt. Own the wording here; the library
     * does not inject a default policy.
     *
     * @param prompt non-{@code null} instruction; must be non-blank after strip
     * @return {@code this}
     * @throws NullPointerException if {@code prompt} is {@code null}
     */
    public Builder prompt(final String prompt) {
      this.prompt = requireNonNull(prompt, "prompt");
      return this;
    }

    /**
     * Validates name and prompt (non-null, non-blank after strip), then constructs an immutable
     * advisor.
     *
     * @return a new {@link LlmAdvisor}
     * @throws NullPointerException     if name or prompt was never set
     * @throws IllegalArgumentException if name or prompt is blank after strip
     */
    public LlmAdvisor build() {
      String trimmedName = requireNonNull(this.name, "name").strip();
      if (trimmedName.isEmpty()) {
        throw new IllegalArgumentException("advisor name must not be blank");
      }
      String trimmedPrompt = requireNonNull(this.prompt, "prompt").strip();
      if (trimmedPrompt.isEmpty()) {
        throw new IllegalArgumentException("advisor prompt must not be blank");
      }
      return new LlmAdvisor(trimmedName, trimmedPrompt);
    }
  }
}
