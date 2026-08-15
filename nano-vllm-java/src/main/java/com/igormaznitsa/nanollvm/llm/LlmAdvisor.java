package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

/**
 * Named advisor role configured on {@link LLM.Builder#advisors(LlmAdvisorMixer, LlmAdvisor...)}.
 *
 * <p>Immutable; construct via {@link #builder()}.
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
   * required before {@link Builder#build()}.
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Unique non-blank role name (must be unique among advisors on one {@link LLM}).
   */
  public String name() {
    return this.name;
  }

  /**
   * Instruction text sent to this advisor on each turn.
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

  public static final class Builder {

    private String name;
    private String prompt;

    private Builder() {
    }

    /**
     * Unique non-blank role name.
     */
    public Builder name(final String name) {
      this.name = requireNonNull(name, "name");
      return this;
    }

    /**
     * Instruction text sent to this advisor on each turn.
     */
    public Builder prompt(final String prompt) {
      this.prompt = requireNonNull(prompt, "prompt");
      return this;
    }

    /**
     * Validates name and prompt, then constructs an immutable advisor.
     *
     * @throws IllegalArgumentException if name or prompt is missing or blank
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
