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

  public static Builder builder() {
    return new Builder();
  }

  public String name() {
    return this.name;
  }

  public String prompt() {
    return this.prompt;
  }

  @Override
  public String toString() {
    return "LlmAdvisor[name=%s]".formatted(this.name);
  }

  public static final class Builder {

    private String name;
    private String prompt;

    private Builder() {
    }

    public Builder name(final String name) {
      this.name = requireNonNull(name, "name");
      return this;
    }

    public Builder prompt(final String prompt) {
      this.prompt = requireNonNull(prompt, "prompt");
      return this;
    }

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
