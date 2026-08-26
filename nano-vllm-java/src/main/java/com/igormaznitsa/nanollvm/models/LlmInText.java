package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Text payload for {@link LlmModel#generate(LlmInput, LlmModality)}.
 *
 * @param text non-blank characters to complete, embed, or synthesize
 * @since 1.3.0
 */
public record LlmInText(CharSequence text) implements LlmInput {

  /**
   * @throws NullPointerException     if {@code text} is {@code null}
   * @throws IllegalArgumentException if {@code text} is blank
   */
  public LlmInText {
    requireNonNull(text, "text");
    if (text.toString().isBlank()) {
      throw new IllegalArgumentException("text must not be blank");
    }
  }

  /**
   * Creates a text input.
   *
   * @param text non-blank characters; must not be {@code null}
   * @return input wrapping {@code text}
   */
  public static LlmInText of(final CharSequence text) {
    return new LlmInText(text);
  }
}
