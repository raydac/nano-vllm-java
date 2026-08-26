package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Text result from completion or speech-to-text.
 *
 * @param text decoded characters; never {@code null} (may be empty)
 * @since 1.3.0
 */
public record LlmOutText(String text) implements LlmOutput {

  /**
   * @throws NullPointerException if {@code text} is {@code null}
   */
  public LlmOutText {
    requireNonNull(text, "text");
  }

  @Override
  public LlmModality modality() {
    return LlmModality.TEXT;
  }
}
