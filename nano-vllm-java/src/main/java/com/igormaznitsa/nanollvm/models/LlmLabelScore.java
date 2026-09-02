package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * One scored label from a text classifier (for example fastText language id).
 *
 * @param label raw model label (often {@code __label__en}); never blank
 * @param score probability in {@code [0, 1]}
 * @since 1.4.0
 */
public record LlmLabelScore(String label, float score) {

  /**
   * @throws NullPointerException     if {@code label} is {@code null}
   * @throws IllegalArgumentException if {@code label} is blank or {@code score} is not finite
   */
  public LlmLabelScore {
    requireNonNull(label, "label");
    if (label.isBlank()) {
      throw new IllegalArgumentException("label must not be blank");
    }
    if (!Float.isFinite(score)) {
      throw new IllegalArgumentException("score must be finite");
    }
  }
}
