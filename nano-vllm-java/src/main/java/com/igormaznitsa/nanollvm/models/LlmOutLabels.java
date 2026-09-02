package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Ranked classification labels from {@link LlmModel#generate(LlmInput, LlmModality)} with
 * {@link LlmModality#LABELS}.
 *
 * @param labels scored labels, highest score first; never {@code null} or empty
 * @since 1.4.0
 */
public record LlmOutLabels(List<LlmLabelScore> labels) implements LlmOutput {

  /**
   * @throws NullPointerException     if {@code labels} or an element is {@code null}
   * @throws IllegalArgumentException if {@code labels} is empty
   */
  public LlmOutLabels {
    requireNonNull(labels, "labels");
    if (labels.isEmpty()) {
      throw new IllegalArgumentException("labels must not be empty");
    }
    labels = List.copyOf(labels);
  }

  /**
   * Highest-scoring label string.
   *
   * @since 1.4.0
   */
  public String topLabel() {
    return this.labels.getFirst().label();
  }

  /**
   * Highest-scoring entry.
   *
   * @since 1.4.0
   */
  public LlmLabelScore top() {
    return this.labels.getFirst();
  }

  @Override
  public LlmModality modality() {
    return LlmModality.LABELS;
  }
}
