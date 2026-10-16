package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Full-vocabulary log-probabilities of a few label tokens at the next-token position, after one
 * prefill. {@code logProbabilities.get(i)} is {@code log P(labelTokenIds[i] | prompt)}.
 * {@link #labelMass()} is the probability the model placed on those labels together.
 *
 * @param logProbabilities one natural log-probability per requested label id, same order
 * @param labelMass        sum of the corresponding probabilities, in {@code [0, 1]}
 * @since 1.5.0
 */
public record NextTokenRead(List<Double> logProbabilities, double labelMass) {

  /**
   * Copies the log-probability list and rejects a non-finite mass.
   *
   * @throws NullPointerException     if {@code logProbabilities} is {@code null}
   * @throws IllegalArgumentException if the list is empty or {@code labelMass} is not finite
   */
  public NextTokenRead {
    logProbabilities = List.copyOf(requireNonNull(logProbabilities, "logProbabilities"));
    if (logProbabilities.isEmpty()) {
      throw new IllegalArgumentException("logProbabilities must not be empty");
    }
    if (!Double.isFinite(labelMass)) {
      throw new IllegalArgumentException("labelMass must be finite");
    }
  }
}
