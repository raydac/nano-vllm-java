package com.igormaznitsa.nanollvm.jev;

import static java.util.Objects.requireNonNull;

/**
 * One option and its probability in a {@link JevDecision}.
 *
 * @param option      option text, the same string passed to {@link JevQuestion}
 * @param probability probability in the returned distribution; finite and {@code >= 0}
 * @since 1.5.0
 */
public record JevOption(String option, double probability) {

  /**
   * Rejects a null option or a non-finite negative probability.
   */
  public JevOption {
    requireNonNull(option, "option");
    if (!Double.isFinite(probability) || probability < 0d) {
      throw new IllegalArgumentException("probability must be finite and >= 0");
    }
  }
}
