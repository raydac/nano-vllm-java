package com.igormaznitsa.nanollvm.jev;

/**
 * How a {@link JevDecision} was read from the model.
 *
 * <p>{@link #RAW} is weaker than {@link #L0}. {@link JevDecision#require(JevLevel)} refuses a
 * result below the asked level.
 *
 * @since 1.5.0
 */
public enum JevLevel {

  /**
   * One prefill. Options stay in the order given. Position bias and label preference stay in
   * the distribution.
   */
  RAW,

  /**
   * Every cyclic rotation of an unordered choice (both phrasings of a yes/no). Log-mean across
   * rotations, then a label prior once enough states of the same question have been seen.
   * Ordered scores stay a single prefill.
   */
  L0
}
