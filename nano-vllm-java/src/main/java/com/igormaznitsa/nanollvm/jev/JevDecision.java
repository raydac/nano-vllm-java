package com.igormaznitsa.nanollvm.jev;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.OptionalDouble;

/**
 * One typed answer: the chosen option, a probability for every option, and the level that
 * produced it.
 *
 * @param name          {@link JevQuestion#name()}, or empty
 * @param level         how the distribution was read
 * @param answer        option with the highest probability; the earliest option wins a tie
 * @param distribution  one entry per option, in the question's option order
 * @param labelMass     mean probability the model placed on the answer labels at all
 * @param rotations     how many option orders were read
 * @param orderFlipRate fraction of later rotations whose winner disagrees with the first;
 *                      {@code 0} when only one rotation was read
 * @param priorApplied  {@code true} when an L0 label prior corrected this distribution
 * @param centers       score-bin values aligned with {@code distribution}; empty for choice and
 *                      yes/no
 * @since 1.5.0
 */
public record JevDecision(
  String name,
  JevLevel level,
  String answer,
  List<JevOption> distribution,
  double labelMass,
  int rotations,
  double orderFlipRate,
  boolean priorApplied,
  List<Double> centers
) {

  /**
   * Copies the lists and checks that the answer is one of the options.
   */
  public JevDecision {
    requireNonNull(name, "name");
    requireNonNull(level, "level");
    requireNonNull(answer, "answer");
    distribution = List.copyOf(requireNonNull(distribution, "distribution"));
    centers = List.copyOf(requireNonNull(centers, "centers"));
    if (distribution.isEmpty()) {
      throw new IllegalArgumentException("distribution must not be empty");
    }
    if (!Double.isFinite(labelMass) || labelMass < 0d) {
      throw new IllegalArgumentException("labelMass must be finite and >= 0");
    }
    if (rotations < 1) {
      throw new IllegalArgumentException("rotations must be >= 1");
    }
    if (!Double.isFinite(orderFlipRate) || orderFlipRate < 0d || orderFlipRate > 1d) {
      throw new IllegalArgumentException("orderFlipRate must be in [0, 1]");
    }
    if (!centers.isEmpty() && centers.size() != distribution.size()) {
      throw new IllegalArgumentException("centers must be empty or match the distribution");
    }
    boolean known = distribution.stream().anyMatch(option -> option.option().equals(answer));
    if (!known) {
      throw new IllegalArgumentException("answer must be one of the options");
    }
  }

  /**
   * Probability of {@code option} in {@link #distribution()}.
   *
   * @param option option text; must be one of the question's options
   * @return probability
   * @throws NullPointerException     if {@code option} is {@code null}
   * @throws IllegalArgumentException if {@code option} is not in the distribution
   * @since 1.5.0
   */
  public double probability(final String option) {
    requireNonNull(option, "option");
    return this.distribution.stream()
      .filter(item -> item.option().equals(option))
      .mapToDouble(JevOption::probability)
      .findFirst()
      .orElseThrow(() -> new IllegalArgumentException("unknown option: " + option));
  }

  /**
   * Expected score, the probability-weighted bin centers. Empty for a choice or a yes/no.
   *
   * @return expected value, or empty when this decision is not a score
   * @since 1.5.0
   */
  public OptionalDouble score() {
    if (this.centers.isEmpty()) {
      return OptionalDouble.empty();
    }
    double value = 0d;
    for (int index = 0; index < this.distribution.size(); index++) {
      value += this.distribution.get(index).probability() * this.centers.get(index);
    }
    return OptionalDouble.of(value);
  }

  /**
   * Returns this decision when {@link #level()} is at least {@code minimum}.
   *
   * @param minimum level the caller is willing to act on
   * @return this decision
   * @throws NullPointerException  if {@code minimum} is {@code null}
   * @throws IllegalStateException if this decision is below {@code minimum}
   * @since 1.5.0
   */
  public JevDecision require(final JevLevel minimum) {
    requireNonNull(minimum, "minimum");
    if (this.level.ordinal() < minimum.ordinal()) {
      throw new IllegalStateException(
        "decision level %s is below required %s".formatted(this.level, minimum));
    }
    return this;
  }
}
