package com.igormaznitsa.nanollvm.jev;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * L0 arithmetic: cyclic rotations, log-mean marginal, and label-prior correction.
 */
final class JevMarginal {

  static final int MIN_PRIOR_STATES = 8;
  static final double PRIOR_STRENGTH = 0.75d;
  private static final double EPS = 1e-12d;

  private JevMarginal() {
  }

  static List<int[]> rotations(final JevQuestion question, final JevLevel level) {
    int count = question.options().size();
    if (level == JevLevel.RAW || question.isOrdered()) {
      return List.of(identity(count));
    }
    if (question.kind() == JevQuestion.Kind.YES_NO) {
      return List.of(new int[] {0, 1}, new int[] {1, 0});
    }
    return choiceRotations(question.options());
  }

  static int[] identity(final int count) {
    return IntStream.range(0, count).toArray();
  }

  static List<int[]> choiceRotations(final List<String> options) {
    int[] canonical = IntStream.range(0, options.size())
      .boxed()
      .sorted(Comparator.comparing(options::get).thenComparingInt(index -> index))
      .mapToInt(Integer::intValue)
      .toArray();
    List<int[]> shifts = new ArrayList<>(canonical.length);
    for (int shift = 0; shift < canonical.length; shift++) {
      int[] permutation = new int[canonical.length];
      for (int position = 0; position < canonical.length; position++) {
        permutation[position] = canonical[(position + shift) % canonical.length];
      }
      shifts.add(permutation);
    }
    return List.copyOf(shifts);
  }

  static double[] softmax(final List<Double> logProbabilities) {
    double max = Double.NEGATIVE_INFINITY;
    for (double value : logProbabilities) {
      max = Math.max(max, value);
    }
    double[] probabilities = new double[logProbabilities.size()];
    double sum = 0d;
    for (int index = 0; index < probabilities.length; index++) {
      probabilities[index] = Math.exp(logProbabilities.get(index) - max);
      sum += probabilities[index];
    }
    for (int index = 0; index < probabilities.length; index++) {
      probabilities[index] /= sum;
    }
    return probabilities;
  }

  static double[] marginalize(final double[][] positionProbabilities, final List<int[]> rotations) {
    int rotationCount = rotations.size();
    int optionCount = rotations.getFirst().length;
    double[] logSum = new double[optionCount];
    for (int rotation = 0; rotation < rotationCount; rotation++) {
      int[] permutation = rotations.get(rotation);
      for (int position = 0; position < optionCount; position++) {
        logSum[permutation[position]] += Math.log(
          Math.max(positionProbabilities[rotation][position], EPS));
      }
    }
    double max = Double.NEGATIVE_INFINITY;
    double[] scores = new double[optionCount];
    for (int option = 0; option < optionCount; option++) {
      scores[option] = logSum[option] / rotationCount;
      max = Math.max(max, scores[option]);
    }
    double[] probabilities = new double[optionCount];
    double sum = 0d;
    for (int option = 0; option < optionCount; option++) {
      probabilities[option] = Math.exp(scores[option] - max);
      sum += probabilities[option];
    }
    for (int option = 0; option < optionCount; option++) {
      probabilities[option] /= sum;
    }
    return probabilities;
  }

  static double[][] applyPrior(
    final double[][] positionProbabilities,
    final double[][] prior,
    final double strength
  ) {
    double[][] corrected = new double[positionProbabilities.length][];
    for (int rotation = 0; rotation < positionProbabilities.length; rotation++) {
      double[] weighted = new double[prior[rotation].length];
      for (int position = 0; position < weighted.length; position++) {
        weighted[position] = Math.pow(Math.max(prior[rotation][position], EPS), strength);
      }
      corrected[rotation] = divideAndNormalize(positionProbabilities[rotation], weighted);
    }
    return corrected;
  }

  static double[][] meanPrior(final double[][] sum, final int count) {
    double[][] prior = new double[sum.length][];
    for (int rotation = 0; rotation < sum.length; rotation++) {
      double[] row = new double[sum[rotation].length];
      double total = 0d;
      for (int position = 0; position < row.length; position++) {
        row[position] = Math.max(sum[rotation][position] / count, EPS);
        total += row[position];
      }
      for (int position = 0; position < row.length; position++) {
        row[position] /= total;
      }
      prior[rotation] = row;
    }
    return prior;
  }

  static double orderFlipRate(final double[][] positionProbabilities, final List<int[]> rotations) {
    if (rotations.size() <= 1) {
      return 0d;
    }
    int first = rotations.getFirst()[argmax(positionProbabilities[0])];
    int disagreements = 0;
    for (int rotation = 1; rotation < rotations.size(); rotation++) {
      int winner = rotations.get(rotation)[argmax(positionProbabilities[rotation])];
      if (winner != first) {
        disagreements++;
      }
    }
    return disagreements / (double) (rotations.size() - 1);
  }

  static int argmax(final double[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static double[] divideAndNormalize(final double[] raw, final double[] prior) {
    double[] corrected = new double[raw.length];
    double sum = 0d;
    for (int index = 0; index < raw.length; index++) {
      corrected[index] = Math.max(raw[index] / Math.max(prior[index], EPS), EPS);
      sum += corrected[index];
    }
    for (int index = 0; index < corrected.length; index++) {
      corrected[index] /= sum;
    }
    return corrected;
  }
}
