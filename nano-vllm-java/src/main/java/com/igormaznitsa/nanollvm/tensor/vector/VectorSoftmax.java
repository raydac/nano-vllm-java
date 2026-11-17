package com.igormaznitsa.nanollvm.tensor.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Dense softmax over a full score row. Loaded only when the incubator Vector API is present.
 *
 * @since 1.5.0
 */
public final class VectorSoftmax {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
  private static final int LANES = SPECIES.length();

  private VectorSoftmax() {
  }

  public static int lanes() {
    return LANES;
  }

  public static void softmax(
    final float[] scores,
    final int base,
    final int length,
    final float scale
  ) {
    FloatVector scaleV = FloatVector.broadcast(SPECIES, scale);
    FloatVector maxV = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
    int upper = SPECIES.loopBound(length);
    int i = 0;
    for (; i < upper; i += LANES) {
      FloatVector scaled = FloatVector.fromArray(SPECIES, scores, base + i).mul(scaleV);
      scaled.intoArray(scores, base + i);
      maxV = maxV.max(scaled);
    }
    float max = maxV.reduceLanes(VectorOperators.MAX);
    for (; i < length; i++) {
      float score = scores[base + i] * scale;
      scores[base + i] = score;
      if (score > max) {
        max = score;
      }
    }
    FloatVector maxB = FloatVector.broadcast(SPECIES, max);
    FloatVector sumV = FloatVector.zero(SPECIES);
    float sum = 0f;
    i = 0;
    for (; i < upper; i += LANES) {
      FloatVector weight = FloatVector.fromArray(SPECIES, scores, base + i)
        .sub(maxB)
        .lanewise(VectorOperators.EXP);
      weight.intoArray(scores, base + i);
      sumV = sumV.add(weight);
    }
    sum += sumV.reduceLanes(VectorOperators.ADD);
    for (; i < length; i++) {
      float weight = (float) Math.exp(scores[base + i] - max);
      scores[base + i] = weight;
      sum += weight;
    }
    float inv = 1f / sum;
    FloatVector invB = FloatVector.broadcast(SPECIES, inv);
    i = 0;
    for (; i < upper; i += LANES) {
      FloatVector.fromArray(SPECIES, scores, base + i).mul(invB).intoArray(scores, base + i);
    }
    for (; i < length; i++) {
      scores[base + i] *= inv;
    }
  }
}
