package com.igormaznitsa.nanollvm.tensor.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Depthwise 1-D convolution along time. Loaded only when the incubator Vector API is present.
 *
 * @since 1.5.0
 */
public final class VectorConv1d {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
  private static final int LANES = SPECIES.length();

  private VectorConv1d() {
  }

  public static void channels(
    final float[] weights,
    final int weightOffset,
    final float[] bias,
    final int biasOffset,
    final float[] activations,
    final int inputOffset,
    final float[] dest,
    final int channel0,
    final int channel1,
    final int inLength,
    final int outLength,
    final int kernel,
    final int stride,
    final int padding,
    final int dilation
  ) {
    int[] index = stride == 1 ? null : new int[LANES];
    for (int channel = channel0; channel < channel1; channel++) {
      float shift = bias == null ? 0f : bias[biasOffset + channel];
      int weightBase = weightOffset + channel * kernel;
      int activationBase = inputOffset + channel * inLength;
      int outputBase = channel * outLength;
      int time = 0;
      while (time < outLength) {
        if (time <= outLength - LANES &&
          windowFits(time, inLength, kernel, stride, padding, dilation)) {
          writeVector(
            weights, weightBase, activations, activationBase, dest, outputBase + time,
            shift, time, kernel, stride, padding, dilation, index);
          time += LANES;
        } else {
          dest[outputBase + time] = oneSample(
            weights, weightBase, activations, activationBase, shift, time,
            inLength, kernel, stride, padding, dilation);
          time++;
        }
      }
    }
  }

  private static void writeVector(
    final float[] weights,
    final int weightBase,
    final float[] activations,
    final int activationBase,
    final float[] dest,
    final int destIndex,
    final float shift,
    final int time,
    final int kernel,
    final int stride,
    final int padding,
    final int dilation,
    final int[] index
  ) {
    FloatVector acc = FloatVector.broadcast(SPECIES, shift);
    int origin = time * stride - padding;
    if (stride == 1) {
      for (int tap = 0; tap < kernel; tap++) {
        FloatVector sample = FloatVector.fromArray(
          SPECIES, activations, activationBase + origin + tap * dilation);
        acc = sample.fma(FloatVector.broadcast(SPECIES, weights[weightBase + tap]), acc);
      }
    } else {
      for (int lane = 0; lane < LANES; lane++) {
        index[lane] = lane * stride;
      }
      for (int tap = 0; tap < kernel; tap++) {
        FloatVector sample = FloatVector.fromArray(
          SPECIES, activations, activationBase + origin + tap * dilation, index, 0);
        acc = sample.fma(FloatVector.broadcast(SPECIES, weights[weightBase + tap]), acc);
      }
    }
    acc.intoArray(dest, destIndex);
  }

  private static boolean windowFits(
    final int time,
    final int inLength,
    final int kernel,
    final int stride,
    final int padding,
    final int dilation
  ) {
    long first = (long) time * stride - padding;
    long last = (long) (time + LANES - 1) * stride - padding + (long) (kernel - 1) * dilation;
    return first >= 0 && last < inLength;
  }

  private static float oneSample(
    final float[] weights,
    final int weightBase,
    final float[] activations,
    final int activationBase,
    final float shift,
    final int time,
    final int inLength,
    final int kernel,
    final int stride,
    final int padding,
    final int dilation
  ) {
    float sum = shift;
    int origin = time * stride - padding;
    for (int tap = 0; tap < kernel; tap++) {
      int src = origin + tap * dilation;
      if (src >= 0 && src < inLength) {
        sum += weights[weightBase + tap] * activations[activationBase + src];
      }
    }
    return sum;
  }
}
