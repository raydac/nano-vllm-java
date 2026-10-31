package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * One TornadoVM task for every row of an RMSNorm. The gain vector is already {@code w} or
 * {@code 1 + w}; a weightless norm stores ones.
 *
 * @since 1.5.0
 */
final class TornadoNormKernels {

  private TornadoNormKernels() {
  }

  static void rmsNormRows(
    final float[] src, final float[] weight, final float[] dst, final float[] eps,
    final int rows, final int last
  ) {
    float epsilon = eps[0];
    for (@Parallel int row = 0; row < rows; row++) {
      int base = row * last;
      float energy = 0f;
      for (int i = 0; i < last; i++) {
        float value = src[base + i];
        energy += value * value;
      }
      float inv = 1f / (float) Math.sqrt(energy / (float) last + epsilon);
      for (int i = 0; i < last; i++) {
        dst[base + i] = src[base + i] * inv * weight[i];
      }
    }
  }

  static void addRmsNormRows(
    final float[] left, final float[] right, final float[] summed,
    final float[] weight, final float[] normed, final float[] eps,
    final int rows, final int last
  ) {
    float epsilon = eps[0];
    for (@Parallel int row = 0; row < rows; row++) {
      int base = row * last;
      float energy = 0f;
      for (int i = 0; i < last; i++) {
        float value = left[base + i] + right[base + i];
        summed[base + i] = value;
        energy += value * value;
      }
      float inv = 1f / (float) Math.sqrt(energy / (float) last + epsilon);
      for (int i = 0; i < last; i++) {
        normed[base + i] = summed[base + i] * inv * weight[i];
      }
    }
  }
}
