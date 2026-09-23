package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;

/**
 * TornadoVM Loop Parallel kernels for the elementwise and reduction half of {@code FloatKernels}.
 * Slices are packed at index {@code 0}; the host copies offsets into these buffers.
 *
 * @since 1.5.0
 */
final class TornadoElementKernels {

  private TornadoElementKernels() {
  }

  static void dotProduct(final float[] left, final float[] right, @Reduce final float[] sum) {
    for (@Parallel int i = 0; i < left.length; i++) {
      sum[0] += left[i] * right[i];
    }
  }

  static void sumSquares(final float[] values, @Reduce final float[] sum) {
    for (@Parallel int i = 0; i < values.length; i++) {
      float value = values[i];
      sum[0] += value * value;
    }
  }

  static void add(final float[] left, final float[] right, final float[] dst, final int n) {
    for (@Parallel int i = 0; i < n; i++) {
      dst[i] = left[i] + right[i];
    }
  }

  static void mul(final float[] left, final float[] right, final float[] dst, final int n) {
    for (@Parallel int i = 0; i < n; i++) {
      dst[i] = left[i] * right[i];
    }
  }

  static void scale(final float[] src, final float[] factor, final float[] dst, final int n) {
    for (@Parallel int i = 0; i < n; i++) {
      dst[i] = src[i] * factor[0];
    }
  }

  static void axpy(final float[] dst, final float[] src, final float[] alpha, final int n) {
    for (@Parallel int i = 0; i < n; i++) {
      dst[i] += alpha[0] * src[i];
    }
  }

  static void scaleAdd(
    final float[] src, final float[] weight, final float[] scale, final float[] dst, final int n
  ) {
    for (@Parallel int i = 0; i < n; i++) {
      dst[i] = src[i] * scale[0] * weight[i];
    }
  }

  static void scaleAddOnePlus(
    final float[] src, final float[] weight, final float[] scale, final float[] dst, final int n
  ) {
    for (@Parallel int i = 0; i < n; i++) {
      dst[i] = src[i] * scale[0] * (1.0f + weight[i]);
    }
  }

  static void siluMul(
    final float[] gate, final float[] up, final float[] dst, final int n
  ) {
    for (@Parallel int i = 0; i < n; i++) {
      float value = gate[i];
      dst[i] = (value / (1.0f + (float) Math.exp(-value))) * up[i];
    }
  }

  static void geluTanh(final float[] src, final float[] dst, final int n) {
    for (@Parallel int i = 0; i < n; i++) {
      float x = src[i];
      dst[i] = 0.5f * x * (1.0f + (float) Math.tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
  }

  static void geluTanhMul(final float[] gate, final float[] up, final float[] dst, final int n) {
    for (@Parallel int i = 0; i < n; i++) {
      float x = gate[i];
      dst[i] =
        0.5f * x * (1.0f + (float) Math.tanh(0.79788456f * (x + 0.044715f * x * x * x))) * up[i];
    }
  }

  static void tanhSoftcap(final float[] src, final float[] cap, final float[] dst, final int n) {
    for (@Parallel int i = 0; i < n; i++) {
      dst[i] = (float) Math.tanh(src[i] / cap[0]) * cap[0];
    }
  }
}
