package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * TornadoVM dense GEMV: Kernel API first, Loop Parallel fallback.
 *
 * @see <a href="https://www.tornadovm.org/">TornadoVM SGEMV API ladder</a>
 * @since 1.4.0
 */
final class TornadoGemvKernels {

  static final float[] NO_BIAS = new float[0];

  private TornadoGemvKernels() {
  }

  static void gemvKernel(
    final KernelContext ctx,
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias, final int hasBias,
    final float[] y, final int yOff,
    final int in, final int out0, final int outCount
  ) {
    int t = ctx.globalIdx;
    int o = out0 + t;
    float sum = 0f;
    if (hasBias != 0) {
      sum = bias[o];
    }
    int row = wOff + o * in;
    for (int i = 0; i < in; i++) {
      sum += x[xOff + i] * w[row + i];
    }
    y[yOff + o] = sum;
  }

  static void gemvParallel(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias, final int hasBias,
    final float[] y, final int yOff,
    final int in, final int out0, final int outCount
  ) {
    for (@Parallel int t = 0; t < outCount; t++) {
      int o = out0 + t;
      float sum = 0f;
      if (hasBias != 0) {
        sum = bias[o];
      }
      int row = wOff + o * in;
      for (int i = 0; i < in; i++) {
        sum += x[xOff + i] * w[row + i];
      }
      y[yOff + o] = sum;
    }
  }
}
