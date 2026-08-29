package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * TornadoVM-parallel dense GEMV: {@code y[o] = bias[o] + dot(x, W[o])} for {@code o in [out0, out1)}.
 *
 * @since 1.3.1
 */
final class TornadoGemvKernels {

  static final float[] NO_BIAS = new float[0];

  private TornadoGemvKernels() {
  }

  static void gemv(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias, final int hasBias,
    final float[] y, final int yOff,
    final int in, final int out0, final int out1
  ) {
    for (@Parallel int o = out0; o < out1; o++) {
      float sum = hasBias != 0 ? bias[o] : 0f;
      int row = wOff + o * in;
      for (int i = 0; i < in; i++) {
        sum += x[xOff + i] * w[row + i];
      }
      y[yOff + o] = sum;
    }
  }
}
