package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * TornadoVM dense GEMV: tiled Kernel API, plain Kernel API, then Loop Parallel.
 *
 * @since 1.4.0
 */
final class TornadoGemvKernels {

  static final int ACTIVATION_TILE = 128;

  static final float[] NO_BIAS = new float[0];

  private TornadoGemvKernels() {
  }

  static void gemvTiled(
    final KernelContext ctx,
    final float[] x,
    final float[] w, final int wOff,
    final float[] bias, final int hasBias,
    final float[] y,
    final int in, final int out0, final int outCount, final int laneWidth
  ) {
    int t = ctx.globalIdx;
    int token = t / outCount;
    int col = t - token * outCount;
    int lane = ctx.localIdx;
    int o = out0 + col;
    float sum = 0f;
    if (hasBias != 0) {
      sum = bias[o];
    }
    int xBase = token * in;
    int wBase = wOff + o * in;
    float[] tile = ctx.allocateFloatLocalArray(ACTIVATION_TILE);
    for (int base = 0; base < in; base += ACTIVATION_TILE) {
      for (int i = lane; i < ACTIVATION_TILE; i += laneWidth) {
        int k = base + i;
        tile[i] = k < in ? x[xBase + k] : 0f;
      }
      ctx.localBarrier();
      int limit = Math.min(ACTIVATION_TILE, in - base);
      for (int i = 0; i < limit; i++) {
        sum += tile[i] * w[wBase + base + i];
      }
      ctx.localBarrier();
    }
    y[token * outCount + col] = sum;
  }

  static void gemvPlain(
    final KernelContext ctx,
    final float[] x,
    final float[] w, final int wOff,
    final float[] bias, final int hasBias,
    final float[] y,
    final int in, final int out0, final int outCount
  ) {
    int t = ctx.globalIdx;
    int token = t / outCount;
    int col = t - token * outCount;
    int o = out0 + col;
    float sum = 0f;
    if (hasBias != 0) {
      sum = bias[o];
    }
    int xBase = token * in;
    int wBase = wOff + o * in;
    int i = 0;
    int inAligned = in & ~3;
    for (; i < inAligned; i += 4) {
      sum += x[xBase + i] * w[wBase + i]
        + x[xBase + i + 1] * w[wBase + i + 1]
        + x[xBase + i + 2] * w[wBase + i + 2]
        + x[xBase + i + 3] * w[wBase + i + 3];
    }
    for (; i < in; i++) {
      sum += x[xBase + i] * w[wBase + i];
    }
    y[token * outCount + col] = sum;
  }

  static void gemvParallel(
    final float[] x,
    final float[] w, final int wOff,
    final float[] bias, final int hasBias,
    final float[] y,
    final int in, final int out0, final int outCount, final int work
  ) {
    for (@Parallel int t = 0; t < work; t++) {
      int token = t / outCount;
      int col = t - token * outCount;
      int o = out0 + col;
      float sum = 0f;
      if (hasBias != 0) {
        sum = bias[o];
      }
      int xBase = token * in;
      int wBase = wOff + o * in;
      for (int i = 0; i < in; i++) {
        sum += x[xBase + i] * w[wBase + i];
      }
      y[token * outCount + col] = sum;
    }
  }
}
