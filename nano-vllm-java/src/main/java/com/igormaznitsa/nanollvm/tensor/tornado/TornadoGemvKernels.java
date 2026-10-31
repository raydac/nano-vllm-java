package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * TornadoVM dense GEMV. The fast kernel follows TornadoVM's row-major example: one workgroup per
 * output row, consecutive lanes reading consecutive weights, then a local reduction. Plain Kernel
 * API and Loop Parallel remain fallbacks.
 *
 * @since 1.4.0
 */
final class TornadoGemvKernels {

  static final int ROW_LANES = 128;

  static final float[] NO_BIAS = new float[1];

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
    int group = ctx.groupIdx;
    int lane = ctx.localIdx;
    int token = group / outCount;
    int col = group - token * outCount;
    int row = out0 + col;
    int xBase = token * in;
    int wBase = wOff + row * in;
    float[] partials = ctx.allocateFloatLocalArray(laneWidth);
    float lane0 = 0f;
    float lane1 = 0f;
    float lane2 = 0f;
    float lane3 = 0f;
    int stride = laneWidth * 4;
    int aligned = in & ~3;
    for (int k = lane * 4; k < aligned; k += stride) {
      lane0 += x[xBase + k] * w[wBase + k];
      lane1 += x[xBase + k + 1] * w[wBase + k + 1];
      lane2 += x[xBase + k + 2] * w[wBase + k + 2];
      lane3 += x[xBase + k + 3] * w[wBase + k + 3];
    }
    float partial = lane0 + lane1 + lane2 + lane3;
    for (int k = aligned + lane; k < in; k += laneWidth) {
      partial += x[xBase + k] * w[wBase + k];
    }
    partials[lane] = partial;
    ctx.localBarrier();
    for (int step = laneWidth >> 1; step > 0; step >>= 1) {
      if (lane < step) {
        partials[lane] += partials[lane + step];
      }
      ctx.localBarrier();
    }
    if (lane == 0) {
      float sum = partials[0];
      if (hasBias != 0) {
        sum += bias[row];
      }
      y[token * outCount + col] = sum;
    }
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
