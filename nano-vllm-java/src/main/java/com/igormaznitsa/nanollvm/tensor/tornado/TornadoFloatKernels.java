package com.igormaznitsa.nanollvm.tensor.tornado;

import com.igormaznitsa.nanollvm.layers.DeviceDecoders;
import com.igormaznitsa.nanollvm.tensor.FloatKernels;

/**
 * TornadoVM {@link FloatKernels}. Dense GEMV, attention, and elementwise work run as device tasks.
 * One-token GEMV uses the device kernel. Multi-token GEMV uses CLBlast when it runs, and the same
 * device kernel when CLBlast does not.
 *
 * @since 1.4.0
 */
public final class TornadoFloatKernels extends FloatKernels {

  TornadoFloatKernels() {
    DeviceDecoders.register(TornadoDecoder::forward);
  }

  @Override
  public String name() {
    return "TornadoVM";
  }

  @Override
  public void releaseWeights(final float[]... weights) {
    TornadoGemvExecutor.releaseWeights(weights);
    TornadoDecoder.releaseWeights(weights);
  }

  @Override
  public boolean prefersSingleShotGemv() {
    return true;
  }

  @Override
  public float dot(
    final float[] a, final int aOffset, final float[] b, final int bOffset, final int n
  ) {
    if (n <= 0) {
      return 0f;
    }
    return TornadoElementExecutor.dot(a, aOffset, b, bOffset, n);
  }

  @Override
  public float sumSquares(final float[] a, final int offset, final int n) {
    if (n <= 0) {
      return 0f;
    }
    return TornadoElementExecutor.sumSquares(a, offset, n);
  }

  @Override
  public void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  @Override
  public void scaleAddOnePlus(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.scaleAddOnePlus(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  @Override
  public void gemv(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int in, final int out0, final int out1
  ) {
    if (out1 <= out0 || in < 0) {
      return;
    }
    TornadoGemvExecutor.gemv(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
  }

  @Override
  public void gemvRows(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int rows, final int in, final int rowStride,
    final int out0, final int out1
  ) {
    if (rows <= 0 || out1 <= out0 || in < 0) {
      return;
    }
    TornadoGemvExecutor.gemvRows(
      x, xOff, w, wOff, bias, y, yOff, rows, in, rowStride, out0, out1);
  }

  @Override
  public void add(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.add(a, aOff, b, bOff, dst, dstOff, n);
  }

  @Override
  public void mul(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.mul(a, aOff, b, bOff, dst, dstOff, n);
  }

  @Override
  public void scale(
    final float[] src, final int srcOff, final float factor,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.scale(src, srcOff, factor, dst, dstOff, n);
  }

  @Override
  public void axpy(
    final float[] dst, final int dstOff, final float alpha,
    final float[] src, final int srcOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.axpy(dst, dstOff, alpha, src, srcOff, n);
  }

  @Override
  public float addSumSquares(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return 0f;
    }
    return TornadoElementExecutor.addSumSquares(a, aOff, b, bOff, dst, dstOff, n);
  }

  @Override
  public void siluMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.siluMul(gate, gateOff, up, upOff, dst, dstOff, n);
  }

  @Override
  public void geluTanh(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.geluTanh(src, srcOff, dst, dstOff, n);
  }

  @Override
  public void geluTanhMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.geluTanhMul(gate, gateOff, up, upOff, dst, dstOff, n);
  }

  @Override
  public boolean rmsNormRows(
    final float[] src, final int srcOff,
    final float[] weight, final int weightOff,
    final float eps, final boolean onePlusWeight,
    final float[] dst, final int dstOff,
    final int rows, final int last
  ) {
    TornadoNormExecutor.rmsNorm(
      src, srcOff, weight, weightOff, eps, onePlusWeight, dst, dstOff, rows, last);
    return true;
  }

  @Override
  public boolean addRmsNormRows(
    final float[] left, final int leftOff,
    final float[] right, final int rightOff,
    final float[] weight, final int weightOff,
    final float eps, final boolean onePlusWeight,
    final float[] summed, final int summedOff,
    final float[] normed, final int normedOff,
    final int rows, final int last
  ) {
    TornadoNormExecutor.addRmsNorm(
      left, leftOff, right, rightOff, weight, weightOff, eps, onePlusWeight,
      summed, summedOff, normed, normedOff, rows, last);
    return true;
  }

  @Override
  public boolean attend(
    final float[] query, final int queryOffset,
    final float[] key, final int keyOffset,
    final float[] value, final int valueOffset,
    final float[] result, final int resultOffset,
    final int queryStart, final int queryLength, final int keyIndexBase, final int keyLength,
    final int numHeads, final int numKvHeads, final int headDim,
    final float scale, final int slidingWindow,
    final boolean causal, final int[] keySlots
  ) {
    if (queryLength <= 0) {
      return true;
    }
    if (!TornadoAttentionExecutor.attend(
      query, queryOffset, key, keyOffset, value, valueOffset, result, resultOffset,
      queryStart, queryLength, keyIndexBase, keyLength,
      numHeads, numKvHeads, headDim, scale, slidingWindow, causal, keySlots
    )) {
      throw new IllegalStateException("TornadoVM attention rejected the query range");
    }
    return true;
  }

  @Override
  public void tanhSoftcap(
    final float[] src, final int srcOff, final float cap,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return;
    }
    TornadoElementExecutor.tanhSoftcap(src, srcOff, cap, dst, dstOff, n);
  }
}
