package com.igormaznitsa.nanollvm.tensor.tornado;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import com.igormaznitsa.nanollvm.tensor.scalar.ScalarFloatKernels;

/**
 * TornadoVM {@link FloatKernels}. Dense GEMV uses the Kernel API. Chat attention is one Loop
 * Parallel task per query range. Elementwise work uses Loop Parallel only when the slice is long
 * enough to hide the launch; shorter slices use scalar loops.
 *
 * @since 1.4.0
 */
public final class TornadoFloatKernels extends FloatKernels {

  private static final int MIN_DEVICE_SPAN = 65536;

  private static final FloatKernels SCALAR = new ScalarFloatKernels();

  TornadoFloatKernels() {
  }

  @Override
  public String name() {
    return "TornadoVM";
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
    return this.value(
      n,
      () -> TornadoElementExecutor.dot(a, aOffset, b, bOffset, n),
      () -> SCALAR.dot(a, aOffset, b, bOffset, n)
    );
  }

  @Override
  public float sumSquares(final float[] a, final int offset, final int n) {
    if (n <= 0) {
      return 0f;
    }
    return this.value(
      n,
      () -> TornadoElementExecutor.sumSquares(a, offset, n),
      () -> SCALAR.sumSquares(a, offset, n)
    );
  }

  @Override
  public void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n),
      () -> SCALAR.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n),
      n
    );
  }

  @Override
  public void scaleAddOnePlus(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.scaleAddOnePlus(src, srcOff, weight, wOff, scale, dst, dstOff,
        n),
      () -> SCALAR.scaleAddOnePlus(src, srcOff, weight, wOff, scale, dst, dstOff, n),
      n
    );
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
    try {
      TornadoGemvExecutor.gemv(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
    } catch (RuntimeException failed) {
      SCALAR.gemv(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
    }
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
    try {
      TornadoGemvExecutor.gemvRows(
        x, xOff, w, wOff, bias, y, yOff, rows, in, rowStride, out0, out1);
    } catch (RuntimeException failed) {
      SCALAR.gemvRows(x, xOff, w, wOff, bias, y, yOff, rows, in, rowStride, out0, out1);
    }
  }

  @Override
  public void add(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.add(a, aOff, b, bOff, dst, dstOff, n),
      () -> SCALAR.add(a, aOff, b, bOff, dst, dstOff, n),
      n
    );
  }

  @Override
  public void mul(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.mul(a, aOff, b, bOff, dst, dstOff, n),
      () -> SCALAR.mul(a, aOff, b, bOff, dst, dstOff, n),
      n
    );
  }

  @Override
  public void scale(
    final float[] src, final int srcOff, final float factor,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.scale(src, srcOff, factor, dst, dstOff, n),
      () -> SCALAR.scale(src, srcOff, factor, dst, dstOff, n),
      n
    );
  }

  @Override
  public void axpy(
    final float[] dst, final int dstOff, final float alpha,
    final float[] src, final int srcOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.axpy(dst, dstOff, alpha, src, srcOff, n),
      () -> SCALAR.axpy(dst, dstOff, alpha, src, srcOff, n),
      n
    );
  }

  @Override
  public float addSumSquares(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    if (n <= 0) {
      return 0f;
    }
    return this.value(
      n,
      () -> TornadoElementExecutor.addSumSquares(a, aOff, b, bOff, dst, dstOff, n),
      () -> SCALAR.addSumSquares(a, aOff, b, bOff, dst, dstOff, n)
    );
  }

  @Override
  public void siluMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.siluMul(gate, gateOff, up, upOff, dst, dstOff, n),
      () -> SCALAR.siluMul(gate, gateOff, up, upOff, dst, dstOff, n),
      n
    );
  }

  @Override
  public void geluTanh(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.geluTanh(src, srcOff, dst, dstOff, n),
      () -> SCALAR.geluTanh(src, srcOff, dst, dstOff, n),
      n
    );
  }

  @Override
  public void geluTanhMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.geluTanhMul(gate, gateOff, up, upOff, dst, dstOff, n),
      () -> SCALAR.geluTanhMul(gate, gateOff, up, upOff, dst, dstOff, n),
      n
    );
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
    return TornadoAttentionExecutor.attend(
      query, queryOffset, key, keyOffset, value, valueOffset, result, resultOffset,
      queryStart, queryLength, keyIndexBase, keyLength,
      numHeads, numKvHeads, headDim, scale, slidingWindow, causal, keySlots
    );
  }

  @Override
  public void tanhSoftcap(
    final float[] src, final int srcOff, final float cap,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.tanhSoftcap(src, srcOff, cap, dst, dstOff, n),
      () -> SCALAR.tanhSoftcap(src, srcOff, cap, dst, dstOff, n),
      n
    );
  }

  private void run(final Runnable tornado, final Runnable scalar, final int n) {
    if (n <= 0) {
      return;
    }
    if (n < MIN_DEVICE_SPAN) {
      scalar.run();
      return;
    }
    try {
      tornado.run();
    } catch (RuntimeException failed) {
      scalar.run();
    }
  }

  private float value(final int n, final FloatSupplier tornado, final FloatSupplier scalar) {
    if (n < MIN_DEVICE_SPAN) {
      return scalar.get();
    }
    try {
      return tornado.get();
    } catch (RuntimeException failed) {
      return scalar.get();
    }
  }

  @FunctionalInterface
  private interface FloatSupplier {
    float get();
  }
}
