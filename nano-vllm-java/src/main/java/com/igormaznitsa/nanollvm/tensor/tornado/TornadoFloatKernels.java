package com.igormaznitsa.nanollvm.tensor.tornado;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import java.util.Objects;

/**
 * TornadoVM {@link FloatKernels}. A multi-token dense GEMV uses CLBlast when it runs, otherwise the
 * Kernel API. One token reads each weight matrix once, so that matmul and a one-query attention
 * step stay on the CPU backend. Elementwise work uses the device only when the slice is long
 * enough to hide the launch.
 *
 * @since 1.4.0
 */
public final class TornadoFloatKernels extends FloatKernels {

  private static final int MIN_DEVICE_SPAN = 65536;

  private final FloatKernels cpu;

  TornadoFloatKernels(final FloatKernels cpu) {
    this.cpu = Objects.requireNonNull(cpu, "cpu");
  }

  @Override
  public String name() {
    return "TornadoVM";
  }

  @Override
  public void releaseWeights(final float[]... weights) {
    TornadoGemvExecutor.releaseWeights(weights);
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
      () -> this.cpu.dot(a, aOffset, b, bOffset, n)
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
      () -> this.cpu.sumSquares(a, offset, n)
    );
  }

  @Override
  public void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n),
      () -> this.cpu.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n),
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
      () -> this.cpu.scaleAddOnePlus(src, srcOff, weight, wOff, scale, dst, dstOff, n),
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
    this.cpu.gemv(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
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
    if (rows == 1) {
      this.cpu.gemvRows(x, xOff, w, wOff, bias, y, yOff, rows, in, rowStride, out0, out1);
      return;
    }
    try {
      TornadoGemvExecutor.gemvRows(
        x, xOff, w, wOff, bias, y, yOff, rows, in, rowStride, out0, out1);
    } catch (RuntimeException failed) {
      this.cpu.gemvRows(x, xOff, w, wOff, bias, y, yOff, rows, in, rowStride, out0, out1);
    }
  }

  @Override
  public void add(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.add(a, aOff, b, bOff, dst, dstOff, n),
      () -> this.cpu.add(a, aOff, b, bOff, dst, dstOff, n),
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
      () -> this.cpu.mul(a, aOff, b, bOff, dst, dstOff, n),
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
      () -> this.cpu.scale(src, srcOff, factor, dst, dstOff, n),
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
      () -> this.cpu.axpy(dst, dstOff, alpha, src, srcOff, n),
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
      () -> this.cpu.addSumSquares(a, aOff, b, bOff, dst, dstOff, n)
    );
  }

  @Override
  public void siluMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.siluMul(gate, gateOff, up, upOff, dst, dstOff, n),
      () -> this.cpu.siluMul(gate, gateOff, up, upOff, dst, dstOff, n),
      n
    );
  }

  @Override
  public void geluTanh(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    this.run(
      () -> TornadoElementExecutor.geluTanh(src, srcOff, dst, dstOff, n),
      () -> this.cpu.geluTanh(src, srcOff, dst, dstOff, n),
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
      () -> this.cpu.geluTanhMul(gate, gateOff, up, upOff, dst, dstOff, n),
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
    if (queryLength <= 1) {
      return false;
    }
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
      () -> this.cpu.tanhSoftcap(src, srcOff, cap, dst, dstOff, n),
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
