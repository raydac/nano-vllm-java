package com.igormaznitsa.nanollvm.tensor.tornado;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;

/**
 * Hybrid {@link FloatKernels}: large dense GEMV on TornadoVM, everything else on a CPU backend.
 *
 * @since 1.3.1
 */
public final class TornadoFloatKernels extends FloatKernels {

  static final int MIN_OUT = 256;
  static final int MIN_IN = 256;

  private final FloatKernels delegate;

  TornadoFloatKernels(final FloatKernels delegate) {
    this.delegate = delegate;
  }

  @Override
  public String name() {
    return "TornadoVM gemv + " + this.delegate.name();
  }

  @Override
  public float dot(
    final float[] a, final int aOffset, final float[] b, final int bOffset, final int n
  ) {
    return this.delegate.dot(a, aOffset, b, bOffset, n);
  }

  @Override
  public float sumSquares(final float[] a, final int offset, final int n) {
    return this.delegate.sumSquares(a, offset, n);
  }

  @Override
  public void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  @Override
  public void scaleAddOnePlus(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.scaleAddOnePlus(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  @Override
  public void gemv(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int in, final int out0, final int out1
  ) {
    if (!this.shouldOffloadGemv(in, out0, out1)) {
      this.delegate.gemv(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
      return;
    }
    try {
      TornadoGemvExecutor.gemv(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
    } catch (RuntimeException failed) {
      this.delegate.gemv(x, xOff, w, wOff, bias, y, yOff, in, out0, out1);
    }
  }

  @Override
  public void add(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.add(a, aOff, b, bOff, dst, dstOff, n);
  }

  @Override
  public void mul(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.mul(a, aOff, b, bOff, dst, dstOff, n);
  }

  @Override
  public void scale(
    final float[] src, final int srcOff, final float factor,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.scale(src, srcOff, factor, dst, dstOff, n);
  }

  @Override
  public void axpy(
    final float[] dst, final int dstOff, final float alpha,
    final float[] src, final int srcOff, final int n
  ) {
    this.delegate.axpy(dst, dstOff, alpha, src, srcOff, n);
  }

  @Override
  public float addSumSquares(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    return this.delegate.addSumSquares(a, aOff, b, bOff, dst, dstOff, n);
  }

  @Override
  public void siluMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.siluMul(gate, gateOff, up, upOff, dst, dstOff, n);
  }

  @Override
  public void geluTanh(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.geluTanh(src, srcOff, dst, dstOff, n);
  }

  @Override
  public void geluTanhMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.geluTanhMul(gate, gateOff, up, upOff, dst, dstOff, n);
  }

  @Override
  public void tanhSoftcap(
    final float[] src, final int srcOff, final float cap,
    final float[] dst, final int dstOff, final int n
  ) {
    this.delegate.tanhSoftcap(src, srcOff, cap, dst, dstOff, n);
  }

  private boolean shouldOffloadGemv(final int in, final int out0, final int out1) {
    int outCount = out1 - out0;
    return outCount >= MIN_OUT && in >= MIN_IN;
  }
}
