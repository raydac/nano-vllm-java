package com.igormaznitsa.nanollvm.tensor;

/**
 * Stateless float primitives over {@link FloatKernels} for layer code and {@link Ops}.
 *
 * <p>Dense parallel matmul lives on a per-{@code LLM} {@link MatmulRuntime} (shared or custom
 * {@link java.util.concurrent.ExecutorService}). Call sites that need matmul use
 * {@link com.igormaznitsa.nanollvm.internal.Context#matmul()} or an explicit runtime.
 *
 * @see MatmulRuntime
 * @see FloatKernels
 * @see Ops
 */
public final class VectorMath {

  private static final FloatKernels KERNELS = FloatKernels.get();

  private VectorMath() {
  }

  public static float dot(final float[] a, final int aOffset, final float[] b, final int bOffset,
                          final int n) {
    return KERNELS.dot(a, aOffset, b, bOffset, n);
  }

  public static float sumSquares(final float[] a, final int offset, final int n) {
    return KERNELS.sumSquares(a, offset, n);
  }

  public static void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  public static void scaleAddOnePlus(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.scaleAddOnePlus(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  public static void add(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.add(a, aOff, b, bOff, dst, dstOff, n);
  }

  public static void mul(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.mul(a, aOff, b, bOff, dst, dstOff, n);
  }

  public static void scale(
    final float[] src, final int srcOff, final float factor,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.scale(src, srcOff, factor, dst, dstOff, n);
  }

  public static void axpy(
    final float[] dst, final int dstOff, final float alpha,
    final float[] src, final int srcOff, final int n
  ) {
    KERNELS.axpy(dst, dstOff, alpha, src, srcOff, n);
  }

  public static float addSumSquares(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    return KERNELS.addSumSquares(a, aOff, b, bOff, dst, dstOff, n);
  }

  public static void siluMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.siluMul(gate, gateOff, up, upOff, dst, dstOff, n);
  }

  public static void geluTanh(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.geluTanh(src, srcOff, dst, dstOff, n);
  }

  public static void geluTanhMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.geluTanhMul(gate, gateOff, up, upOff, dst, dstOff, n);
  }

  public static void tanhSoftcap(
    final float[] src, final int srcOff, final float cap,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.tanhSoftcap(src, srcOff, cap, dst, dstOff, n);
  }

  public static String backendInfo() {
    return KERNELS.name();
  }
}
