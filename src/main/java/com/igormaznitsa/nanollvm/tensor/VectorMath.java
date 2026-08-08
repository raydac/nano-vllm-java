package com.igormaznitsa.nanollvm.tensor;

/**
 * Stateless float primitives over {@link FloatKernels} for layer code and {@link Ops}.
 *
 * <p>Dense parallel matmul lives on a per-{@code LLM} {@link MatmulRuntime} (shared or custom
 * {@link java.util.concurrent.ExecutorService}). Call sites that need matmul use
 * {@link MatmulRuntime#current()} (bound on inference
 * {@link com.igormaznitsa.nanollvm.internal.Context}) or an explicit runtime.
 *
 * @see MatmulRuntime
 * @see FloatKernels
 * @see Ops
 */
public final class VectorMath {

  private static final FloatKernels KERNELS = FloatKernels.get();

  private VectorMath() {
  }

  /**
   * Dot product of two equal-length slices; delegates to {@link FloatKernels#dot}.
   */
  public static float dot(final float[] a, final int aOffset, final float[] b, final int bOffset,
                          final int n) {
    return KERNELS.dot(a, aOffset, b, bOffset, n);
  }

  /**
   * Sum of squares over one slice; delegates to {@link FloatKernels#sumSquares}.
   */
  public static float sumSquares(final float[] a, final int offset, final int n) {
    return KERNELS.sumSquares(a, offset, n);
  }

  /**
   * Elementwise {@code dst = src * scale * weight} over slices; delegates to
   * {@link FloatKernels#scaleAdd}.
   */
  public static void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    KERNELS.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  /**
   * Backend label for probes (kernels only; thread count belongs to {@link MatmulRuntime}).
   */
  public static String backendInfo() {
    return KERNELS.name();
  }
}
