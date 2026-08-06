package com.igormaznitsa.nanollvm.tensor.scalar;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;

/**
 * Portable {@link FloatKernels} backend: one Java scalar loop per operation.
 *
 * <p>No Vector API / SIMD dependency. Used when the incubator module is unavailable, when
 * {@code -Dnanovllm.kernels=scalar} is set, or as the reference behaviour for tests.
 * Numerically straightforward; usually slower than {@code VectorFloatKernels} on wide SIMD CPUs.
 *
 * @see com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory
 */
public final class ScalarFloatKernels extends FloatKernels {

  /**
   * {@inheritDoc}
   *
   * @return always {@code "scalar"}
   */
  @Override
  public String name() {
    return "scalar";
  }

  /**
   * {@inheritDoc}
   *
   * <p>Plain loop: {@code sum += a[aOff+i] * b[bOff+i]}.
   */
  @Override
  public float dot(final float[] a, final int aOffset, final float[] b, final int bOffset,
                   final int n) {
    float sum = 0f;
    for (int i = 0; i < n; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Plain loop: {@code sum += v * v} for each element in the slice.
   */
  @Override
  public float sumSquares(final float[] a, final int offset, final int n) {
    float sum = 0f;
    for (int i = 0; i < n; i++) {
      float v = a[offset + i];
      sum += v * v;
    }
    return sum;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Plain loop writing {@code dst[i] = src[i] * scale * weight[i]} over the slices.
   */
  @Override
  public void scaleAdd(
      final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
      final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * scale * weight[wOff + i];
    }
  }
}
