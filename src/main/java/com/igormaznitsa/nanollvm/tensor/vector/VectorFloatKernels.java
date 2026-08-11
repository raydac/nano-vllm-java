package com.igormaznitsa.nanollvm.tensor.vector;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD {@link FloatKernels} backend using the JDK incubator Vector API.
 *
 * <h2>Hard part — species, main loop, scalar tail</h2>
 * {@link FloatVector#SPECIES_PREFERRED} picks a lane width for the current CPU (e.g. 8 floats on
 * 256-bit AVX). Each kernel:
 * <ol>
 *   <li>Advances {@code i} by {@code SPECIES.length()} while {@code i < SPECIES.loopBound(n)}</li>
 *   <li>Loads vectors with {@link FloatVector#fromArray} at {@code offset + i}</li>
 *   <li>Finishes {@code i .. n-1} with a scalar loop (the <em>tail</em>)</li>
 * </ol>
 * Without the tail, the last {@code n % laneCount} elements would be skipped. Without correct
 * offsets, Vector loads would read from the wrong place in a {@link com.igormaznitsa.nanollvm.tensor.Tensor}
 * view’s backing array.
 *
 * <p>Requires {@code --add-modules jdk.incubator.vector} (see {@code .mvn/jvm.config} /
 * surefire {@code jvm.module.args}). Constructed via {@link com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory}
 * when the module and this class are loadable.
 *
 * <p>FMA and reduction order can make results differ slightly from {@code ScalarFloatKernels};
 * that is expected.
 *
 * @see com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory
 */
public final class VectorFloatKernels extends FloatKernels {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  /**
   * {@inheritDoc}
   *
   * @return species identity and lane count, e.g. {@code Vector API … (len=8)}
   */
  @Override
  public String name() {
    return "Vector API %s (len=%d)".formatted(SPECIES, SPECIES.length());
  }

  /**
   * {@inheritDoc}
   *
   * <p><strong>SIMD:</strong> accumulate {@code va.fma(vb, acc)} over full vectors, then
   * {@code reduceLanes(ADD)}, then scalar multiply-add for the tail.
   */
  @Override
  public float dot(final float[] a, final int aOffset, final float[] b, final int bOffset,
                   final int n) {
    int i = 0;
    FloatVector acc = FloatVector.zero(SPECIES);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(SPECIES, a, aOffset + i);
      FloatVector vb = FloatVector.fromArray(SPECIES, b, bOffset + i);
      acc = va.fma(vb, acc);
    }
    float sum = acc.reduceLanes(VectorOperators.ADD);
    for (; i < n; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  /**
   * {@inheritDoc}
   *
   * <p><strong>SIMD:</strong> {@code v.fma(v, acc)} squares each lane in-register; scalar tail
   * mirrors the scalar backend.
   */
  @Override
  public float sumSquares(final float[] a, final int offset, final int n) {
    int i = 0;
    FloatVector acc = FloatVector.zero(SPECIES);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += SPECIES.length()) {
      FloatVector v = FloatVector.fromArray(SPECIES, a, offset + i);
      acc = v.fma(v, acc);
    }
    float sum = acc.reduceLanes(VectorOperators.ADD);
    for (; i < n; i++) {
      float v = a[offset + i];
      sum += v * v;
    }
    return sum;
  }

  /**
   * {@inheritDoc}
   *
   * <p><strong>SIMD:</strong> broadcast {@code scale} once; store
   * {@code (src * scale) * weight} lanes into {@code dst}; scalar tail for the remainder.
   * Destination may alias neither or both sources; overlapping partially is undefined for SIMD
   * stores (callers use distinct buffers in this engine).
   */
  @Override
  public void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    FloatVector vScale = FloatVector.broadcast(SPECIES, scale);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += SPECIES.length()) {
      FloatVector vs = FloatVector.fromArray(SPECIES, src, srcOff + i);
      FloatVector vw = FloatVector.fromArray(SPECIES, weight, wOff + i);
      vs.mul(vScale).mul(vw).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * scale * weight[wOff + i];
    }
  }
}
