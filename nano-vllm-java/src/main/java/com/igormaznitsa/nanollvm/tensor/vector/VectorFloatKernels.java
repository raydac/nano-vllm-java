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
  private static final int LANE = SPECIES.length();
  private static final int UNROLL = 4;
  private static final int GEMV_PANEL = 8;
  private static final float GELU_K = 0.044715f;
  private static final float GELU_SQRT_2_OVER_PI = 0.7978846f;

  private static FloatVector geluTanhLanes(final FloatVector x) {
    FloatVector x3 = x.mul(x).mul(x);
    return x.mul(0.5f).mul(
      x.add(x3.mul(GELU_K)).mul(GELU_SQRT_2_OVER_PI).lanewise(VectorOperators.TANH).add(1.0f));
  }

  private static float geluPytorchTanh(final float x) {
    return 0.5f * x * (1.0f + (float) Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  }

  @Override
  public String name() {
    return "Vector API %s (len=%d)".formatted(SPECIES, LANE);
  }

  @Override
  public float dot(final float[] a, final int aOffset, final float[] b, final int bOffset,
                   final int n) {
    int i = 0;
    int upper = SPECIES.loopBound(n);
    int unrollSpan = LANE * UNROLL;
    int unrollBound = upper - (upper % unrollSpan);
    FloatVector acc0 = FloatVector.zero(SPECIES);
    FloatVector acc1 = FloatVector.zero(SPECIES);
    FloatVector acc2 = FloatVector.zero(SPECIES);
    FloatVector acc3 = FloatVector.zero(SPECIES);
    for (; i < unrollBound; i += unrollSpan) {
      acc0 = FloatVector.fromArray(SPECIES, a, aOffset + i)
        .fma(FloatVector.fromArray(SPECIES, b, bOffset + i), acc0);
      acc1 = FloatVector.fromArray(SPECIES, a, aOffset + i + LANE)
        .fma(FloatVector.fromArray(SPECIES, b, bOffset + i + LANE), acc1);
      acc2 = FloatVector.fromArray(SPECIES, a, aOffset + i + LANE * 2)
        .fma(FloatVector.fromArray(SPECIES, b, bOffset + i + LANE * 2), acc2);
      acc3 = FloatVector.fromArray(SPECIES, a, aOffset + i + LANE * 3)
        .fma(FloatVector.fromArray(SPECIES, b, bOffset + i + LANE * 3), acc3);
    }
    for (; i < upper; i += LANE) {
      acc0 = FloatVector.fromArray(SPECIES, a, aOffset + i)
        .fma(FloatVector.fromArray(SPECIES, b, bOffset + i), acc0);
    }
    float sum = acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
    for (; i < n; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  @Override
  public float sumSquares(final float[] a, final int offset, final int n) {
    int i = 0;
    int upper = SPECIES.loopBound(n);
    int unrollSpan = LANE * UNROLL;
    int unrollBound = upper - (upper % unrollSpan);
    FloatVector acc0 = FloatVector.zero(SPECIES);
    FloatVector acc1 = FloatVector.zero(SPECIES);
    FloatVector acc2 = FloatVector.zero(SPECIES);
    FloatVector acc3 = FloatVector.zero(SPECIES);
    for (; i < unrollBound; i += unrollSpan) {
      FloatVector v0 = FloatVector.fromArray(SPECIES, a, offset + i);
      FloatVector v1 = FloatVector.fromArray(SPECIES, a, offset + i + LANE);
      FloatVector v2 = FloatVector.fromArray(SPECIES, a, offset + i + LANE * 2);
      FloatVector v3 = FloatVector.fromArray(SPECIES, a, offset + i + LANE * 3);
      acc0 = v0.fma(v0, acc0);
      acc1 = v1.fma(v1, acc1);
      acc2 = v2.fma(v2, acc2);
      acc3 = v3.fma(v3, acc3);
    }
    for (; i < upper; i += LANE) {
      FloatVector v = FloatVector.fromArray(SPECIES, a, offset + i);
      acc0 = v.fma(v, acc0);
    }
    float sum = acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
    for (; i < n; i++) {
      float v = a[offset + i];
      sum += v * v;
    }
    return sum;
  }

  @Override
  public void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    FloatVector vScale = FloatVector.broadcast(SPECIES, scale);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector vs = FloatVector.fromArray(SPECIES, src, srcOff + i);
      FloatVector vw = FloatVector.fromArray(SPECIES, weight, wOff + i);
      vs.mul(vScale).mul(vw).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * scale * weight[wOff + i];
    }
  }

  @Override
  public void scaleAddOnePlus(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    FloatVector vScale = FloatVector.broadcast(SPECIES, scale);
    FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector vs = FloatVector.fromArray(SPECIES, src, srcOff + i);
      FloatVector vw = FloatVector.fromArray(SPECIES, weight, wOff + i);
      vs.mul(vScale).mul(vw.add(one)).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * scale * (1.0f + weight[wOff + i]);
    }
  }

  @Override
  public void gemv(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int in, final int out0, final int out1
  ) {
    int o = out0;
    for (; o + GEMV_PANEL <= out1; o += GEMV_PANEL) {
      this.gemvPanel8(x, xOff, w, wOff, bias, y, yOff, in, o);
    }
    for (; o < out1; o++) {
      float sum = bias != null ? bias[o] : 0f;
      y[yOff + o] = sum + this.dot(x, xOff, w, wOff + o * in, in);
    }
  }

  private void gemvPanel8(
    final float[] x, final int xOff,
    final float[] w, final int wOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int in, final int out0
  ) {
    int w0 = wOff + out0 * in;
    int w1 = w0 + in;
    int w2 = w1 + in;
    int w3 = w2 + in;
    int w4 = w3 + in;
    int w5 = w4 + in;
    int w6 = w5 + in;
    int w7 = w6 + in;
    FloatVector acc0 = FloatVector.zero(SPECIES);
    FloatVector acc1 = FloatVector.zero(SPECIES);
    FloatVector acc2 = FloatVector.zero(SPECIES);
    FloatVector acc3 = FloatVector.zero(SPECIES);
    FloatVector acc4 = FloatVector.zero(SPECIES);
    FloatVector acc5 = FloatVector.zero(SPECIES);
    FloatVector acc6 = FloatVector.zero(SPECIES);
    FloatVector acc7 = FloatVector.zero(SPECIES);
    int i = 0;
    int upper = SPECIES.loopBound(in);
    for (; i < upper; i += LANE) {
      FloatVector vx = FloatVector.fromArray(SPECIES, x, xOff + i);
      acc0 = vx.fma(FloatVector.fromArray(SPECIES, w, w0 + i), acc0);
      acc1 = vx.fma(FloatVector.fromArray(SPECIES, w, w1 + i), acc1);
      acc2 = vx.fma(FloatVector.fromArray(SPECIES, w, w2 + i), acc2);
      acc3 = vx.fma(FloatVector.fromArray(SPECIES, w, w3 + i), acc3);
      acc4 = vx.fma(FloatVector.fromArray(SPECIES, w, w4 + i), acc4);
      acc5 = vx.fma(FloatVector.fromArray(SPECIES, w, w5 + i), acc5);
      acc6 = vx.fma(FloatVector.fromArray(SPECIES, w, w6 + i), acc6);
      acc7 = vx.fma(FloatVector.fromArray(SPECIES, w, w7 + i), acc7);
    }
    float s0 = acc0.reduceLanes(VectorOperators.ADD);
    float s1 = acc1.reduceLanes(VectorOperators.ADD);
    float s2 = acc2.reduceLanes(VectorOperators.ADD);
    float s3 = acc3.reduceLanes(VectorOperators.ADD);
    float s4 = acc4.reduceLanes(VectorOperators.ADD);
    float s5 = acc5.reduceLanes(VectorOperators.ADD);
    float s6 = acc6.reduceLanes(VectorOperators.ADD);
    float s7 = acc7.reduceLanes(VectorOperators.ADD);
    for (; i < in; i++) {
      float xv = x[xOff + i];
      s0 += xv * w[w0 + i];
      s1 += xv * w[w1 + i];
      s2 += xv * w[w2 + i];
      s3 += xv * w[w3 + i];
      s4 += xv * w[w4 + i];
      s5 += xv * w[w5 + i];
      s6 += xv * w[w6 + i];
      s7 += xv * w[w7 + i];
    }
    if (bias != null) {
      s0 += bias[out0];
      s1 += bias[out0 + 1];
      s2 += bias[out0 + 2];
      s3 += bias[out0 + 3];
      s4 += bias[out0 + 4];
      s5 += bias[out0 + 5];
      s6 += bias[out0 + 6];
      s7 += bias[out0 + 7];
    }
    y[yOff + out0] = s0;
    y[yOff + out0 + 1] = s1;
    y[yOff + out0 + 2] = s2;
    y[yOff + out0 + 3] = s3;
    y[yOff + out0 + 4] = s4;
    y[yOff + out0 + 5] = s5;
    y[yOff + out0 + 6] = s6;
    y[yOff + out0 + 7] = s7;
  }

  @Override
  public void add(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector.fromArray(SPECIES, a, aOff + i)
        .add(FloatVector.fromArray(SPECIES, b, bOff + i))
        .intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = a[aOff + i] + b[bOff + i];
    }
  }

  @Override
  public void mul(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector.fromArray(SPECIES, a, aOff + i)
        .mul(FloatVector.fromArray(SPECIES, b, bOff + i))
        .intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = a[aOff + i] * b[bOff + i];
    }
  }

  @Override
  public void scale(
    final float[] src, final int srcOff, final float factor,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    FloatVector vFactor = FloatVector.broadcast(SPECIES, factor);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector.fromArray(SPECIES, src, srcOff + i).mul(vFactor).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * factor;
    }
  }

  @Override
  public void axpy(
    final float[] dst, final int dstOff, final float alpha,
    final float[] src, final int srcOff, final int n
  ) {
    int i = 0;
    FloatVector vAlpha = FloatVector.broadcast(SPECIES, alpha);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector vd = FloatVector.fromArray(SPECIES, dst, dstOff + i);
      FloatVector vs = FloatVector.fromArray(SPECIES, src, srcOff + i);
      vs.fma(vAlpha, vd).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] += alpha * src[srcOff + i];
    }
  }

  @Override
  public float addSumSquares(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    FloatVector acc = FloatVector.zero(SPECIES);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector v = FloatVector.fromArray(SPECIES, a, aOff + i)
        .add(FloatVector.fromArray(SPECIES, b, bOff + i));
      v.intoArray(dst, dstOff + i);
      acc = v.fma(v, acc);
    }
    float sum = acc.reduceLanes(VectorOperators.ADD);
    for (; i < n; i++) {
      float v = a[aOff + i] + b[bOff + i];
      dst[dstOff + i] = v;
      sum += v * v;
    }
    return sum;
  }

  @Override
  public void siluMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector g = FloatVector.fromArray(SPECIES, gate, gateOff + i);
      FloatVector u = FloatVector.fromArray(SPECIES, up, upOff + i);
      g.div(g.neg().lanewise(VectorOperators.EXP).add(one)).mul(u).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      float g = gate[gateOff + i];
      dst[dstOff + i] = (g / (1.0f + (float) Math.exp(-g))) * up[upOff + i];
    }
  }

  @Override
  public void geluTanh(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      geluTanhLanes(FloatVector.fromArray(SPECIES, src, srcOff + i)).intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = geluPytorchTanh(src[srcOff + i]);
    }
  }

  @Override
  public void geluTanhMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      geluTanhLanes(FloatVector.fromArray(SPECIES, gate, gateOff + i))
        .mul(FloatVector.fromArray(SPECIES, up, upOff + i))
        .intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = geluPytorchTanh(gate[gateOff + i]) * up[upOff + i];
    }
  }

  @Override
  public void tanhSoftcap(
    final float[] src, final int srcOff, final float cap,
    final float[] dst, final int dstOff, final int n
  ) {
    int i = 0;
    FloatVector vCap = FloatVector.broadcast(SPECIES, cap);
    FloatVector vInvCap = FloatVector.broadcast(SPECIES, 1.0f / cap);
    int upper = SPECIES.loopBound(n);
    for (; i < upper; i += LANE) {
      FloatVector.fromArray(SPECIES, src, srcOff + i)
        .mul(vInvCap)
        .lanewise(VectorOperators.TANH)
        .mul(vCap)
        .intoArray(dst, dstOff + i);
    }
    for (; i < n; i++) {
      dst[dstOff + i] = (float) Math.tanh(src[srcOff + i] / cap) * cap;
    }
  }
}
