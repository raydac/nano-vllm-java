package com.igormaznitsa.nanollvm.tensor.scalar;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;

/**
 * Portable {@link FloatKernels} backend: one Java scalar loop per operation.
 *
 * <p>No Vector API / SIMD dependency. Used when the incubator module is unavailable, when
 * {@code -Dnanollvm.kernels=scalar} is set, or as the reference behaviour for tests.
 * Numerically straightforward; usually slower than {@code VectorFloatKernels} on wide SIMD CPUs.
 *
 * @see com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory
 */
public final class ScalarFloatKernels extends FloatKernels {

  private static final int GEMV_PANEL = 8;

  @Override
  public String name() {
    return "scalar";
  }

  private static float geluPytorchTanh(final float x) {
    return 0.5f * x * (1.0f + (float) Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  }

  @Override
  public float dot(final float[] a, final int aOffset, final float[] b, final int bOffset,
                   final int n) {
    float sum = 0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
      sum += a[aOffset + i] * b[bOffset + i]
        + a[aOffset + i + 1] * b[bOffset + i + 1]
        + a[aOffset + i + 2] * b[bOffset + i + 2]
        + a[aOffset + i + 3] * b[bOffset + i + 3];
    }
    for (; i < n; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  @Override
  public void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * scale * weight[wOff + i];
    }
  }

  @Override
  public float sumSquares(final float[] a, final int offset, final int n) {
    float sum = 0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
      float v0 = a[offset + i];
      float v1 = a[offset + i + 1];
      float v2 = a[offset + i + 2];
      float v3 = a[offset + i + 3];
      sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for (; i < n; i++) {
      float v = a[offset + i];
      sum += v * v;
    }
    return sum;
  }

  @Override
  public void scaleAddOnePlus(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
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
    float a0 = bias != null ? bias[out0] : 0f;
    float a1 = bias != null ? bias[out0 + 1] : 0f;
    float a2 = bias != null ? bias[out0 + 2] : 0f;
    float a3 = bias != null ? bias[out0 + 3] : 0f;
    float a4 = bias != null ? bias[out0 + 4] : 0f;
    float a5 = bias != null ? bias[out0 + 5] : 0f;
    float a6 = bias != null ? bias[out0 + 6] : 0f;
    float a7 = bias != null ? bias[out0 + 7] : 0f;
    for (int i = 0; i < in; i++) {
      float xv = x[xOff + i];
      a0 += xv * w[w0 + i];
      a1 += xv * w[w1 + i];
      a2 += xv * w[w2 + i];
      a3 += xv * w[w3 + i];
      a4 += xv * w[w4 + i];
      a5 += xv * w[w5 + i];
      a6 += xv * w[w6 + i];
      a7 += xv * w[w7 + i];
    }
    y[yOff + out0] = a0;
    y[yOff + out0 + 1] = a1;
    y[yOff + out0 + 2] = a2;
    y[yOff + out0 + 3] = a3;
    y[yOff + out0 + 4] = a4;
    y[yOff + out0 + 5] = a5;
    y[yOff + out0 + 6] = a6;
    y[yOff + out0 + 7] = a7;
  }

  @Override
  public void add(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = a[aOff + i] + b[bOff + i];
    }
  }

  @Override
  public void mul(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = a[aOff + i] * b[bOff + i];
    }
  }

  @Override
  public void scale(
    final float[] src, final int srcOff, final float factor,
    final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * factor;
    }
  }

  @Override
  public void axpy(
    final float[] dst, final int dstOff, final float alpha,
    final float[] src, final int srcOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] += alpha * src[srcOff + i];
    }
  }

  @Override
  public float addSumSquares(
    final float[] a, final int aOff, final float[] b, final int bOff,
    final float[] dst, final int dstOff, final int n
  ) {
    float sum = 0f;
    for (int i = 0; i < n; i++) {
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
    for (int i = 0; i < n; i++) {
      float g = gate[gateOff + i];
      dst[dstOff + i] = (g / (1.0f + (float) Math.exp(-g))) * up[upOff + i];
    }
  }

  @Override
  public void geluTanh(
    final float[] src, final int srcOff, final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = geluPytorchTanh(src[srcOff + i]);
    }
  }

  @Override
  public void geluTanhMul(
    final float[] gate, final int gateOff, final float[] up, final int upOff,
    final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = geluPytorchTanh(gate[gateOff + i]) * up[upOff + i];
    }
  }

  @Override
  public void tanhSoftcap(
    final float[] src, final int srcOff, final float cap,
    final float[] dst, final int dstOff, final int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = (float) Math.tanh(src[srcOff + i] / cap) * cap;
    }
  }
}
