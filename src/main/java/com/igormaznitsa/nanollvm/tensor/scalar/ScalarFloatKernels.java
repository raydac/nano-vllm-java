package com.igormaznitsa.nanollvm.tensor.scalar;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;

public final class ScalarFloatKernels extends FloatKernels {

  @Override
  public String name() {
    return "scalar";
  }

  @Override
  public float dot(float[] a, int aOffset, float[] b, int bOffset, int n) {
    float sum = 0f;
    for (int i = 0; i < n; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  @Override
  public float sumSquares(float[] a, int offset, int n) {
    float sum = 0f;
    for (int i = 0; i < n; i++) {
      float v = a[offset + i];
      sum += v * v;
    }
    return sum;
  }

  @Override
  public void scaleAdd(
      float[] src, int srcOff, float[] weight, int wOff, float scale, float[] dst, int dstOff, int n
  ) {
    for (int i = 0; i < n; i++) {
      dst[dstOff + i] = src[srcOff + i] * scale * weight[wOff + i];
    }
  }
}
