package com.igormaznitsa.nanollvm.tensor;

public final class VectorMath {

  private static final int TILE_N = 64;
  private static final int TILE_K = 256;

  private static final FloatKernels KERNELS = FloatKernels.get();

  private VectorMath() {
  }

  public static String backendInfo() {
    return "%s, tileN=%d tileK=%d".formatted(KERNELS.name(), TILE_N, TILE_K);
  }

  public static float dot(float[] a, int aOffset, float[] b, int bOffset, int n) {
    return KERNELS.dot(a, aOffset, b, bOffset, n);
  }

  public static float sumSquares(float[] a, int offset, int n) {
    return KERNELS.sumSquares(a, offset, n);
  }

  public static void scaleAdd(
      float[] src, int srcOff, float[] weight, int wOff, float scale, float[] dst, int dstOff, int n
  ) {
    KERNELS.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  public static void linear(
      float[] x, int xOffset,
      float[] w, int wOffset,
      float[] bias,
      float[] y, int yOffset,
      int rows, int in, int out
  ) {
    for (int r = 0; r < rows; r++) {
      int xBase = xOffset + r * in;
      int yBase = yOffset + r * out;
      for (int tile0 = 0; tile0 < out; tile0 += TILE_N) {
        int tile1 = Math.min(out, tile0 + TILE_N);
        for (int o = tile0; o < tile1; o++) {
          y[yBase + o] = bias != null ? bias[o] : 0f;
        }
        for (int k0 = 0; k0 < in; k0 += TILE_K) {
          int k1 = Math.min(in, k0 + TILE_K);
          int kLen = k1 - k0;
          for (int o = tile0; o < tile1; o++) {
            y[yBase + o] += KERNELS.dot(x, xBase + k0, w, wOffset + o * in + k0, kLen);
          }
        }
      }
    }
  }
}
