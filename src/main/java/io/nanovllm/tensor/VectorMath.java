package io.nanovllm.tensor;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public final class VectorMath {

  private static final int TILE_N = 64;
  private static final int TILE_K = 256;
  private static final long PARALLEL_FLOP_THRESHOLD = 256_000L;
  private static final int PARALLEL_ROW_THRESHOLD = 2;

  private static final ForkJoinPool POOL = createPool();
  private static final FloatKernels KERNELS = FloatKernels.get();

  private VectorMath() {
  }

  private static ForkJoinPool createPool() {
    int cores = Runtime.getRuntime().availableProcessors();
    String prop = System.getProperty("nanovllm.threads");
    int n = prop != null && !prop.isBlank()
        ? Math.max(1, Integer.parseInt(prop.trim()))
        : Math.max(1, cores);
    return new ForkJoinPool(n);
  }

  public static String backendInfo() {
    return "%s, workers=%d, tileN=%d tileK=%d".formatted(
        KERNELS.name(), POOL.getParallelism(), TILE_N, TILE_K);
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
    long flops = (long) rows * out * in;
    boolean parallel = flops >= PARALLEL_FLOP_THRESHOLD
        && (rows >= PARALLEL_ROW_THRESHOLD || (rows == 1 && out >= PARALLEL_ROW_THRESHOLD));
    if (parallel) {
      POOL.submit(
              new LinearTiles(x, xOffset, w, wOffset, bias, y, yOffset, rows, in, out, 0, rows, 0, out))
          .join();
    } else {
      for (int r = 0; r < rows; r++) {
        linearRowBlocked(x, xOffset, w, wOffset, bias, y, yOffset, r, in, out, 0, out);
      }
    }
  }

  private static void linearRowBlocked(
      float[] x, int xOffset,
      float[] w, int wOffset,
      float[] bias,
      float[] y, int yOffset,
      int r, int in, int out, int o0, int o1
  ) {
    int xBase = xOffset + r * in;
    int yBase = yOffset + r * out;
    for (int tile0 = o0; tile0 < o1; tile0 += TILE_N) {
      int tile1 = Math.min(o1, tile0 + TILE_N);
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

  private static final class LinearTiles extends RecursiveAction {
    private static final int OUT_GRAIN = TILE_N;

    private final float[] x;
    private final int xOffset;
    private final float[] w;
    private final int wOffset;
    private final float[] bias;
    private final float[] y;
    private final int yOffset;
    private final int rows;
    private final int in;
    private final int out;
    private final int r0;
    private final int r1;
    private final int o0;
    private final int o1;

    LinearTiles(
        float[] x, int xOffset, float[] w, int wOffset, float[] bias,
        float[] y, int yOffset, int rows, int in, int out,
        int r0, int r1, int o0, int o1
    ) {
      this.x = x;
      this.xOffset = xOffset;
      this.w = w;
      this.wOffset = wOffset;
      this.bias = bias;
      this.y = y;
      this.yOffset = yOffset;
      this.rows = rows;
      this.in = in;
      this.out = out;
      this.r0 = r0;
      this.r1 = r1;
      this.o0 = o0;
      this.o1 = o1;
    }

    @Override
    protected void compute() {
      int nRows = this.r1 - this.r0;
      int nOut = this.o1 - this.o0;
      if (nRows > 1) {
        int mid = this.r0 + (nRows >>> 1);
        invokeAll(
            new LinearTiles(this.x, this.xOffset, this.w, this.wOffset, this.bias,
                this.y, this.yOffset, this.rows, this.in, this.out, this.r0, mid, this.o0, this.o1),
            new LinearTiles(this.x, this.xOffset, this.w, this.wOffset, this.bias,
                this.y, this.yOffset, this.rows, this.in, this.out, mid, this.r1, this.o0, this.o1)
        );
        return;
      }
      if (nOut > OUT_GRAIN) {
        int mid = this.o0 + (nOut >>> 1);
        invokeAll(
            new LinearTiles(this.x, this.xOffset, this.w, this.wOffset, this.bias,
                this.y, this.yOffset, this.rows, this.in, this.out, this.r0, this.r1, this.o0, mid),
            new LinearTiles(this.x, this.xOffset, this.w, this.wOffset, this.bias,
                this.y, this.yOffset, this.rows, this.in, this.out, this.r0, this.r1, mid, this.o1)
        );
        return;
      }
      for (int r = this.r0; r < this.r1; r++) {
        linearRowBlocked(
            this.x, this.xOffset, this.w, this.wOffset, this.bias,
            this.y, this.yOffset, r, this.in, this.out, this.o0, this.o1
        );
      }
    }
  }
}
