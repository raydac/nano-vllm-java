package com.igormaznitsa.nanollvm.tensor;

/**
 * Facade over {@link FloatKernels} plus the tiled dense linear (GEMM-like) routine used by
 * {@link Ops#linear}.
 *
 * <h2>What this class is</h2>
 * Layer code and {@link Ops} should call {@code VectorMath} rather than {@link FloatKernels}
 * directly for the shared primitives ({@link #dot}, {@link #sumSquares}, {@link #scaleAdd}) and
 * for batched {@link #linear}. The active backend (scalar vs Vector API) is fixed at class init via
 * {@link FloatKernels#get()}.
 *
 * <p>All methods work on <strong>raw {@code float[]} slices</strong> ({@code offset} + length),
 * matching {@link Tensor}’s view model. Bounds are the caller’s responsibility (hot path).
 *
 * <h2>Hard part — tiled {@link #linear}</h2>
 * A naïve triple loop over {@code rows × out × in} thrashes caches. This implementation tiles:
 * <ul>
 *   <li>{@link #TILE_N} — block of output channels (N)</li>
 *   <li>{@link #TILE_K} — block of input features (K)</li>
 * </ul>
 * For each row, each output tile is seeded with bias (or zero), then for each K-tile every output
 * in the N-tile accumulates a {@link FloatKernels#dot} between the corresponding {@code x} slice
 * and the matching segment of that output’s weight row ({@code w} layout {@code [out, in]},
 * row-major). That keeps a working set of weights and activations in cache longer and reuses the
 * SIMD/scalar {@code dot} kernel.
 *
 * @see FloatKernels
 * @see Ops#linear(Tensor, Tensor, Tensor)
 */
public final class VectorMath {

  /**
   * Output-channel tile width for {@link #linear}: process at most this many {@code out} indices
   * before advancing the K (input) tiles.
   */
  private static final int TILE_N = 64;

  /**
   * Input-feature tile width for {@link #linear}: length of each partial dot along {@code in}.
   */
  private static final int TILE_K = 256;

  private static final FloatKernels KERNELS = FloatKernels.get();

  private VectorMath() {
  }

  /**
   * Short description of the active float backend and tiling knobs (for load logs / diagnostics).
   *
   * @return e.g. {@code "scalar, tileN=64 tileK=256"} or a Vector API species string with the same tiles
   */
  public static String backendInfo() {
    return "%s, tileN=%d tileK=%d".formatted(KERNELS.name(), TILE_N, TILE_K);
  }

  /**
   * Dot product of two equal-length slices; delegates to {@link FloatKernels#dot}.
   *
   * @param a       left buffer
   * @param aOffset start of left slice
   * @param b       right buffer
   * @param bOffset start of right slice
   * @param n       number of paired elements
   * @return {@code Σ a[i]*b[i]} over the slices
   * @see FloatKernels#dot(float[], int, float[], int, int)
   */
  public static float dot(float[] a, int aOffset, float[] b, int bOffset, int n) {
    return KERNELS.dot(a, aOffset, b, bOffset, n);
  }

  /**
   * Sum of squares over one slice; delegates to {@link FloatKernels#sumSquares}.
   *
   * @param a      buffer
   * @param offset start of the slice
   * @param n      number of elements
   * @return {@code Σ v*v}
   * @see FloatKernels#sumSquares(float[], int, int)
   */
  public static float sumSquares(float[] a, int offset, int n) {
    return KERNELS.sumSquares(a, offset, n);
  }

  /**
   * Elementwise {@code dst = src * scale * weight} over slices; delegates to
   * {@link FloatKernels#scaleAdd}.
   *
   * @see FloatKernels#scaleAdd(float[], int, float[], int, float, float[], int, int)
   */
  public static void scaleAdd(
      float[] src, int srcOff, float[] weight, int wOff, float scale, float[] dst, int dstOff, int n
  ) {
    KERNELS.scaleAdd(src, srcOff, weight, wOff, scale, dst, dstOff, n);
  }

  /**
   * Dense batched linear map: for each of {@code rows} inputs of width {@code in}, write
   * {@code out} outputs {@code y = x Wᵀ (+ bias)}.
   *
   * <p><strong>Buffer layouts (row-major):</strong>
   * <ul>
   *   <li>{@code x} — {@code rows} contiguous vectors of length {@code in}, starting at
   *       {@code xOffset}</li>
   *   <li>{@code w} — {@code out} weight rows of length {@code in} (shape {@code [out, in]}),
   *       starting at {@code wOffset}; row {@code o} begins at {@code wOffset + o * in}</li>
   *   <li>{@code y} — {@code rows} contiguous vectors of length {@code out}, starting at
   *       {@code yOffset}</li>
   *   <li>{@code bias} — length {@code out} indexed from {@code 0}, or {@code null}</li>
   * </ul>
   *
   * <p><strong>Hard part — tiling order:</strong>
   * <pre>
   *   for each row r
   *     for each output tile [tile0, tile1) of width ≤ TILE_N
   *       seed y[r, o] ← bias[o] or 0
   *       for each input tile [k0, k1) of width ≤ TILE_K
   *         for each o in the output tile
   *           y[r, o] += dot(x[r, k0:k1], w[o, k0:k1])
   * </pre>
   * Partial dots accumulate into {@code y}; the same {@code x} K-tile is reused across the N-tile’s
   * output channels before moving on.
   *
   * @param x       input activations
   * @param xOffset start of the first input row
   * @param w       weight matrix storage {@code [out, in]}
   * @param wOffset start of weight row 0
   * @param bias    optional bias of length {@code out} (index 0-based), or {@code null}
   * @param y       destination activations
   * @param yOffset start of the first output row
   * @param rows    batch / leading size
   * @param in      input feature width
   * @param out     output feature width
   */
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
