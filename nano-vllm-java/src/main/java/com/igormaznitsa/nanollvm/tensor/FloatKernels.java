package com.igormaznitsa.nanollvm.tensor;

/**
 * Pluggable low-level float kernels over contiguous {@code float[]} slices.
 *
 * <h2>What this is</h2>
 * Hot numeric work in this engine (dot products inside linear layers, RMSNorm energy,
 * scaled elementwise products) does not sit on {@link Tensor} directly. Kernels operate on
 * <strong>raw buffers with explicit offsets and lengths</strong>, matching how {@link Tensor}
 * stores a view: logical element {@code i} lives at {@code array[offset + i]}.
 *
 * <p>Two implementations exist:
 * <ul>
 *   <li><strong>Scalar</strong> — plain Java loops ({@code ScalarFloatKernels})</li>
 *   <li><strong>Vector</strong> — JDK incubator Vector API / SIMD ({@code VectorFloatKernels})</li>
 * </ul>
 * Selection is done once by {@link FloatKernelsFactory} (see {@code -Dnanollvm.kernels}).
 * The process-wide default instance is {@link #get()}; {@link VectorMath} delegates to it.
 *
 * <h2>Contract shared by every method</h2>
 * <ul>
 *   <li>Slices are half-open ranges {@code [offset, offset + n)} inside each array.</li>
 *   <li>Callers must ensure bounds; kernels do not re-validate on every call (hot path).</li>
 *   <li>{@code n == 0} is allowed and is a no-op / returns {@code 0} for reductions.</li>
 *   <li>Results are IEEE {@code float32}; Vector and scalar paths are intended to be
 *       numerically close, not bit-identical (FMA / reduction order can differ).</li>
 * </ul>
 *
 * <h2>Hard part — SIMD + scalar tail</h2>
 * The Vector backend processes {@code SPECIES.length()} lanes per iteration, then a
 * <em>scalar tail</em> for the remaining {@code n % laneCount} elements. That split is why
 * offsets matter: loads are {@code fromArray(species, array, offset + i)}, not “start of array”.
 *
 * @see FloatKernelsFactory
 * @see VectorMath
 * @see Tensor
 */
public abstract class FloatKernels {

  private static final FloatKernels INSTANCE = FloatKernelsFactory.create();

  /**
   * For subclasses ({@code ScalarFloatKernels}, {@code VectorFloatKernels}) only.
   * Application code should use {@link #get()} or {@link FloatKernelsFactory}.
   */
  protected FloatKernels() {
  }

  /**
   * Process-wide kernel backend chosen at class initialization via {@link FloatKernelsFactory#create()}.
   *
   * <p>Honors {@code -Dnanollvm.kernels=auto|vector|scalar} (aliases {@code simd}/{@code plain}).
   * Prefer this over constructing kernels repeatedly.
   *
   * @return the singleton scalar or Vector API implementation
   */
  public static FloatKernels get() {
    return INSTANCE;
  }

  /**
   * Human-readable backend label for logs and {@link VectorMath#backendInfo()}.
   *
   * @return e.g. {@code "scalar"} or {@code "Vector API Species[float, 8, S_256_BIT] (len=8)"}
   */
  public abstract String name();

  /**
   * Dot product of two equal-length float slices: {@code Σ<sub>i=0..n-1</sub> a[aOff+i] * b[bOff+i]}.
   *
   * <p>Used heavily by {@link MatmulRuntime#linear} (each output channel accumulates tiled dots against
   * weight rows). The Vector implementation uses lane-wise FMA into an accumulator vector, then
   * {@code reduceLanes(ADD)}, then a scalar remainder loop.
   *
   * @param a       left-hand buffer
   * @param aOffset start index of the left slice
   * @param b       right-hand buffer
   * @param bOffset start index of the right slice
   * @param n       number of paired elements to multiply-add
   * @return the scalar sum of products
   */
  public abstract float dot(final float[] a, final int aOffset, final float[] b, final int bOffset,
                            final int n);

  /**
   * Sum of squares over one slice: {@code Σ<sub>i=0..n-1</sub> a[offset+i]<sup>2</sup>}.
   *
   * <p>Used by RMSNorm-style normalization (energy of a hidden vector before scaling). Equivalent
   * to {@code dot(a, offset, a, offset, n)} but specialized so the Vector path can FMA a lane
   * with itself without a second load.
   *
   * @param a      buffer
   * @param offset start of the slice
   * @param n      number of elements
   * @return {@code Σ v*v} over the slice
   */
  public abstract float sumSquares(final float[] a, final int offset, final int n);

  /**
   * Elementwise {@code dst[dstOff+i] = src[srcOff+i] * scale * weight[wOff+i]} for {@code i in [0,n)}.
   *
   * <p>Despite the name, this is a <em>scaled product</em> into {@code dst}, not
   * {@code dst += …}. Typical use: RMSNorm output {@code x * (1/rms) * weight} written into a
   * destination buffer. The Vector path broadcasts {@code scale} once, then multiplies source
   * and weight lanes; a scalar tail finishes the remainder.
   *
   * @param src    input activations
   * @param srcOff start of the source slice
   * @param weight per-element scale (e.g. RMSNorm gain); same logical length {@code n}
   * @param wOff   start of the weight slice
   * @param scale  shared scalar multiplier applied to every element
   * @param dst    destination buffer (may be distinct from {@code src})
   * @param dstOff start of the destination slice
   * @param n      number of elements to write
   */
  public abstract void scaleAdd(
    final float[] src, final int srcOff, final float[] weight, final int wOff, final float scale,
    final float[] dst, final int dstOff, final int n
  );
}
