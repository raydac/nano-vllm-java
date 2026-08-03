package com.igormaznitsa.nanollvm.tensor;

import static java.util.Objects.requireNonNull;

import java.util.Arrays;

/**
 * Multidimensional array of {@code float} values used for weights and activations in this engine.
 *
 * <h2>What a tensor is here</h2>
 * In deep-learning practice a <em>tensor</em> is a numeric array with a {@linkplain #shape() shape}
 * (ordered list of axis lengths). Rank {@code 1} is a vector, rank {@code 2} a matrix, higher ranks
 * hold batched activations, KV pages, and similar layouts. The product of the shape dimensions is
 * {@linkplain #numel() numel} — the number of scalar elements.
 *
 * <p>This educational CPU port stores every element as IEEE {@code float32} in a contiguous
 * <strong>row-major</strong> flat buffer: the last axis changes fastest. For a matrix of shape
 * {@code [V, H]}, element {@code (t, j)} lives at flat index {@code t * H + j} (relative to
 * {@link #offset()}).
 *
 * <h2>Storage model (the hard part)</h2>
 * A {@code Tensor} is a <em>view</em> over a {@code float[]} slice:
 * <ul>
 *   <li>{@code data} — backing array (may be shared by several tensors after {@link #reshape})</li>
 *   <li>{@code offset} — start index of this view inside {@code data}</li>
 *   <li>{@code size} / {@link #numel()} — number of logical elements in the view</li>
 *   <li>{@code shape} — how those {@code size} elements are interpreted as axes</li>
 * </ul>
 * {@link #get(int)} / {@link #set(int, float)} index <em>within the view</em> ({@code 0 .. numel-1}),
 * not absolute positions in {@code data}. Absolute index is {@code offset + index}.
 *
 * <p>{@link #reshape(int...)} changes only the shape metadata when {@code numel} is preserved; it
 * returns a new {@code Tensor} that <strong>shares</strong> the same backing array and offset.
 * Mutations through either view are visible to the other.
 *
 * <p>{@link #data()} returns the raw backing array (not a copy). Callers that need an owned dense
 * copy of the logical contents should use {@link #toFloatArray()}.
 *
 * <h2>Role in the engine</h2>
 * Both <em>weights</em> (loaded once from {@code .safetensors}) and <em>activations</em> (ephemeral
 * forward-pass results) are {@code Tensor} instances. Kernels live in {@link Ops}; this class only
 * owns layout and storage access.
 *
 * @see Ops
 * @see #reshape(int...)
 * @see #offset()
 */
public final class Tensor {

  private final float[] data;
  private final int[] shape;
  private final int offset;
  private final int size;

  private Tensor(float[] data, int[] shape, int offset, int size) {
    this.data = data;
    this.shape = shape;
    this.offset = offset;
    this.size = size;
  }

  /**
   * Allocates a new tensor filled with zeros.
   *
   * @param shape axis lengths; must be non-empty and non-negative; {@code numel} is their product
   * @return a tensor owning a fresh {@code float[numel]} buffer at offset {@code 0}
   * @throws IllegalArgumentException if {@code shape} is empty or contains a negative dimension
   * @throws ArithmeticException      if the product of dimensions overflows {@code int}
   */
  public static Tensor zeros(int... shape) {
    int[] s = requireShape(shape);
    return new Tensor(new float[numel(s)], s, 0, numel(s));
  }

  /**
   * Allocates a new tensor filled with {@code 1.0f}.
   *
   * @param shape axis lengths (same constraints as {@link #zeros(int...)})
   * @return a tensor owning a fresh buffer of ones
   */
  public static Tensor ones(int... shape) {
    Tensor t = zeros(shape);
    Arrays.fill(t.data, 1.0f);
    return t;
  }

  /**
   * Wraps an existing dense {@code float[]} as a tensor of the given shape.
   *
   * <p><strong>Ownership:</strong> the array is stored as-is (not copied). Later mutations of
   * {@code data} or of this tensor’s {@link #set(int, float)} affect the same memory. The array
   * length must equal {@code numel(shape)}; offset is {@code 0}.
   *
   * @param data  row-major element storage; length must equal product of {@code shape}
   * @param shape axis lengths for interpreting {@code data}
   * @return a tensor view over {@code data}
   * @throws NullPointerException     if {@code data} or {@code shape} is {@code null}
   * @throws IllegalArgumentException if lengths disagree or {@code shape} is invalid
   */
  public static Tensor of(float[] data, int... shape) {
    int[] s = requireShape(shape);
    int n = numel(s);
    requireNonNull(data, "data");
    if (data.length != n) {
      throw new IllegalArgumentException(
          "data length %d != shape numel %d".formatted(data.length, n));
    }
    return new Tensor(data, s, 0, n);
  }

  /**
   * Validates and defensively copies a shape array.
   *
   * @throws IllegalArgumentException if empty or any dimension is negative
   */
  private static int[] requireShape(int... shape) {
    requireNonNull(shape, "shape");
    if (shape.length == 0) {
      throw new IllegalArgumentException("shape must not be empty");
    }
    for (int d : shape) {
      if (d < 0) {
        throw new IllegalArgumentException("negative dim: " + Arrays.toString(shape));
      }
    }
    return shape.clone();
  }

  /**
   * Product of all dimensions; uses {@link Math#multiplyExact(int, int)} to fail on overflow.
   */
  static int numel(int[] shape) {
    int n = 1;
    for (int d : shape) {
      n = Math.multiplyExact(n, d);
    }
    return n;
  }

  /**
   * Defensive copy of the axis lengths (rank = {@code shape.length}).
   *
   * @return a new {@code int[]} the caller may mutate freely
   */
  public int[] shape() {
    return this.shape.clone();
  }

  /**
   * Package-local shape reference — <strong>must not be mutated</strong> by callers.
   * Used by kernels ({@link Ops}) to avoid cloning on every hot call.
   */
  int[] rawShape() {
    return this.shape;
  }

  /**
   * Rank (number of axes), i.e. {@code shape().length}.
   */
  public int ndim() {
    return this.shape.length;
  }

  /**
   * Length of one axis.
   *
   * @param dim axis index in {@code [0, ndim())}
   * @return {@code shape[dim]}
   * @throws ArrayIndexOutOfBoundsException if {@code dim} is out of range
   */
  public int size(int dim) {
    return this.shape[dim];
  }

  /**
   * Number of logical scalar elements in this view ({@code size} field).
   * Equals the product of {@link #shape()} for a dense view starting at {@link #offset()}.
   */
  public int numel() {
    return this.size;
  }

  /**
   * Backing storage array.
   *
   * <p><strong>Not a copy.</strong> The logical contents of this tensor occupy
   * {@code data[offset .. offset + numel)}. Other tensors may share this array after
   * {@link #reshape(int...)}. Prefer {@link #get(int)} / {@link #set(int, float)} or
   * {@link #toFloatArray()} unless a kernel needs direct buffer access.
   *
   * @return the shared {@code float[]} buffer
   */
  public float[] data() {
    return this.data;
  }

  /**
   * Start index of this view inside {@link #data()}.
   *
   * <p>Logical element {@code i} (for {@code i} in {@code [0, numel)}) is stored at
   * {@code data[offset + i]}. Factories {@link #zeros}, {@link #ones}, and {@link #of} use
   * {@code offset == 0}; {@link #reshape} preserves the current offset.
   */
  public int offset() {
    return this.offset;
  }

  /**
   * Reads one element by <em>logical</em> flat index within this view.
   *
   * @param index position in {@code [0, numel())}; absolute buffer index is {@code offset + index}
   * @return the float at that logical position
   * @throws ArrayIndexOutOfBoundsException if {@code offset + index} is outside {@code data}
   */
  public float get(int index) {
    return this.data[this.offset + index];
  }

  /**
   * Writes one element by <em>logical</em> flat index within this view.
   *
   * @param index position in {@code [0, numel())}
   * @param value new float value
   * @throws ArrayIndexOutOfBoundsException if {@code offset + index} is outside {@code data}
   */
  public void set(int index, float value) {
    this.data[this.offset + index] = value;
  }

  /**
   * Reinterprets the same contiguous elements under a new shape (view, not a copy).
   *
   * <p><strong>Hard invariant:</strong> {@code numel(newShape)} must equal {@link #numel()}.
   * The returned tensor shares {@link #data()} and {@link #offset()} with {@code this}; only the
   * shape metadata changes. There is no data movement and no allocation of a new float buffer.
   *
   * <p>Example: a vector of length {@code 12} may reshape to {@code [3, 4]} or {@code [2, 2, 3]}
   * but not to {@code [5, 3]}.
   *
   * @param newShape target axes; product must match current {@code numel}
   * @return a new {@code Tensor} view over the same storage
   * @throws IllegalArgumentException if {@code newShape} is invalid or {@code numel} differs
   */
  public Tensor reshape(int... newShape) {
    int[] s = requireShape(newShape);
    if (numel(s) != this.size) {
      throw new IllegalArgumentException(
          "cannot reshape %s to %s".formatted(Arrays.toString(this.shape), Arrays.toString(s)));
    }
    return new Tensor(this.data, s, this.offset, this.size);
  }

  /**
   * Dense copy of the logical elements {@code data[offset .. offset + numel)}.
   *
   * <p>Unlike {@link #data()}, the returned array is independent of this tensor’s buffer and is
   * always length {@link #numel()} starting at index {@code 0} (offset stripped).
   *
   * @return a new {@code float[]} owned by the caller
   */
  public float[] toFloatArray() {
    return Arrays.copyOfRange(this.data, this.offset, this.offset + this.size);
  }

  /**
   * Copies all logical elements from {@code other} into this view (same {@link #numel()} required).
   *
   * <p>Uses {@link System#arraycopy} from {@code other}'s {@code [offset, offset+numel)} into
   * this tensor’s corresponding range. Shapes need not match — only element counts must agree
   * (e.g. copying a {@code [2, 3]} matrix into a length-{@code 6} vector view).
   *
   * @param other source tensor view
   * @throws IllegalArgumentException if {@code other.numel() != this.numel()}
   */
  public void copyFrom(Tensor other) {
    if (this.size != other.size) {
      throw new IllegalArgumentException("size mismatch");
    }
    System.arraycopy(other.data, other.offset, this.data, this.offset, this.size);
  }

  /**
   * Short debug label with shape and numel (does not print element values).
   */
  @Override
  public String toString() {
    return "Tensor(shape=%s, size=%d)".formatted(Arrays.toString(this.shape), this.size);
  }
}
