package io.nanovllm.tensor;

import static java.util.Objects.requireNonNull;

import java.util.Arrays;

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

  public static Tensor zeros(int... shape) {
    int[] s = requireShape(shape);
    return new Tensor(new float[numel(s)], s, 0, numel(s));
  }

  public static Tensor ones(int... shape) {
    Tensor t = zeros(shape);
    Arrays.fill(t.data, 1.0f);
    return t;
  }

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

  static int numel(int[] shape) {
    int n = 1;
    for (int d : shape) {
      n = Math.multiplyExact(n, d);
    }
    return n;
  }

  public int[] shape() {
    return this.shape.clone();
  }

  /**
   * Package-local view of shape — callers must not mutate.
   */
  int[] rawShape() {
    return this.shape;
  }

  public int ndim() {
    return this.shape.length;
  }

  public int size(int dim) {
    return this.shape[dim];
  }

  public int numel() {
    return this.size;
  }

  public float[] data() {
    return this.data;
  }

  public int offset() {
    return this.offset;
  }

  public float get(int index) {
    return this.data[this.offset + index];
  }

  public void set(int index, float value) {
    this.data[this.offset + index] = value;
  }

  public Tensor reshape(int... newShape) {
    int[] s = requireShape(newShape);
    if (numel(s) != this.size) {
      throw new IllegalArgumentException(
          "cannot reshape %s to %s".formatted(Arrays.toString(this.shape), Arrays.toString(s)));
    }
    return new Tensor(this.data, s, this.offset, this.size);
  }

  public float[] toFloatArray() {
    return Arrays.copyOfRange(this.data, this.offset, this.offset + this.size);
  }

  public void copyFrom(Tensor other) {
    if (this.size != other.size) {
      throw new IllegalArgumentException("size mismatch");
    }
    System.arraycopy(other.data, other.offset, this.data, this.offset, this.size);
  }

  @Override
  public String toString() {
    return "Tensor(shape=%s, size=%d)".formatted(Arrays.toString(this.shape), this.size);
  }
}
