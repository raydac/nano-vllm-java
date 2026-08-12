package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.GgufDequant;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * GGUF weight kept in packed GGML blocks. Dequantizes contiguous element ranges (rows) into
 * float scratch during matmul / embedding — does not expand the full tensor to float32.
 *
 * <p>{@link #releasePackedBytes()} drops the quantized payload after a full materialize so unpack
 * does not keep packed and float32 copies at once.
 */
public final class PackedWeight {

  private static final byte[] RELEASED = new byte[0];
  private final int ggmlType;
  private final int[] shape;
  private final int numel;
  private byte[] packed;

  public PackedWeight(
    final byte[] packed,
    final int ggmlType,
    final int[] shape,
    final long numel) {
    this.packed = requireNonNull(packed, "packed");
    this.ggmlType = ggmlType;
    this.shape = Arrays.copyOf(requireNonNull(shape, "shape"), shape.length);
    if (numel <= 0 || numel > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("numel out of range: " + numel);
    }
    this.numel = (int) numel;
    GgufDequant.typeBlockElems(ggmlType);
    long expectedBytes = GgufDequant.packedByteLength(ggmlType, this.numel);
    if (this.packed.length != expectedBytes) {
      throw new IllegalArgumentException(
        "packed length " + this.packed.length + " != expected " + expectedBytes);
    }
    int product = 1;
    for (int dim : this.shape) {
      product = Math.multiplyExact(product, dim);
    }
    if (product != this.numel) {
      throw new IllegalArgumentException("shape product != numel");
    }
  }

  public int ggmlType() {
    return this.ggmlType;
  }

  public int[] shape() {
    return this.shape.clone();
  }

  public int ndim() {
    return this.shape.length;
  }

  public int size(final int dim) {
    return this.shape[dim];
  }

  public int numel() {
    return this.numel;
  }

  public int packedBytes() {
    return this.packed.length;
  }

  public boolean isReleased() {
    return this.packed == RELEASED;
  }

  /**
   * Drops the quantized byte payload. Safe after {@link #materialize()}; further dequant throws.
   */
  public void releasePackedBytes() {
    this.packed = RELEASED;
  }

  public void dequantizeRange(
    final int elemStart,
    final int elemCount,
    final float[] dst,
    final int dstOff) {
    this.requirePackedBytes();
    GgufDequant.dequantizeRange(
      this.packed, this.ggmlType, this.numel, elemStart, elemCount, dst, dstOff);
  }

  /**
   * Dequantizes one row of a rank-2 weight {@code [rows, cols]} into {@code dst[0..cols)}.
   */
  public void dequantizeRow(final int row, final float[] dst) {
    if (this.shape.length != 2) {
      throw new IllegalStateException("dequantizeRow requires rank-2 weight");
    }
    int cols = this.shape[1];
    if (dst.length < cols) {
      throw new IllegalArgumentException("dst too short for row width " + cols);
    }
    this.dequantizeRange(Math.multiplyExact(row, cols), cols, dst, 0);
  }

  /**
   * Backing GGML bytes for specialized kernels. Do not mutate; empty after
   * {@link #releasePackedBytes()}.
   */
  public byte[] rawPacked() {
    this.requirePackedBytes();
    return this.packed;
  }

  public Tensor materialize() {
    this.requirePackedBytes();
    float[] data = new float[this.numel];
    GgufDequant.dequantizeRange(this.packed, this.ggmlType, this.numel, 0, this.numel, data, 0);
    return Tensor.of(data, this.shape.clone());
  }

  ByteBuffer packedBuffer() {
    this.requirePackedBytes();
    return ByteBuffer.wrap(this.packed).order(ByteOrder.LITTLE_ENDIAN);
  }

  private void requirePackedBytes() {
    if (this.packed == RELEASED) {
      throw new IllegalStateException("packed bytes already released");
    }
  }
}
