package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;

/**
 * Packed Gemma QAT matrix {@code [rows, cols]} with per-row or per-block scales.
 */
public final class GemmaQatWeight {

  private static final byte[] RELEASED = new byte[0];

  private final int rows;
  private final int cols;
  private final int bits;
  private final int packedWidth;
  private final int scaleCols;
  private final float inputActivationScale;
  private final float outputActivationScale;
  private final String name;
  private final float[] scales;
  private byte[] packed;

  public GemmaQatWeight(
    final String name,
    final byte[] packed,
    final float[] scales,
    final int rows,
    final int cols,
    final int bits,
    final int scaleCols,
    final float inputActivationScale,
    final float outputActivationScale
  ) {
    this.name = requireNonNull(name, "name");
    this.packed = requireNonNull(packed, "packed");
    this.scales = Arrays.copyOf(requireNonNull(scales, "scales"), scales.length);
    this.rows = rows;
    this.cols = cols;
    this.bits = bits;
    this.scaleCols = scaleCols;
    this.inputActivationScale = inputActivationScale;
    this.outputActivationScale = outputActivationScale;
    this.packedWidth = GemmaQat.packedWidth(cols, bits);
    if (rows <= 0 || cols <= 0) {
      throw new IllegalArgumentException("QAT weight shape must be positive");
    }
    if (this.packed.length != Math.multiplyExact(rows, this.packedWidth)) {
      throw new IllegalArgumentException(
        "packed bytes %d != rows %d * packedWidth %d for %s".formatted(
          this.packed.length, rows, this.packedWidth, name));
    }
    if (scaleCols <= 0 || cols % scaleCols != 0) {
      throw new IllegalArgumentException(
        "scaleCols %d does not divide cols %d for %s".formatted(scaleCols, cols, name));
    }
    if (this.scales.length != Math.multiplyExact(rows, scaleCols)) {
      throw new IllegalArgumentException(
        "scale count %d != rows %d * scaleCols %d for %s".formatted(
          this.scales.length, rows, scaleCols, name));
    }
  }

  public String name() {
    return this.name;
  }

  public int rows() {
    return this.rows;
  }

  public int cols() {
    return this.cols;
  }

  public int bits() {
    return this.bits;
  }

  public float inputActivationScale() {
    return this.inputActivationScale;
  }

  public float outputActivationScale() {
    return this.outputActivationScale;
  }

  public void dequantizeRow(final int row, final float[] dst) {
    this.requirePacked();
    if (row < 0 || row >= this.rows) {
      throw new IndexOutOfBoundsException("QAT row " + row);
    }
    if (dst.length < this.cols) {
      throw new IllegalArgumentException("dst shorter than unpacked width");
    }
    GemmaQat.unpackRow(
      this.packed,
      row * this.packedWidth,
      this.packedWidth,
      this.bits,
      this.scales,
      row * this.scaleCols,
      this.scaleCols,
      dst,
      this.cols);
  }

  public Tensor materialize() {
    this.requirePacked();
    float[] data = new float[Math.multiplyExact(this.rows, this.cols)];
    float[] row = new float[this.cols];
    for (int r = 0; r < this.rows; r++) {
      this.dequantizeRow(r, row);
      System.arraycopy(row, 0, data, r * this.cols, this.cols);
    }
    return Tensor.of(data, this.rows, this.cols);
  }

  public void releasePackedBytes() {
    this.packed = RELEASED;
  }

  public boolean isReleased() {
    return this.packed == RELEASED;
  }

  private void requirePacked() {
    if (this.packed == RELEASED) {
      throw new IllegalStateException("QAT packed bytes already released: " + this.name);
    }
  }
}
