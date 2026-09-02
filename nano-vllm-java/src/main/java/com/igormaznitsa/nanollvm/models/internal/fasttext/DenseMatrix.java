package com.igormaznitsa.nanollvm.models.internal.fasttext;

import java.io.IOException;

final class DenseMatrix implements Matrix {

  private final long rows;
  private final long cols;
  private final float[] data;

  private DenseMatrix(final long rows, final long cols, final float[] data) {
    this.rows = rows;
    this.cols = cols;
    this.data = data;
  }

  static DenseMatrix load(final LittleEndianInput in) throws IOException {
    final long rows = in.readLong();
    final long cols = in.readLong();
    final long cells = Math.multiplyExact(rows, cols);
    if (cells > Integer.MAX_VALUE) {
      throw new IOException("DenseMatrix too large: %d x %d".formatted(rows, cols));
    }
    final float[] data = new float[(int) cells];
    in.readFloats(data);
    return new DenseMatrix(rows, cols, data);
  }

  @Override
  public long rows() {
    return this.rows;
  }

  @Override
  public long cols() {
    return this.cols;
  }

  @Override
  public float dotRow(final float[] vector, final int row) {
    final int cols = (int) this.cols;
    final int offset = row * cols;
    float sum = 0f;
    for (int j = 0; j < cols; j++) {
      sum += this.data[offset + j] * vector[j];
    }
    return sum;
  }

  @Override
  public void addRowToVector(final float[] vector, final int row) {
    final int cols = (int) this.cols;
    final int offset = row * cols;
    for (int j = 0; j < cols; j++) {
      vector[j] += this.data[offset + j];
    }
  }

  @Override
  public void averageRowsToVector(final float[] vector, final int[] rows, final int rowCount) {
    VectorOps.zero(vector);
    for (int i = 0; i < rowCount; i++) {
      this.addRowToVector(vector, rows[i]);
    }
    if (rowCount > 0) {
      VectorOps.scale(vector, 1f / rowCount);
    }
  }
}
