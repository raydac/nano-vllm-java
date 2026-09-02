package com.igormaznitsa.nanollvm.models.internal.fasttext;

import java.io.IOException;

final class QuantMatrix implements Matrix {

  private final boolean qnorm;
  private final long rows;
  private final long cols;
  private final byte[] codes;
  private final ProductQuantizer pq;
  private final byte[] normCodes;
  private final ProductQuantizer npq;

  private QuantMatrix(
    final boolean qnorm,
    final long rows,
    final long cols,
    final byte[] codes,
    final ProductQuantizer pq,
    final byte[] normCodes,
    final ProductQuantizer npq
  ) {
    this.qnorm = qnorm;
    this.rows = rows;
    this.cols = cols;
    this.codes = codes;
    this.pq = pq;
    this.normCodes = normCodes;
    this.npq = npq;
  }

  static QuantMatrix load(final LittleEndianInput in) throws IOException {
    final boolean qnorm = in.readBoolean();
    final long rows = in.readLong();
    final long cols = in.readLong();
    final int codesize = in.readInt();
    final byte[] codes = new byte[codesize];
    in.readFully(codes);
    final ProductQuantizer pq = ProductQuantizer.load(in);
    byte[] normCodes = null;
    ProductQuantizer npq = null;
    if (qnorm) {
      if (rows > Integer.MAX_VALUE) {
        throw new IOException("QuantMatrix rows too large: " + rows);
      }
      normCodes = new byte[(int) rows];
      in.readFully(normCodes);
      npq = ProductQuantizer.load(in);
    }
    return new QuantMatrix(qnorm, rows, cols, codes, pq, normCodes, npq);
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
    return this.pq.mulcode(vector, this.codes, row, this.rowNorm(row));
  }

  @Override
  public void addRowToVector(final float[] vector, final int row) {
    this.pq.addcode(vector, this.codes, row, this.rowNorm(row));
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

  private float rowNorm(final int row) {
    return this.qnorm ? this.npq.normCentroid(this.normCodes[row] & 0xFF) : 1f;
  }
}
