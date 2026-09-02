package com.igormaznitsa.nanollvm.models.internal.fasttext;

import java.io.IOException;

final class ProductQuantizer {

  static final int KSUB = 256;

  private final int nsubq;
  private final int dsub;
  private final int lastdsub;
  private final float[] centroids;

  private ProductQuantizer(
    final int nsubq,
    final int dsub,
    final int lastdsub,
    final float[] centroids
  ) {
    this.nsubq = nsubq;
    this.dsub = dsub;
    this.lastdsub = lastdsub;
    this.centroids = centroids;
  }

  static ProductQuantizer load(final LittleEndianInput in) throws IOException {
    final int dim = in.readInt();
    final int nsubq = in.readInt();
    final int dsub = in.readInt();
    final int lastdsub = in.readInt();
    final float[] centroids = new float[Math.multiplyExact(dim, KSUB)];
    in.readFloats(centroids);
    return new ProductQuantizer(nsubq, dsub, lastdsub, centroids);
  }

  float mulcode(final float[] vector, final byte[] codes, final int row, final float alpha) {
    float result = 0f;
    final int codeOffset = this.nsubq * row;
    int d = this.dsub;
    for (int m = 0; m < this.nsubq; m++) {
      if (m == this.nsubq - 1) {
        d = this.lastdsub;
      }
      final int centroidOffset = this.centroidOffset(m, codes[codeOffset + m] & 0xFF);
      final int vectorOffset = m * this.dsub;
      for (int n = 0; n < d; n++) {
        result += vector[vectorOffset + n] * this.centroids[centroidOffset + n];
      }
    }
    return result * alpha;
  }

  void addcode(final float[] vector, final byte[] codes, final int row, final float alpha) {
    final int codeOffset = this.nsubq * row;
    int d = this.dsub;
    for (int m = 0; m < this.nsubq; m++) {
      if (m == this.nsubq - 1) {
        d = this.lastdsub;
      }
      final int centroidOffset = this.centroidOffset(m, codes[codeOffset + m] & 0xFF);
      final int vectorOffset = m * this.dsub;
      for (int n = 0; n < d; n++) {
        vector[vectorOffset + n] += alpha * this.centroids[centroidOffset + n];
      }
    }
  }

  float normCentroid(final int code) {
    return this.centroids[this.centroidOffset(0, code & 0xFF)];
  }

  private int centroidOffset(final int subspace, final int code) {
    if (subspace == this.nsubq - 1) {
      return subspace * KSUB * this.dsub + code * this.lastdsub;
    }
    return (subspace * KSUB + code) * this.dsub;
  }
}
