package com.igormaznitsa.nanollvm.models.internal.fasttext;

final class VectorOps {

  private VectorOps() {
  }

  static void zero(final float[] data) {
    for (int i = 0; i < data.length; i++) {
      data[i] = 0f;
    }
  }

  static void scale(final float[] data, final float factor) {
    for (int i = 0; i < data.length; i++) {
      data[i] *= factor;
    }
  }
}
