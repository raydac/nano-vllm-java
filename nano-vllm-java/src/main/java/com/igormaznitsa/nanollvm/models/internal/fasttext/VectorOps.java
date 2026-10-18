package com.igormaznitsa.nanollvm.models.internal.fasttext;

import java.util.Arrays;

final class VectorOps {

  private VectorOps() {
  }

  static void zero(final float[] data) {
    Arrays.fill(data, 0f);
  }

  static void scale(final float[] data, final float factor) {
    for (int i = 0; i < data.length; i++) {
      data[i] *= factor;
    }
  }
}
