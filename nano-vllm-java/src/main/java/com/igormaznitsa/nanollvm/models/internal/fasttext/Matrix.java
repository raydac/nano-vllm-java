package com.igormaznitsa.nanollvm.models.internal.fasttext;

interface Matrix {

  long rows();

  long cols();

  float dotRow(final float[] vector, final int row);

  void addRowToVector(final float[] vector, final int row);

  void averageRowsToVector(final float[] vector, final int[] rows, final int rowCount);
}
