package com.igormaznitsa.nanollvm.tensor.kernels;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.EmbeddingKernel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime.PackedRowDequant;

/**
 * Packed embedding gather configured with a row-dequant bound at construction.
 */
public final class PackedEmbeddingKernel implements EmbeddingKernel {

  private final int vocabSize;
  private final int embeddingDim;
  private final String name;
  private final PackedRowDequant dequant;

  private PackedEmbeddingKernel(
    final int vocabSize,
    final int embeddingDim,
    final String name,
    final PackedRowDequant dequant
  ) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.name = name;
    this.dequant = dequant;
  }

  public static PackedEmbeddingKernel of(
    final PackedWeight weight,
    final String name,
    final PackedRowDequant dequant
  ) {
    requireNonNull(weight, "weight");
    requireNonNull(name, "name");
    requireNonNull(dequant, "dequant");
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException("embedding weight must be rank 2");
    }
    return new PackedEmbeddingKernel(weight.size(0), weight.size(1), name, dequant);
  }

  public static PackedEmbeddingKernel of(
    final int vocabSize,
    final int embeddingDim,
    final String name,
    final PackedRowDequant dequant
  ) {
    requireNonNull(name, "name");
    requireNonNull(dequant, "dequant");
    if (vocabSize <= 0 || embeddingDim <= 0) {
      throw new IllegalArgumentException("embedding dimensions must be positive");
    }
    return new PackedEmbeddingKernel(vocabSize, embeddingDim, name, dequant);
  }

  @Override
  public int vocabSize() {
    return this.vocabSize;
  }

  @Override
  public int embeddingDim() {
    return this.embeddingDim;
  }

  @Override
  public String name() {
    return this.name;
  }

  @Override
  public void gather(
    final float[] ids, final int idsOff, final int count,
    final float[] out, final int outOff
  ) {
    float[] row = new float[this.embeddingDim];
    for (int i = 0; i < count; i++) {
      int id = Math.round(ids[idsOff + i]);
      if (id < 0 || id >= this.vocabSize) {
        throw new IndexOutOfBoundsException("token id " + id);
      }
      this.dequant.dequantizeRow(id, row);
      System.arraycopy(row, 0, out, outOff + i * this.embeddingDim, this.embeddingDim);
    }
  }
}
