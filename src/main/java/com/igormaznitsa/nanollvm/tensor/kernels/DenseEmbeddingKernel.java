package com.igormaznitsa.nanollvm.tensor.kernels;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.EmbeddingKernel;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Dense float32 embedding gather.
 */
public final class DenseEmbeddingKernel implements EmbeddingKernel {

  private final float[] weight;
  private final int weightOffset;
  private final int vocabSize;
  private final int embeddingDim;

  private DenseEmbeddingKernel(
    final float[] weight,
    final int weightOffset,
    final int vocabSize,
    final int embeddingDim
  ) {
    this.weight = weight;
    this.weightOffset = weightOffset;
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
  }

  public static DenseEmbeddingKernel of(final Tensor weight) {
    requireNonNull(weight, "weight");
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException("embedding weight must be rank 2");
    }
    return new DenseEmbeddingKernel(
      weight.data(), weight.offset(), weight.size(0), weight.size(1));
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
    return "dense-f32-embed";
  }

  @Override
  public void gather(
    final float[] ids, final int idsOff, final int count,
    final float[] out, final int outOff
  ) {
    for (int i = 0; i < count; i++) {
      int id = Math.round(ids[idsOff + i]);
      if (id < 0 || id >= this.vocabSize) {
        throw new IndexOutOfBoundsException("token id " + id);
      }
      System.arraycopy(
        this.weight, this.weightOffset + id * this.embeddingDim,
        out, outOff + i * this.embeddingDim,
        this.embeddingDim);
    }
  }
}
