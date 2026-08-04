package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Immutable token embedding table. Weight matrix is fixed at construction.
 */
public class VocabParallelEmbedding {

  private final int numEmbeddings;
  private final int embeddingDim;
  protected final Tensor weight;

  public VocabParallelEmbedding(Tensor weight) {
    this.weight = requireNonNull(weight, "weight");
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException("embedding weight must be rank 2, got " + weight.ndim());
    }
    this.numEmbeddings = weight.size(0);
    this.embeddingDim = weight.size(1);
  }

  public Tensor weight() {
    return this.weight;
  }

  public Tensor forward(Tensor inputIds) {
    return Ops.embedding(inputIds, this.weight);
  }

  public int numEmbeddings() {
    return this.numEmbeddings;
  }

  public int embeddingDim() {
    return this.embeddingDim;
  }

  public static final class ParallelLMHead extends VocabParallelEmbedding {

    public ParallelLMHead(Tensor weight) {
      super(weight);
    }

    @Override
    public Tensor forward(Tensor x) {
      Context ctx = Context.get();
      Tensor hidden = x;
      if (ctx.isPrefill()) {
        int[] cu = ctx.cuSeqlensQ();
        int batch = cu.length - 1;
        int dim = x.size(x.ndim() - 1);
        Tensor gathered = Tensor.zeros(batch, dim);
        for (int i = 0; i < batch; i++) {
          int idx = cu[i + 1] - 1;
          System.arraycopy(x.data(), x.offset() + idx * dim, gathered.data(), i * dim, dim);
        }
        hidden = gathered;
      }
      return Ops.linear(hidden, this.weight, null);
    }
  }
}
