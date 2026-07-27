package io.nanovllm.layers;

import io.nanovllm.tensor.Ops;
import io.nanovllm.tensor.Tensor;
import io.nanovllm.utils.Context;

public class VocabParallelEmbedding {

  private final int numEmbeddings;
  private final int embeddingDim;
  protected Tensor weight;

  public VocabParallelEmbedding(int numEmbeddings, int embeddingDim) {
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.weight = Tensor.zeros(numEmbeddings, embeddingDim);
  }

  public Tensor weight() {
    return this.weight;
  }

  public void setWeight(Tensor weight) {
    this.weight = weight;
  }

  public void loadWeight(Tensor loaded) {
    this.weight.copyFrom(loaded);
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

    public ParallelLMHead(int numEmbeddings, int embeddingDim) {
      super(numEmbeddings, embeddingDim);
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
