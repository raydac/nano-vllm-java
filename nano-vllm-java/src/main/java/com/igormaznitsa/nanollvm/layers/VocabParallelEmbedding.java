package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.EmbeddingKernel;
import com.igormaznitsa.nanollvm.tensor.LinearKernel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Immutable token embedding table. A layout-specialized {@link EmbeddingKernel} is bound at
 * construction (dense or GGML-typed packed).
 */
public class VocabParallelEmbedding {

  protected final PackedWeight packedWeight;
  protected final Tensor weight;
  private final EmbeddingKernel embeddingKernel;

  public VocabParallelEmbedding(final Tensor weight) {
    this.weight = requireNonNull(weight, "weight");
    this.packedWeight = null;
    this.embeddingKernel = EmbeddingKernel.of(weight);
  }

  public VocabParallelEmbedding(final PackedWeight weight) {
    this(denseOrPacked(requireNonNull(weight, "weight")));
  }

  private VocabParallelEmbedding(final VocabParallelEmbedding assembled) {
    this.weight = assembled.weight;
    this.packedWeight = assembled.packedWeight;
    this.embeddingKernel = assembled.embeddingKernel;
  }

  private VocabParallelEmbedding(final EmbeddingKernel kernel, final PackedWeight packed) {
    this.weight = null;
    this.packedWeight = packed;
    this.embeddingKernel = requireNonNull(kernel, "kernel");
  }

  private static VocabParallelEmbedding denseOrPacked(final PackedWeight weight) {
    if (weight.isFloat32()) {
      Tensor dense = weight.materialize();
      weight.releasePackedBytes();
      return new VocabParallelEmbedding(dense);
    }
    return new VocabParallelEmbedding(EmbeddingKernel.of(weight), weight);
  }

  public VocabParallelEmbedding(final EmbeddingKernel kernel) {
    this.weight = null;
    this.packedWeight = null;
    this.embeddingKernel = requireNonNull(kernel, "kernel");
  }

  public EmbeddingKernel embeddingKernel() {
    return this.embeddingKernel;
  }

  public Tensor weight() {
    if (this.weight != null) {
      return this.weight;
    }
    if (this.packedWeight != null) {
      return this.packedWeight.materialize();
    }
    throw new IllegalStateException("embedding has no dense weight table");
  }

  public PackedWeight packedWeight() {
    return this.packedWeight;
  }

  public boolean isPacked() {
    return this.packedWeight != null;
  }

  public Tensor forward(final Tensor inputIds, final Context context) {
    int n = inputIds.numel();
    Tensor out = Tensor.zeros(n, this.embeddingKernel.embeddingDim());
    this.embeddingKernel.gather(inputIds.data(), inputIds.offset(), n, out.data(), 0);
    return out;
  }

  public int numEmbeddings() {
    return this.embeddingKernel.vocabSize();
  }

  public int embeddingDim() {
    return this.embeddingKernel.embeddingDim();
  }

  public static final class ParallelLMHead extends VocabParallelEmbedding {

    private final LinearKernel linearKernel;

    public ParallelLMHead(final Tensor weight) {
      super(weight);
      this.linearKernel = LinearKernel.of(weight);
    }

    public ParallelLMHead(final PackedWeight weight) {
      super(requireNonNull(weight, "weight"));
      this.linearKernel = this.weight != null
        ? LinearKernel.of(this.weight)
        : LinearKernel.of(weight);
    }

    public ParallelLMHead(final EmbeddingKernel embedding, final LinearKernel linear) {
      super(embedding);
      this.linearKernel = requireNonNull(linear, "linear");
    }

    public LinearKernel linearKernel() {
      return this.linearKernel;
    }

    @Override
    public Tensor forward(final Tensor x, final Context ctx) {
      requireNonNull(ctx, "ctx");
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
      MatmulRuntime matmul = ctx.matmul() != null ? ctx.matmul() : MatmulRuntime.sequential();
      int in = this.linearKernel.inFeatures();
      int out = this.linearKernel.outFeatures();
      int rows = hidden.numel() / in;
      if (hidden.numel() % in != 0) {
        throw new IllegalArgumentException("hidden last dim mismatch");
      }
      Tensor y = Tensor.zeros(rows, out);
      this.linearKernel.apply(hidden.data(), hidden.offset(), null, y.data(), 0, rows, matmul);
      return y;
    }
  }
}
