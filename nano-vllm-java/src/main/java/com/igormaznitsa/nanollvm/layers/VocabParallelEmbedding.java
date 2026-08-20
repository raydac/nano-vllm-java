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
 * construction (dense or GGML-typed packed). The vLLM name is kept; this port is single-device
 * (no vocab-parallel split).
 *
 * <p>Table layout is {@code [vocab, dim]}. {@link #forward(Tensor, Context)} gathers one row per
 * token id. Tied output heads use {@link ParallelLMHead}, which reuses the same table as a linear
 * map from hidden states to logits.
 */
public class VocabParallelEmbedding {

  protected final PackedWeight packedWeight;
  protected final Tensor weight;
  private final EmbeddingKernel embeddingKernel;

  /**
   * Dense float32 table {@code [vocab, dim]}.
   *
   * @param weight embedding matrix
   */
  public VocabParallelEmbedding(final Tensor weight) {
    this.weight = requireNonNull(weight, "weight");
    this.packedWeight = null;
    this.embeddingKernel = EmbeddingKernel.of(weight);
  }

  /**
   * Packed GGUF table. Float32 packs are materialized and the packed bytes released; other GGML
   * types stay packed and dequant on each gather.
   *
   * @param weight packed matrix {@code [vocab, dim]}
   */
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

  /**
   * Already-bound gather kernel (tests and custom graphs). No dense/packed table is stored.
   *
   * @param kernel embedding gather
   */
  public VocabParallelEmbedding(final EmbeddingKernel kernel) {
    this.weight = null;
    this.packedWeight = null;
    this.embeddingKernel = requireNonNull(kernel, "kernel");
  }

  /**
   * Gather kernel bound at construction.
   *
   * @return embedding kernel
   */
  public EmbeddingKernel embeddingKernel() {
    return this.embeddingKernel;
  }

  /**
   * Dense float32 table. Packed embeddings materialize a copy (expensive); kernel-only instances
   * have none.
   *
   * @return weight {@code [vocab, dim]}
   * @throws IllegalStateException if this embedding has no dense or packed table
   */
  public Tensor weight() {
    if (this.weight != null) {
      return this.weight;
    }
    if (this.packedWeight != null) {
      return this.packedWeight.materialize();
    }
    throw new IllegalStateException("embedding has no dense weight table");
  }

  /**
   * Packed GGUF payload, or {@code null} for dense / kernel-only embeddings.
   *
   * @return packed weight, or {@code null}
   */
  public PackedWeight packedWeight() {
    return this.packedWeight;
  }

  /**
   * {@code true} when this table still holds a GGUF packed payload.
   *
   * @return whether {@link #packedWeight()} is non-null
   */
  public boolean isPacked() {
    return this.packedWeight != null;
  }

  /**
   * Gathers embedding rows for each token id in {@code inputIds}. {@code context} is unused here
   * (kept so call sites match {@link Linear#forward(Tensor, Context)}); {@link ParallelLMHead} uses it.
   *
   * @param inputIds rank-1 token ids stored as floats
   * @param context  step context (ignored)
   * @return {@code [n, embeddingDim]}
   */
  public Tensor forward(final Tensor inputIds, final Context context) {
    int n = inputIds.numel();
    Tensor out = Tensor.zeros(n, this.embeddingKernel.embeddingDim());
    this.embeddingKernel.gather(inputIds.data(), inputIds.offset(), n, out.data(), 0);
    return out;
  }

  /**
   * Vocabulary size (rows of the table).
   *
   * @return token count
   */
  public int numEmbeddings() {
    return this.embeddingKernel.vocabSize();
  }

  /**
   * Hidden width (columns of the table).
   *
   * @return embedding dimension
   */
  public int embeddingDim() {
    return this.embeddingKernel.embeddingDim();
  }

  /**
   * Output projection that reuses an embedding table as {@code logits = hidden @ Wᵀ} (tied LM
   * head). On prefill, only the last token of each sequence is projected (via
   * {@link Context#cuSeqlensQ()}); decode projects every row.
   */
  public static final class ParallelLMHead extends VocabParallelEmbedding {

    private final LinearKernel linearKernel;

    /**
     * Dense tied head from a float32 embedding table.
     *
     * @param weight matrix {@code [vocab, dim]}
     */
    public ParallelLMHead(final Tensor weight) {
      super(weight);
      this.linearKernel = LinearKernel.of(weight);
    }

    /**
     * Packed tied head from a GGUF embedding table.
     *
     * @param weight packed matrix {@code [vocab, dim]}
     */
    public ParallelLMHead(final PackedWeight weight) {
      super(requireNonNull(weight, "weight"));
      this.linearKernel = this.weight != null
        ? LinearKernel.of(this.weight)
        : LinearKernel.of(weight);
    }

    /**
     * Already-bound gather and linear kernels (must describe the same table).
     *
     * @param embedding gather kernel
     * @param linear    {@code hidden → vocab} kernel
     */
    public ParallelLMHead(final EmbeddingKernel embedding, final LinearKernel linear) {
      super(embedding);
      this.linearKernel = requireNonNull(linear, "linear");
    }

    /**
     * Linear kernel used for the hidden→vocab map (same weights as the embedding gather).
     *
     * @return affine kernel
     */
    public LinearKernel linearKernel() {
      return this.linearKernel;
    }

    /**
     * Projects hidden states to vocabulary logits. Prefill gathers the last token of each
     * sequence first so the matmul is {@code [batch, dim] → [batch, vocab]}.
     *
     * @param x   last-layer hidden states
     * @param ctx step context (cu-seqlens on prefill, matmul runtime)
     * @return logits {@code [rows, vocab]}
     * @throws NullPointerException     if {@code ctx} is {@code null}
     * @throws IllegalArgumentException if {@code x} width is not a multiple of {@code in}
     */
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
