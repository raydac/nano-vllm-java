package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_POSITION_EMBD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD_NORM_BIAS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_TYPES;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ggufBlk;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.BidirectionalAttention;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.LayerNorm;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.List;
import java.util.stream.IntStream;

/**
 * BERT-family encoder for GGUF embedding checkpoints: token+position+type embeddings,
 * post-LN transformer blocks, mean pooling, L2 normalize.
 *
 * @since 1.1.0
 */
public final class BertForEmbedding implements EmbeddingEncoder {

  private final Config.HfConfig config;
  private final VocabParallelEmbedding tokenEmbed;
  private final VocabParallelEmbedding positionEmbed;
  private final VocabParallelEmbedding tokenTypeEmbed;
  private final LayerNorm embedNorm;
  private final List<BertBlock> blocks;

  public BertForEmbedding(final Config.HfConfig config, final WeightBag weights) {
    this.config = requireNonNull(config, "config");
    requireNonNull(weights, "weights");
    this.tokenEmbed = embedding(weights, GGUF_TOKEN_EMBD);
    this.positionEmbed = embedding(weights, GGUF_POSITION_EMBD);
    this.tokenTypeEmbed = embedding(weights, GGUF_TOKEN_TYPES);
    this.embedNorm = new LayerNorm(
      weights.require(GGUF_TOKEN_EMBD_NORM),
      weights.require(GGUF_TOKEN_EMBD_NORM_BIAS),
      config.rmsNormEps());
    this.blocks = IntStream.range(0, config.numHiddenLayers())
      .mapToObj(i -> new BertBlock(config, weights, i))
      .toList();
  }

  private static VocabParallelEmbedding embedding(final WeightBag weights, final String name) {
    return weights.findPacked(name)
      .map(VocabParallelEmbedding::new)
      .orElseGet(() -> new VocabParallelEmbedding(weights.require(name)));
  }

  private static Linear.Row linear(final WeightBag weights, final String weight,
                                   final String bias) {
    Tensor biasTensor = weights.require(bias);
    return weights.findPacked(weight)
      .map(packed -> new Linear.Row(packed, biasTensor))
      .orElseGet(() -> new Linear.Row(weights.require(weight), biasTensor));
  }

  private static Tensor idsAsFloat(final int[] tokenIds) {
    float[] data = new float[tokenIds.length];
    for (int i = 0; i < tokenIds.length; i++) {
      data[i] = tokenIds[i];
    }
    return Tensor.of(data, tokenIds.length);
  }

  private static Tensor positionIds(final int length) {
    float[] data = new float[length];
    for (int i = 0; i < length; i++) {
      data[i] = i;
    }
    return Tensor.of(data, length);
  }

  private static float[] meanPool(final Tensor hidden) {
    int seq = hidden.size(0);
    int dim = hidden.size(1);
    float[] out = new float[dim];
    float[] data = hidden.data();
    int off = hidden.offset();
    float inv = 1f / seq;
    for (int i = 0; i < seq; i++) {
      int row = off + i * dim;
      for (int d = 0; d < dim; d++) {
        out[d] += data[row + d];
      }
    }
    for (int d = 0; d < dim; d++) {
      out[d] *= inv;
    }
    return out;
  }

  private static float[] l2Normalize(final float[] vector) {
    double sumSq = 0.0;
    for (float v : vector) {
      sumSq += (double) v * v;
    }
    float inv = sumSq > 0.0 ? (float) (1.0 / Math.sqrt(sumSq)) : 0f;
    for (int i = 0; i < vector.length; i++) {
      vector[i] *= inv;
    }
    return vector;
  }

  @Override
  public String architectureName() {
    return ARCH_BERT;
  }

  @Override
  public int embeddingDim() {
    return this.config.hiddenSize();
  }

  @Override
  public float[] encode(final int[] tokenIds, final MatmulRuntime runtime) {
    requireNonNull(tokenIds, "tokenIds");
    if (tokenIds.length == 0) {
      throw new IllegalArgumentException("tokenIds must not be empty");
    }
    if (tokenIds.length > this.config.maxPositionEmbeddings()) {
      throw new IllegalArgumentException(
        "sequence length %d exceeds max position %d"
          .formatted(tokenIds.length, this.config.maxPositionEmbeddings()));
    }
    MatmulRuntime matmul = runtime == null ? MatmulRuntime.sequential() : runtime;
    Context context = new Context();
    context.bindMatmul(matmul);

    Tensor ids = idsAsFloat(tokenIds);
    Tensor positions = positionIds(tokenIds.length);
    Tensor typeIds = Tensor.zeros(tokenIds.length);

    Tensor hidden = Ops.add(
      Ops.add(
        this.tokenEmbed.forward(ids, context),
        this.positionEmbed.forward(positions, context)),
      this.tokenTypeEmbed.forward(typeIds, context));
    hidden = this.embedNorm.forward(hidden);

    for (BertBlock block : this.blocks) {
      hidden = block.forward(hidden, context);
    }

    return l2Normalize(meanPool(hidden));
  }

  private record BertBlock(
    Linear.Row qProj,
    Linear.Row kProj,
    Linear.Row vProj,
    Linear.Row oProj,
    BidirectionalAttention attn,
    LayerNorm attnNorm,
    Linear.Row ffnUp,
    Linear.Row ffnDown,
    LayerNorm ffnNorm
  ) {
    BertBlock(final Config.HfConfig config, final WeightBag weights, final int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private BertBlock(final BertBlock assembled) {
      this(
        assembled.qProj, assembled.kProj, assembled.vProj, assembled.oProj,
        assembled.attn, assembled.attnNorm, assembled.ffnUp, assembled.ffnDown, assembled.ffnNorm);
    }

    private static BertBlock assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      String blk = ggufBlk(layerIndex);
      float scale = (float) Math.pow(config.headDim(), -0.5);
      return new BertBlock(
        linear(weights, blk + "attn_q.weight", blk + "attn_q.bias"),
        linear(weights, blk + "attn_k.weight", blk + "attn_k.bias"),
        linear(weights, blk + "attn_v.weight", blk + "attn_v.bias"),
        linear(weights, blk + "attn_output.weight", blk + "attn_output.bias"),
        new BidirectionalAttention(config.numAttentionHeads(), config.headDim(), scale),
        new LayerNorm(
          weights.require(blk + "attn_output_norm.weight"),
          weights.require(blk + "attn_output_norm.bias"),
          config.rmsNormEps()),
        linear(weights, blk + "ffn_up.weight", blk + "ffn_up.bias"),
        linear(weights, blk + "ffn_down.weight", blk + "ffn_down.bias"),
        new LayerNorm(
          weights.require(blk + "layer_output_norm.weight"),
          weights.require(blk + "layer_output_norm.bias"),
          config.rmsNormEps()));
    }

    Tensor forward(final Tensor hidden, final Context context) {
      Tensor q = this.qProj.forward(hidden, context);
      Tensor k = this.kProj.forward(hidden, context);
      Tensor v = this.vProj.forward(hidden, context);
      Tensor attnOut = this.oProj.forward(this.attn.forward(q, k, v), context);
      Tensor mid = this.attnNorm.forward(Ops.add(hidden, attnOut));
      Tensor ffn = this.ffnDown.forward(Ops.gelu(this.ffnUp.forward(mid, context)), context);
      return this.ffnNorm.forward(Ops.add(mid, ffn));
    }
  }
}
