package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_OUTPUT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GGUF_TOKEN_EMBD_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ggufBlk;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding.Tables;
import com.igormaznitsa.nanollvm.layers.ShortConv;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Causal LM for the {@code lfm2} architecture family (short-conv + GQA) from GGUF weight names.
 *
 * <p>Weights may stay packed (default) or be dense float32 after {@link WeightBag#asDense()}.
 */
public record Lfm2ForCausalLM(Lfm2Model model, ParallelLMHead lmHead) implements CausalLM {

  public Lfm2ForCausalLM(final Config.HfConfig config, final WeightBag weights) {
    this(assemble(requireNonNull(config, "config"), requireNonNull(weights, "weights")));
  }

  private Lfm2ForCausalLM(final Lfm2ForCausalLM assembled) {
    this(assembled.model, assembled.lmHead);
  }

  private static Lfm2ForCausalLM assemble(final Config.HfConfig config, final WeightBag weights) {
    Lfm2Model model = new Lfm2Model(config, weights);
    ParallelLMHead lmHead = weights.findPacked(GGUF_OUTPUT)
      .map(ParallelLMHead::new)
      .or(() -> weights.find(GGUF_OUTPUT).map(ParallelLMHead::new))
      .orElseGet(() -> tiedLmHead(model.embedTokens()));
    return new Lfm2ForCausalLM(model, lmHead);
  }

  private static ParallelLMHead tiedLmHead(final VocabParallelEmbedding embed) {
    return embed.isPacked()
      ? new ParallelLMHead(embed.packedWeight())
      : new ParallelLMHead(embed.weight());
  }

  private static Linear.Row linearRow(final WeightBag weights, final String name) {
    return weights.findPacked(name)
      .map(Linear.Row::new)
      .orElseGet(() -> new Linear.Row(weights.require(name)));
  }

  private static VocabParallelEmbedding embedding(final WeightBag weights, final String name) {
    return weights.findPacked(name)
      .map(VocabParallelEmbedding::new)
      .orElseGet(() -> new VocabParallelEmbedding(weights.require(name)));
  }

  private static ShortConv buildShortConv(
    final WeightBag weights,
    final String blk,
    final int layerIndex
  ) {
    return weights.isPacked(blk + "shortconv.in_proj.weight")
      ? new ShortConv(
      weights.requirePacked(blk + "shortconv.in_proj.weight"),
      weights.require(blk + "shortconv.conv.weight"),
      weights.requirePacked(blk + "shortconv.out_proj.weight"),
      layerIndex)
      : new ShortConv(
      weights.require(blk + "shortconv.in_proj.weight"),
      weights.require(blk + "shortconv.conv.weight"),
      weights.require(blk + "shortconv.out_proj.weight"),
      layerIndex);
  }

  @Override
  public String architectureName() {
    return ARCH_LFM2;
  }

  @Override
  public Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
    return this.model.forward(inputIds, positions, context);
  }

  @Override
  public Tensor computeLogits(final Tensor hiddenStates, final Context context) {
    return this.lmHead.forward(hiddenStates, context);
  }

  @Override
  public List<Attention> attentionLayers() {
    return this.model.layers().stream()
      .filter(layer -> layer.attention() != null)
      .map(layer -> layer.attention().attn())
      .toList();
  }

  @Override
  public boolean equals(final Object other) {
    return this == other;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }

  record Lfm2Attention(
    Linear.Row qProj,
    Linear.Row kProj,
    Linear.Row vProj,
    Linear.Row oProj,
    RotaryEmbedding rotaryEmb,
    Attention attn,
    RMSNorm qNorm,
    RMSNorm kNorm,
    int numHeads,
    int numKvHeads,
    int headDim
  ) {
    Lfm2Attention(final Config.HfConfig config, final WeightBag weights, final int layerIndex,
                  final Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private Lfm2Attention(final Lfm2Attention assembled) {
      this(
        assembled.qProj, assembled.kProj, assembled.vProj, assembled.oProj,
        assembled.rotaryEmb, assembled.attn, assembled.qNorm, assembled.kNorm,
        assembled.numHeads, assembled.numKvHeads, assembled.headDim);
    }

    private static Lfm2Attention assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex,
      final Tables ropes
    ) {
      int numHeads = config.numAttentionHeads();
      int numKvHeads = config.numKeyValueHeads();
      int headDim = config.headDim();
      float scaling = (float) Math.pow(headDim, -0.5);
      String blk = ggufBlk(layerIndex);
      return new Lfm2Attention(
        linearRow(weights, blk + "attn_q.weight"),
        linearRow(weights, blk + "attn_k.weight"),
        linearRow(weights, blk + "attn_v.weight"),
        linearRow(weights, blk + "attn_output.weight"),
        ropes.get(headDim, headDim, config.maxPositionEmbeddings(), config.ropeTheta()),
        new Attention(numHeads, headDim, scaling, numKvHeads, layerIndex),
        new RMSNorm(weights.require(blk + "attn_q_norm.weight"), config.rmsNormEps()),
        new RMSNorm(weights.require(blk + "attn_k_norm.weight"), config.rmsNormEps()),
        numHeads,
        numKvHeads,
        headDim);
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor q = this.normHeads(
        this.qProj.forward(hiddenStates, context)
          .reshape(hiddenStates.size(0), this.numHeads, this.headDim),
        this.qNorm);
      Tensor k = this.normHeads(
        this.kProj.forward(hiddenStates, context)
          .reshape(hiddenStates.size(0), this.numKvHeads, this.headDim),
        this.kNorm);
      Tensor v = this.vProj.forward(hiddenStates, context)
        .reshape(hiddenStates.size(0), this.numKvHeads, this.headDim);
      Tensor[] rotated = this.rotaryEmb.forward(positions, q, k);
      Tensor o = this.attn.forward(rotated[0], rotated[1], v, context);
      return this.oProj.forward(o.reshape(o.size(0), this.numHeads * this.headDim), context);
    }

    private Tensor normHeads(final Tensor x, final RMSNorm norm) {
      return norm.forward(x.reshape(x.size(0) * x.size(1), this.headDim))
        .reshape(x.size(0), x.size(1), this.headDim);
    }
  }

  record Lfm2MLP(Linear.Row gateProj, Linear.Row upProj, Linear.Row downProj) {
    Lfm2MLP(final Config.HfConfig config, final WeightBag weights, final int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Lfm2MLP(final Lfm2MLP assembled) {
      this(assembled.gateProj, assembled.upProj, assembled.downProj);
    }

    private static Lfm2MLP assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      if (!"silu".equals(config.effectiveActivation()) && !"silu".equals(config.hiddenAct())) {
        throw new IllegalArgumentException(
          "lfm2 architecture expects silu, got " + config.effectiveActivation());
      }
      String blk = ggufBlk(layerIndex);
      return new Lfm2MLP(
        linearRow(weights, blk + "ffn_gate.weight"),
        linearRow(weights, blk + "ffn_up.weight"),
        linearRow(weights, blk + "ffn_down.weight"));
    }

    Tensor forward(final Tensor x, final Context context) {
      return this.downProj.forward(
        Ops.siluAndMul(this.gateProj.forward(x, context), this.upProj.forward(x, context)),
        context);
    }
  }

  record Lfm2DecoderLayer(
    Lfm2Attention attention,
    ShortConv shortConv,
    Lfm2MLP mlp,
    RMSNorm operatorNorm,
    RMSNorm ffnNorm
  ) {
    Lfm2DecoderLayer(final Config.HfConfig config, final WeightBag weights, final int layerIndex,
                     final Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private Lfm2DecoderLayer(final Lfm2DecoderLayer assembled) {
      this(
        assembled.attention, assembled.shortConv, assembled.mlp,
        assembled.operatorNorm, assembled.ffnNorm);
    }

    private static Lfm2DecoderLayer assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex,
      final Tables ropes
    ) {
      String blk = ggufBlk(layerIndex);
      Lfm2Attention attention = null;
      ShortConv shortConv = null;
      if (config.isConvLayer(layerIndex)) {
        shortConv = buildShortConv(weights, blk, layerIndex);
      } else {
        attention = new Lfm2Attention(config, weights, layerIndex, ropes);
      }
      return new Lfm2DecoderLayer(
        attention,
        shortConv,
        new Lfm2MLP(config, weights, layerIndex),
        new RMSNorm(weights.require(blk + "attn_norm.weight"), config.rmsNormEps()),
        new RMSNorm(weights.require(blk + "ffn_norm.weight"), config.rmsNormEps()));
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor normed = this.operatorNorm.forward(hiddenStates);
      Tensor mixed = this.attention != null
        ? this.attention.forward(positions, normed, context)
        : this.shortConv.forward(normed, context);
      Tensor afterOp = Ops.add(hiddenStates, mixed);
      return Ops.add(afterOp, this.mlp.forward(this.ffnNorm.forward(afterOp), context));
    }
  }

  record Lfm2Model(
    VocabParallelEmbedding embedTokens,
    List<Lfm2DecoderLayer> layers,
    RMSNorm embeddingNorm
  ) {
    Lfm2Model(final Config.HfConfig config, final WeightBag weights) {
      this(assemble(config, weights));
    }

    private Lfm2Model(final Lfm2Model assembled) {
      this(assembled.embedTokens, assembled.layers, assembled.embeddingNorm);
    }

    private static Lfm2Model assemble(final Config.HfConfig config, final WeightBag weights) {
      Tables ropes = new Tables();
      List<Lfm2DecoderLayer> built = IntStream.range(0, config.numHiddenLayers())
        .mapToObj(i -> new Lfm2DecoderLayer(config, weights, i, ropes))
        .toList();
      return new Lfm2Model(
        embedding(weights, GGUF_TOKEN_EMBD),
        List.copyOf(built),
        new RMSNorm(weights.require(GGUF_TOKEN_EMBD_NORM), config.rmsNormEps()));
    }

    Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
      Tensor hidden = this.embedTokens.forward(inputIds, context);
      for (Lfm2DecoderLayer layer : this.layers) {
        hidden = layer.forward(positions, hidden, context);
      }
      return this.embeddingNorm.forward(hidden);
    }
  }
}
