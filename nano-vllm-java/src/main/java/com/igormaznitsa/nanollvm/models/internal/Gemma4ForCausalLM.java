package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA4;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.PRE_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.selfAttn;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.kernels.PackedEmbeddingKernel;
import com.igormaznitsa.nanollvm.tensor.kernels.PackedLinearKernel;
import java.util.ArrayList;
import java.util.List;

/**
 * Causal LM for the {@code gemma4} / {@code gemma4_text} family (text decoder, QAT weights stay packed).
 */
public record Gemma4ForCausalLM(Gemma4Model model, ParallelLMHead lmHead, float logitSoftcap)
  implements CausalLM {

  public Gemma4ForCausalLM(final Config.HfConfig config, final WeightBag weights) {
    this(assemble(requireNonNull(config, "config"), requireNonNull(weights, "weights")));
  }

  private Gemma4ForCausalLM(final Gemma4ForCausalLM assembled) {
    this(assembled.model, assembled.lmHead, assembled.logitSoftcap);
  }

  private static Gemma4ForCausalLM assemble(final Config.HfConfig config, final WeightBag weights) {
    Gemma4Model model = new Gemma4Model(config, weights);
    GemmaQatWeight lm = weights.requireQat(LM_HEAD);
    return new Gemma4ForCausalLM(
      model,
      new ParallelLMHead(
        PackedEmbeddingKernel.of(lm.rows(), lm.cols(), LM_HEAD, lm::dequantizeRow),
        PackedLinearKernel.of(lm.cols(), lm.rows(), LM_HEAD, lm::dequantizeRow)),
      config.gemma4() == null ? 0f : config.gemma4().finalLogitSoftcapping());
  }

  static Linear qatLinear(final WeightBag weights, final String name) {
    return new Linear(weights.requireQat(name));
  }

  @Override
  public String architectureName() {
    return ARCH_GEMMA4;
  }

  @Override
  public Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
    return this.model.forward(inputIds, positions, context);
  }

  @Override
  public Tensor computeLogits(final Tensor hiddenStates, final Context context) {
    return Ops.tanhSoftcap(this.lmHead.forward(hiddenStates, context), this.logitSoftcap);
  }

  @Override
  public List<Attention> attentionLayers() {
    return this.model.layers().stream()
      .map(layer -> layer.selfAttn().attn())
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

  record Gemma4Attention(
    Linear qProj,
    Linear kProj,
    Linear vProj,
    Linear oProj,
    RotaryEmbedding rotaryEmb,
    Attention attn,
    RMSNorm qNorm,
    RMSNorm kNorm,
    RMSNorm vNorm,
    boolean kvShared,
    int numHeads,
    int numKvHeads,
    int headDim
  ) {
    Gemma4Attention(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Gemma4Attention(final Gemma4Attention assembled) {
      this(
        assembled.qProj, assembled.kProj, assembled.vProj, assembled.oProj,
        assembled.rotaryEmb, assembled.attn, assembled.qNorm, assembled.kNorm, assembled.vNorm,
        assembled.kvShared, assembled.numHeads, assembled.numKvHeads, assembled.headDim);
    }

    private static Gemma4Attention assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      int headDim = config.layerHeadDim(layerIndex);
      boolean sliding = config.isSlidingLayer(layerIndex);
      boolean shared = config.isKvSharedLayer(layerIndex);
      int window = sliding ? config.slidingWindow() : 0;
      String p = selfAttn(layerIndex);
      RotaryEmbedding rope = sliding
        ? RotaryEmbedding.get(headDim, headDim, config.maxPositionEmbeddings(),
        config.ropeLocalBaseFreq())
        : RotaryEmbedding.proportional(
        headDim,
        config.maxPositionEmbeddings(),
        config.ropeTheta(),
        config.gemma4().fullPartialRotaryFactor());
      return new Gemma4Attention(
        qatLinear(weights, p + "q_proj.weight"),
        shared ? null : qatLinear(weights, p + "k_proj.weight"),
        shared ? null : qatLinear(weights, p + "v_proj.weight"),
        qatLinear(weights, p + O_PROJ_WEIGHT),
        rope,
        new Attention(
          config.numAttentionHeads(),
          headDim,
          config.attentionScale(),
          config.numKeyValueHeads(),
          window,
          config.kvProducerLayer(layerIndex),
          !shared),
        new RMSNorm(weights.require(p + Q_NORM_WEIGHT), config.rmsNormEps(), false),
        shared ? null : new RMSNorm(weights.require(p + K_NORM_WEIGHT), config.rmsNormEps(), false),
        shared ? null : RMSNorm.weightless(config.rmsNormEps()),
        shared,
        config.numAttentionHeads(),
        config.numKeyValueHeads(),
        headDim);
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor q = this.reshapeHeads(
        this.qProj.forward(hiddenStates, context), this.numHeads);
      q = this.normHeads(q, this.qNorm);
      Tensor k;
      Tensor v;
      if (this.kvShared) {
        Tensor[] rotated = this.rotaryEmb.forward(
          positions, q, Tensor.zeros(q.size(0), this.numKvHeads, this.headDim));
        q = rotated[0];
        k = Tensor.zeros(q.size(0), this.numKvHeads, this.headDim);
        v = Tensor.zeros(q.size(0), this.numKvHeads, this.headDim);
      } else {
        k = this.normHeads(
          this.reshapeHeads(this.kProj.forward(hiddenStates, context), this.numKvHeads),
          this.kNorm);
        v = this.vNorm.forward(
          this.reshapeHeads(this.vProj.forward(hiddenStates, context), this.numKvHeads));
        Tensor[] rotated = this.rotaryEmb.forward(positions, q, k);
        q = rotated[0];
        k = rotated[1];
      }
      Tensor o = this.attn.forward(q, k, v, context);
      return this.oProj.forward(o.reshape(o.size(0), this.numHeads * this.headDim), context);
    }

    private Tensor reshapeHeads(final Tensor projected, final int heads) {
      return projected.reshape(projected.size(0), heads, this.headDim);
    }

    private Tensor normHeads(final Tensor x, final RMSNorm norm) {
      return norm.forward(x.reshape(x.size(0) * x.size(1), this.headDim))
        .reshape(x.size(0), x.size(1), this.headDim);
    }
  }

  record Gemma4Mlp(Linear gateProj, Linear upProj, Linear downProj) {
    Gemma4Mlp(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Gemma4Mlp(final Gemma4Mlp assembled) {
      this(assembled.gateProj, assembled.upProj, assembled.downProj);
    }

    private static Gemma4Mlp assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      String act = config.effectiveActivation().toLowerCase(ROOT);
      if (!act.contains("gelu")) {
        throw new IllegalArgumentException(
          "gemma4 architecture expects gelu_pytorch_tanh, got " + act);
      }
      String p = mlp(layerIndex);
      return new Gemma4Mlp(
        qatLinear(weights, p + "gate_proj.weight"),
        qatLinear(weights, p + "up_proj.weight"),
        qatLinear(weights, p + "down_proj.weight"));
    }

    Tensor forward(final Tensor x, final Context context) {
      return this.downProj.forward(
        Ops.mul(Ops.gelu(this.gateProj.forward(x, context)), this.upProj.forward(x, context)),
        context);
    }
  }

  record Gemma4DecoderLayer(
    Gemma4Attention selfAttn,
    Gemma4Mlp mlp,
    RMSNorm inputLayernorm,
    RMSNorm postAttentionLayernorm,
    RMSNorm preFeedforwardLayernorm,
    RMSNorm postFeedforwardLayernorm,
    Linear perLayerInputGate,
    Linear perLayerProjection,
    RMSNorm postPerLayerInputNorm,
    float layerScalar
  ) {
    Gemma4DecoderLayer(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Gemma4DecoderLayer(final Gemma4DecoderLayer assembled) {
      this(
        assembled.selfAttn, assembled.mlp,
        assembled.inputLayernorm, assembled.postAttentionLayernorm,
        assembled.preFeedforwardLayernorm, assembled.postFeedforwardLayernorm,
        assembled.perLayerInputGate, assembled.perLayerProjection, assembled.postPerLayerInputNorm,
        assembled.layerScalar);
    }

    private static Gemma4DecoderLayer assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      String p = layer(layerIndex);
      Tensor scalar = weights.require(p + "layer_scalar");
      return new Gemma4DecoderLayer(
        new Gemma4Attention(config, weights, layerIndex),
        new Gemma4Mlp(config, weights, layerIndex),
        new RMSNorm(weights.require(p + INPUT_LAYERNORM), config.rmsNormEps(), false),
        new RMSNorm(weights.require(p + POST_ATTENTION_LAYERNORM), config.rmsNormEps(), false),
        new RMSNorm(weights.require(p + PRE_FEEDFORWARD_LAYERNORM), config.rmsNormEps(), false),
        new RMSNorm(weights.require(p + POST_FEEDFORWARD_LAYERNORM), config.rmsNormEps(), false),
        qatLinear(weights, p + "per_layer_input_gate.weight"),
        qatLinear(weights, p + "per_layer_projection.weight"),
        new RMSNorm(weights.require(p + "post_per_layer_input_norm.weight"), config.rmsNormEps(),
          false),
        scalar.data()[scalar.offset()]);
    }

    Tensor forward(
      final Tensor positions,
      final Tensor hiddenStates,
      final Tensor perLayerInput,
      final Context context
    ) {
      Tensor hidden = Ops.add(
        hiddenStates,
        this.postAttentionLayernorm.forward(
          this.selfAttn.forward(positions, this.inputLayernorm.forward(hiddenStates), context)));
      hidden = Ops.add(
        hidden,
        this.postFeedforwardLayernorm.forward(
          this.mlp.forward(this.preFeedforwardLayernorm.forward(hidden), context)));
      Tensor gated =
        Ops.mul(Ops.gelu(this.perLayerInputGate.forward(hidden, context)), perLayerInput);
      hidden = Ops.add(hidden, this.postPerLayerInputNorm.forward(
        this.perLayerProjection.forward(gated, context)));
      return Ops.scale(hidden, this.layerScalar);
    }
  }

  record Gemma4Model(
    VocabParallelEmbedding embedTokens,
    VocabParallelEmbedding embedTokensPerLayer,
    Linear perLayerModelProjection,
    RMSNorm perLayerProjectionNorm,
    List<Gemma4DecoderLayer> layers,
    RMSNorm norm,
    float embedScale,
    float perLayerTokenScale,
    float perLayerProjScale,
    float perLayerCombineScale,
    int numLayers,
    int perLayerWidth
  ) {
    Gemma4Model(Config.HfConfig config, WeightBag weights) {
      this(assemble(config, weights));
    }

    private Gemma4Model(final Gemma4Model assembled) {
      this(
        assembled.embedTokens, assembled.embedTokensPerLayer, assembled.perLayerModelProjection,
        assembled.perLayerProjectionNorm, assembled.layers, assembled.norm,
        assembled.embedScale, assembled.perLayerTokenScale, assembled.perLayerProjScale,
        assembled.perLayerCombineScale, assembled.numLayers, assembled.perLayerWidth);
    }

    private static Gemma4Model assemble(final Config.HfConfig config, final WeightBag weights) {
      int layers = config.numHiddenLayers();
      int ple = config.gemma4().hiddenSizePerLayerInput();
      List<Gemma4DecoderLayer> built = new ArrayList<>(layers);
      for (int i = 0; i < layers; i++) {
        built.add(new Gemma4DecoderLayer(config, weights, i));
      }
      GemmaQatWeight tokens = weights.requireQat(EMBED_TOKENS);
      GemmaQatWeight perLayer = weights.requireQat("model.embed_tokens_per_layer.weight");
      return new Gemma4Model(
        new VocabParallelEmbedding(
          PackedEmbeddingKernel.of(tokens.rows(), tokens.cols(), EMBED_TOKENS,
            tokens::dequantizeRow)),
        new VocabParallelEmbedding(
          PackedEmbeddingKernel.of(
            perLayer.rows(), perLayer.cols(), "model.embed_tokens_per_layer.weight",
            perLayer::dequantizeRow)),
        new Linear.Row(weights.require("model.per_layer_model_projection.weight")),
        new RMSNorm(weights.require("model.per_layer_projection_norm.weight"), config.rmsNormEps(),
          false),
        List.copyOf(built),
        new RMSNorm(weights.require(MODEL_NORM), config.rmsNormEps(), false),
        (float) Math.sqrt(config.hiddenSize()),
        (float) Math.sqrt(ple),
        (float) (1.0 / Math.sqrt(config.hiddenSize())),
        (float) (1.0 / Math.sqrt(2.0)),
        layers,
        ple);
    }

    Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
      Tensor hidden = Ops.scale(this.embedTokens.forward(inputIds, context), this.embedScale);
      Tensor perLayer = this.combinePerLayerInputs(inputIds, hidden, context);
      for (int i = 0; i < this.layers.size(); i++) {
        hidden = this.layers.get(i).forward(
          positions, hidden, this.sliceLayer(perLayer, i), context);
      }
      return this.norm.forward(hidden);
    }

    private Tensor combinePerLayerInputs(
      final Tensor inputIds,
      final Tensor embeds,
      final Context context
    ) {
      Tensor tokenIdentity = Ops.scale(
          this.embedTokensPerLayer.forward(inputIds, context), this.perLayerTokenScale)
        .reshape(inputIds.numel(), this.numLayers, this.perLayerWidth);
      Tensor projected = Ops.scale(
          this.perLayerModelProjection.forward(embeds, context), this.perLayerProjScale)
        .reshape(inputIds.numel(), this.numLayers, this.perLayerWidth);
      projected = this.perLayerProjectionNorm.forward(projected);
      return Ops.scale(Ops.add(projected, tokenIdentity), this.perLayerCombineScale);
    }

    private Tensor sliceLayer(final Tensor perLayer, final int layerIndex) {
      int tokens = perLayer.size(0);
      Tensor out = Tensor.zeros(tokens, this.perLayerWidth);
      int stride = this.numLayers * this.perLayerWidth;
      for (int t = 0; t < tokens; t++) {
        System.arraycopy(
          perLayer.data(),
          perLayer.offset() + t * stride + layerIndex * this.perLayerWidth,
          out.data(),
          t * this.perLayerWidth,
          this.perLayerWidth);
      }
      return out;
    }
  }
}
