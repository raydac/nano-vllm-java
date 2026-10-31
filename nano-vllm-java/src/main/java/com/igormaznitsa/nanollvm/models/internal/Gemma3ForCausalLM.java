package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.PRE_FEEDFORWARD_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.QKV_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.selfAttn;
import static com.igormaznitsa.nanollvm.tensor.ActivationWorkspace.beginLayer;
import static com.igormaznitsa.nanollvm.tensor.ActivationWorkspace.endLayer;
import static com.igormaznitsa.nanollvm.tensor.ActivationWorkspace.releaseUnless;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.layers.DeviceDecoders;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding.Tables;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Causal LM for the {@code gemma3} / {@code gemma3_text} architecture family. Weights from {@link WeightBag} at
 * construction.
 */
public record Gemma3ForCausalLM(Gemma3Model model, ParallelLMHead lmHead) implements CausalLM {

  public Gemma3ForCausalLM(final Config.HfConfig config, final WeightBag weights) {
    this(assemble(requireNonNull(config, "config"), requireNonNull(weights, "weights")));
  }

  private Gemma3ForCausalLM(final Gemma3ForCausalLM assembled) {
    this(assembled.model, assembled.lmHead);
  }

  private static Gemma3ForCausalLM assemble(final Config.HfConfig config, final WeightBag weights) {
    Gemma3Model model = new Gemma3Model(config, weights);
    // When lm_head is absent in the checkpoint, reuse tied embeddings.
    Tensor lmWeight = weights.find(LM_HEAD)
      .orElseGet(() -> model.embedTokens().weight());
    return new Gemma3ForCausalLM(model, new ParallelLMHead(lmWeight));
  }

  @Override
  public String architectureName() {
    return ARCH_GEMMA3;
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

  record Gemma3Attention(
    Linear.Qkv qkvProj,
    Linear.Row oProj,
    RotaryEmbedding rotaryEmb,
    Attention attn,
    RMSNorm qNorm,
    RMSNorm kNorm,
    int numHeads,
    int numKvHeads,
    int headDim,
    int qSize,
    int kvSize
  ) {
    Gemma3Attention(Config.HfConfig config, WeightBag weights, int layerIndex, Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private Gemma3Attention(final Gemma3Attention assembled) {
      this(
        assembled.qkvProj, assembled.oProj, assembled.rotaryEmb, assembled.attn,
        assembled.qNorm, assembled.kNorm,
        assembled.numHeads, assembled.numKvHeads, assembled.headDim,
        assembled.qSize, assembled.kvSize);
    }

    private static Gemma3Attention assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex,
      final Tables ropes
    ) {
      int numHeads = config.numAttentionHeads();
      int numKvHeads = config.numKeyValueHeads();
      int headDim = config.headDim();
      boolean sliding = config.isSlidingLayer(layerIndex);
      float ropeBase = sliding ? config.ropeLocalBaseFreq() : config.ropeTheta();
      int window = sliding ? config.slidingWindow() : 0;
      String p = selfAttn(layerIndex);
      return new Gemma3Attention(
        new Linear.Qkv(weights.require(p + QKV_PROJ_WEIGHT)),
        new Linear.Row(weights.require(p + O_PROJ_WEIGHT)),
        ropes.get(headDim, headDim, config.maxPositionEmbeddings(), ropeBase),
        new Attention(numHeads, headDim, config.attentionScale(), numKvHeads, window, layerIndex),
        new RMSNorm(weights.require(p + Q_NORM_WEIGHT), config.rmsNormEps(), true),
        new RMSNorm(weights.require(p + K_NORM_WEIGHT), config.rmsNormEps(), true),
        numHeads,
        numKvHeads,
        headDim,
        numHeads * headDim,
        numKvHeads * headDim);
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor qkv = this.qkvProj.forward(hiddenStates, context);
      Tensor[] parts = Ops.splitLast(qkv, this.qSize, this.kvSize, this.kvSize);
      Tensor q = parts[0].reshape(parts[0].size(0), this.numHeads, this.headDim);
      Tensor k = parts[1].reshape(parts[1].size(0), this.numKvHeads, this.headDim);
      Tensor v = parts[2].reshape(parts[2].size(0), this.numKvHeads, this.headDim);
      q = this.qNorm.forward(q);
      k = this.kNorm.forward(k);
      Tensor[] rotated = this.rotaryEmb.forward(positions, q, k, context);
      Tensor o = this.attn.forward(rotated[0], rotated[1], v, context);
      return this.oProj.forward(o.reshape(o.size(0), this.numHeads * this.headDim), context);
    }
  }

  record Gemma3MLP(Linear.Merged gateUpProj, Linear.Row downProj) {
    Gemma3MLP(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Gemma3MLP(final Gemma3MLP assembled) {
      this(assembled.gateUpProj, assembled.downProj);
    }

    private static Gemma3MLP assemble(final Config.HfConfig config, final WeightBag weights,
                                      final int layerIndex) {
      String act = config.effectiveActivation().toLowerCase(ROOT);
      if (!act.contains("gelu")) {
        throw new IllegalArgumentException(
          "gemma3 architecture expects gelu_pytorch_tanh, got " + act);
      }
      String p = mlp(layerIndex);
      return new Gemma3MLP(
        new Linear.Merged(weights.require(p + GATE_UP_PROJ_WEIGHT)),
        new Linear.Row(weights.require(p + DOWN_PROJ_WEIGHT)));
    }

    Tensor forward(final Tensor x, final Context context) {
      return this.downProj.forward(
        Ops.geluPytorchTanhAndMul(this.gateUpProj.forward(x, context)), context);
    }
  }

  record Gemma3DecoderLayer(
    Gemma3Attention selfAttn,
    Gemma3MLP mlp,
    RMSNorm inputLayernorm,
    RMSNorm postAttentionLayernorm,
    RMSNorm preFeedforwardLayernorm,
    RMSNorm postFeedforwardLayernorm
  ) {
    Gemma3DecoderLayer(Config.HfConfig config, WeightBag weights, int layerIndex, Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private Gemma3DecoderLayer(final Gemma3DecoderLayer assembled) {
      this(
        assembled.selfAttn, assembled.mlp,
        assembled.inputLayernorm, assembled.postAttentionLayernorm,
        assembled.preFeedforwardLayernorm, assembled.postFeedforwardLayernorm);
    }

    private static Gemma3DecoderLayer assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex,
      final Tables ropes
    ) {
      String p = layer(layerIndex);
      return new Gemma3DecoderLayer(
        new Gemma3Attention(config, weights, layerIndex, ropes),
        new Gemma3MLP(config, weights, layerIndex),
        new RMSNorm(weights.require(p + INPUT_LAYERNORM), config.rmsNormEps(), true),
        new RMSNorm(weights.require(p + POST_ATTENTION_LAYERNORM), config.rmsNormEps(), true),
        new RMSNorm(weights.require(p + PRE_FEEDFORWARD_LAYERNORM), config.rmsNormEps(), true),
        new RMSNorm(weights.require(p + POST_FEEDFORWARD_LAYERNORM), config.rmsNormEps(), true));
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor device = this.deviceLayer(positions, hiddenStates, context);
      if (device != null) {
        return device;
      }
      Tensor residual = hiddenStates;
      Tensor hidden = this.selfAttn.forward(
        positions, this.inputLayernorm.forward(hiddenStates), context);
      hidden = Ops.add(residual, this.postAttentionLayernorm.forward(hidden));

      residual = hidden;
      hidden = this.mlp.forward(this.preFeedforwardLayernorm.forward(hidden), context);
      return Ops.add(residual, this.postFeedforwardLayernorm.forward(hidden));
    }

    private Tensor deviceLayer(
      final Tensor positions, final Tensor hiddenStates, final Context context
    ) {
      DeviceDecoders.Step step = DeviceDecoders.decoder();
      if (step == null || context == null || context.kvCache() == null) {
        return null;
      }
      int kvLayer = this.selfAttn.attn().getKvLayerIndex();
      Tensor[] out = step.forward(DeviceDecoders.layer()
        .normBranch()
        .gelu()
        .positions(positions)
        .hidden(hiddenStates)
        .context(context)
        .qkv(this.selfAttn.qkvProj())
        .output(this.selfAttn.oProj())
        .gateUp(this.mlp.gateUpProj())
        .down(this.mlp.downProj())
        .inputNorm(this.inputLayernorm)
        .postAttentionNorm(this.postAttentionLayernorm)
        .preMlpNorm(this.preFeedforwardLayernorm)
        .postMlpNorm(this.postFeedforwardLayernorm)
        .queryNorm(this.selfAttn.qNorm())
        .keyNorm(this.selfAttn.kNorm())
        .epsilon(this.inputLayernorm.getEpsilon())
        .rope(this.selfAttn.rotaryEmb())
        .heads(this.selfAttn.numHeads(), this.selfAttn.numKvHeads(), this.selfAttn.headDim())
        .scoreScale(this.selfAttn.attn().getScoreScale())
        .window(this.selfAttn.attn().getSlidingWindow())
        .caches(
          context.kvCache().getKeyCache(kvLayer),
          context.kvCache().getValueCache(kvLayer))
        .build());
      return out == null ? null : out[0];
    }
  }

  record Gemma3Model(
    VocabParallelEmbedding embedTokens,
    List<Gemma3DecoderLayer> layers,
    RMSNorm norm,
    float embedScale
  ) {
    Gemma3Model(Config.HfConfig config, WeightBag weights) {
      this(assemble(config, weights));
    }

    private Gemma3Model(final Gemma3Model assembled) {
      this(assembled.embedTokens, assembled.layers, assembled.norm, assembled.embedScale);
    }

    private static Gemma3Model assemble(final Config.HfConfig config, final WeightBag weights) {
      Tables ropes = new Tables();
      return new Gemma3Model(
        new VocabParallelEmbedding(weights.require(EMBED_TOKENS)),
        IntStream.range(0, config.numHiddenLayers())
          .mapToObj(i -> new Gemma3DecoderLayer(config, weights, i, ropes))
          .toList(),
        new RMSNorm(weights.require(MODEL_NORM), config.rmsNormEps(), true),
        (float) Math.sqrt(config.hiddenSize()));
    }

    Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
      beginLayer();
      Tensor hiddenStates = Ops.scale(this.embedTokens.forward(inputIds, context), this.embedScale);
      for (Gemma3DecoderLayer layer : this.layers) {
        Tensor next = layer.forward(positions, hiddenStates, context);
        endLayer(next);
        releaseUnless(hiddenStates, next);
        hiddenStates = next;
        beginLayer();
      }
      Tensor normed = this.norm.forward(hiddenStates);
      endLayer(normed);
      releaseUnless(hiddenStates, normed);
      return normed;
    }
  }
}
