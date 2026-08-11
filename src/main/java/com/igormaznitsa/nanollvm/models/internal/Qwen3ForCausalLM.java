package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.QKV_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.selfAttn;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * Immutable Qwen3 causal LM. All weights are taken from {@link WeightBag} at construction.
 */
public record Qwen3ForCausalLM(Qwen3Model model, ParallelLMHead lmHead) implements CausalLM {

  public Qwen3ForCausalLM(final Config.HfConfig config, final WeightBag weights) {
    this(assemble(requireNonNull(config, "config"), requireNonNull(weights, "weights")));
  }

  private Qwen3ForCausalLM(final Qwen3ForCausalLM assembled) {
    this(assembled.model, assembled.lmHead);
  }

  private static Qwen3ForCausalLM assemble(final Config.HfConfig config, final WeightBag weights) {
    Qwen3Model model = new Qwen3Model(config, weights);
    Tensor lmWeight = weights.find(LM_HEAD)
      .orElseGet(() -> model.embedTokens().weight());
    return new Qwen3ForCausalLM(model, new ParallelLMHead(lmWeight));
  }

  @Override
  public String architectureName() {
    return ARCH_QWEN3;
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

  record Qwen3Attention(
    Linear.Qkv qkvProj,
    Linear.Row oProj,
    RotaryEmbedding rotaryEmb,
    Attention attn,
    RMSNorm qNorm,
    RMSNorm kNorm,
    boolean qkvBias,
    int numHeads,
    int numKvHeads,
    int headDim,
    int qSize,
    int kvSize
  ) {
    Qwen3Attention(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Qwen3Attention(final Qwen3Attention assembled) {
      this(
        assembled.qkvProj, assembled.oProj, assembled.rotaryEmb, assembled.attn,
        assembled.qNorm, assembled.kNorm, assembled.qkvBias,
        assembled.numHeads, assembled.numKvHeads, assembled.headDim,
        assembled.qSize, assembled.kvSize);
    }

    private static Qwen3Attention assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      int numHeads = config.numAttentionHeads();
      int numKvHeads = config.numKeyValueHeads();
      int headDim = config.headDim();
      int qSize = numHeads * headDim;
      int kvSize = numKvHeads * headDim;
      float scaling = (float) Math.pow(headDim, -0.5);
      boolean qkvBias = config.attentionBias();
      float ropeTheta = config.ropeTheta();
      if (config.ropeScaling() != null && config.ropeScaling().containsKey("rope_theta")) {
        ropeTheta = Json.asFloat(config.ropeScaling().get("rope_theta"), ropeTheta);
      }
      String p = selfAttn(layerIndex);
      Tensor qkvWeight = weights.require(p + QKV_PROJ_WEIGHT);
      Linear.Qkv qkvProj = qkvBias
        ? new Linear.Qkv(qkvWeight, Tensor.zeros(qkvWeight.size(0)))
        : new Linear.Qkv(qkvWeight);
      return new Qwen3Attention(
        qkvProj,
        new Linear.Row(weights.require(p + O_PROJ_WEIGHT)),
        RotaryEmbedding.get(headDim, headDim, config.maxPositionEmbeddings(), ropeTheta),
        new Attention(numHeads, headDim, scaling, numKvHeads, layerIndex),
        qkvBias ? null : new RMSNorm(weights.require(p + Q_NORM_WEIGHT), config.rmsNormEps()),
        qkvBias ? null : new RMSNorm(weights.require(p + K_NORM_WEIGHT), config.rmsNormEps()),
        qkvBias,
        numHeads,
        numKvHeads,
        headDim,
        qSize,
        kvSize);
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor qkv = this.qkvProj.forward(hiddenStates, context);
      Tensor[] parts = Ops.splitLast(qkv, this.qSize, this.kvSize, this.kvSize);
      Tensor q = parts[0].reshape(parts[0].size(0), this.numHeads, this.headDim);
      Tensor k = parts[1].reshape(parts[1].size(0), this.numKvHeads, this.headDim);
      Tensor v = parts[2].reshape(parts[2].size(0), this.numKvHeads, this.headDim);
      if (!this.qkvBias) {
        q = this.normHeads(q, this.qNorm);
        k = this.normHeads(k, this.kNorm);
      }
      Tensor[] rotated = this.rotaryEmb.forward(positions, q, k);
      Tensor o = this.attn.forward(rotated[0], rotated[1], v, context);
      return this.oProj.forward(o.reshape(o.size(0), this.numHeads * this.headDim), context);
    }

    private Tensor normHeads(final Tensor x, final RMSNorm norm) {
      return norm.forward(x.reshape(x.size(0) * x.size(1), this.headDim))
        .reshape(x.size(0), x.size(1), this.headDim);
    }
  }

  record Qwen3MLP(Linear.Merged gateUpProj, Linear.Row downProj) {
    Qwen3MLP(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Qwen3MLP(final Qwen3MLP assembled) {
      this(assembled.gateUpProj, assembled.downProj);
    }

    private static Qwen3MLP assemble(final Config.HfConfig config, final WeightBag weights,
                                     final int layerIndex) {
      if (!"silu".equals(config.effectiveActivation()) && !"silu".equals(config.hiddenAct())) {
        throw new IllegalArgumentException("only silu supported for Qwen3");
      }
      String p = mlp(layerIndex);
      return new Qwen3MLP(
        new Linear.Merged(weights.require(p + GATE_UP_PROJ_WEIGHT)),
        new Linear.Row(weights.require(p + DOWN_PROJ_WEIGHT)));
    }

    Tensor forward(final Tensor x, final Context context) {
      return this.downProj.forward(Ops.siluAndMul(this.gateUpProj.forward(x, context)), context);
    }
  }

  record Qwen3DecoderLayer(
    Qwen3Attention selfAttn,
    Qwen3MLP mlp,
    RMSNorm inputLayernorm,
    RMSNorm postAttentionLayernorm
  ) {
    Qwen3DecoderLayer(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Qwen3DecoderLayer(final Qwen3DecoderLayer assembled) {
      this(
        assembled.selfAttn, assembled.mlp,
        assembled.inputLayernorm, assembled.postAttentionLayernorm);
    }

    private static Qwen3DecoderLayer assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      String p = layer(layerIndex);
      return new Qwen3DecoderLayer(
        new Qwen3Attention(config, weights, layerIndex),
        new Qwen3MLP(config, weights, layerIndex),
        new RMSNorm(weights.require(p + INPUT_LAYERNORM), config.rmsNormEps()),
        new RMSNorm(weights.require(p + POST_ATTENTION_LAYERNORM), config.rmsNormEps()));
    }

    Tensor[] forward(
      final Tensor positions,
      final Tensor hiddenStates,
      final Tensor residual,
      final Context context) {
      Tensor hidden;
      Tensor resid;
      if (residual == null) {
        resid = hiddenStates;
        hidden = this.inputLayernorm.forward(hiddenStates);
      } else {
        Tensor[] n = this.inputLayernorm.forward(hiddenStates, residual);
        hidden = n[0];
        resid = n[1];
      }
      hidden = this.selfAttn.forward(positions, hidden, context);
      Tensor[] n = this.postAttentionLayernorm.forward(hidden, resid);
      hidden = this.mlp.forward(n[0], context);
      return new Tensor[] {hidden, n[1]};
    }
  }

  record Qwen3Model(
    VocabParallelEmbedding embedTokens,
    List<Qwen3DecoderLayer> layers,
    RMSNorm norm
  ) {
    Qwen3Model(Config.HfConfig config, WeightBag weights) {
      this(assemble(config, weights));
    }

    private Qwen3Model(final Qwen3Model assembled) {
      this(assembled.embedTokens, assembled.layers, assembled.norm);
    }

    private static Qwen3Model assemble(final Config.HfConfig config, final WeightBag weights) {
      List<Qwen3DecoderLayer> built = new ArrayList<>(config.numHiddenLayers());
      for (int i = 0; i < config.numHiddenLayers(); i++) {
        built.add(new Qwen3DecoderLayer(config, weights, i));
      }
      return new Qwen3Model(
        new VocabParallelEmbedding(weights.require(EMBED_TOKENS)),
        List.copyOf(built),
        new RMSNorm(weights.require(MODEL_NORM), config.rmsNormEps()));
    }

    Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
      Tensor hiddenStates = this.embedTokens.forward(inputIds, context);
      Tensor residual = null;
      for (Qwen3DecoderLayer layer : this.layers) {
        Tensor[] out = layer.forward(positions, hiddenStates, residual, context);
        hiddenStates = out[0];
        residual = out[1];
      }
      return this.norm.forward(hiddenStates, residual)[0];
    }
  }
}
