package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.QKV_PROJ_WEIGHT;
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
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding.Tables;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * Causal LM for the {@code llama} architecture family (RMSNorm, RoPE, GQA, SiLU MLP; no Q/K head norms).
 *
 * @since 1.1.0
 */
public record LlamaForCausalLM(LlamaModel model, ParallelLMHead lmHead) implements CausalLM {

  public LlamaForCausalLM(final Config.HfConfig config, final WeightBag weights) {
    this(assemble(requireNonNull(config, "config"), requireNonNull(weights, "weights")));
  }

  private LlamaForCausalLM(final LlamaForCausalLM assembled) {
    this(assembled.model, assembled.lmHead);
  }

  private static LlamaForCausalLM assemble(final Config.HfConfig config, final WeightBag weights) {
    LlamaModel model = new LlamaModel(config, weights);
    Tensor lmWeight = weights.find(LM_HEAD)
      .orElseGet(() -> model.embedTokens().weight());
    return new LlamaForCausalLM(model, new ParallelLMHead(lmWeight));
  }

  @Override
  public String architectureName() {
    return ARCH_LLAMA;
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

  record LlamaAttention(
    Linear.Qkv qkvProj,
    Linear.Row oProj,
    RotaryEmbedding rotaryEmb,
    Attention attn,
    int numHeads,
    int numKvHeads,
    int headDim,
    int qSize,
    int kvSize
  ) {
    LlamaAttention(final Config.HfConfig config, final WeightBag weights, final int layerIndex,
                   final Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private LlamaAttention(final LlamaAttention assembled) {
      this(
        assembled.qkvProj, assembled.oProj, assembled.rotaryEmb, assembled.attn,
        assembled.numHeads, assembled.numKvHeads, assembled.headDim,
        assembled.qSize, assembled.kvSize);
    }

    private static LlamaAttention assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex,
      final Tables ropes
    ) {
      int numHeads = config.numAttentionHeads();
      int numKvHeads = config.numKeyValueHeads();
      int headDim = config.headDim();
      int qSize = numHeads * headDim;
      int kvSize = numKvHeads * headDim;
      float scaling = (float) Math.pow(headDim, -0.5);
      float ropeTheta = config.ropeTheta();
      if (config.ropeScaling() != null && config.ropeScaling().containsKey("rope_theta")) {
        ropeTheta = Json.asFloat(config.ropeScaling().get("rope_theta"), ropeTheta);
      }
      String p = selfAttn(layerIndex);
      Tensor qkvWeight = weights.require(p + QKV_PROJ_WEIGHT);
      Linear.Qkv qkvProj = config.attentionBias()
        ? new Linear.Qkv(qkvWeight, Tensor.zeros(qkvWeight.size(0)))
        : new Linear.Qkv(qkvWeight);
      return new LlamaAttention(
        qkvProj,
        new Linear.Row(weights.require(p + O_PROJ_WEIGHT)),
        ropes.get(headDim, headDim, config.maxPositionEmbeddings(), ropeTheta),
        new Attention(numHeads, headDim, scaling, numKvHeads, layerIndex),
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
      Tensor[] rotated = this.rotaryEmb.forward(positions, q, k);
      Tensor o = this.attn.forward(rotated[0], rotated[1], v, context);
      return this.oProj.forward(o.reshape(o.size(0), this.numHeads * this.headDim), context);
    }
  }

  record LlamaMLP(Linear.Merged gateUpProj, Linear.Row downProj) {
    LlamaMLP(final Config.HfConfig config, final WeightBag weights, final int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private LlamaMLP(final LlamaMLP assembled) {
      this(assembled.gateUpProj, assembled.downProj);
    }

    private static LlamaMLP assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      if (!"silu".equals(config.effectiveActivation()) && !"silu".equals(config.hiddenAct())) {
        throw new IllegalArgumentException(
          "llama architecture expects silu, got " + config.effectiveActivation());
      }
      String p = mlp(layerIndex);
      return new LlamaMLP(
        new Linear.Merged(weights.require(p + GATE_UP_PROJ_WEIGHT)),
        new Linear.Row(weights.require(p + DOWN_PROJ_WEIGHT)));
    }

    Tensor forward(final Tensor x, final Context context) {
      return this.downProj.forward(Ops.siluAndMul(this.gateUpProj.forward(x, context)), context);
    }
  }

  record LlamaDecoderLayer(
    LlamaAttention selfAttn,
    LlamaMLP mlp,
    RMSNorm inputLayernorm,
    RMSNorm postAttentionLayernorm
  ) {
    LlamaDecoderLayer(final Config.HfConfig config, final WeightBag weights, final int layerIndex,
                      final Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private LlamaDecoderLayer(final LlamaDecoderLayer assembled) {
      this(
        assembled.selfAttn, assembled.mlp,
        assembled.inputLayernorm, assembled.postAttentionLayernorm);
    }

    private static LlamaDecoderLayer assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex,
      final Tables ropes
    ) {
      String p = layer(layerIndex);
      return new LlamaDecoderLayer(
        new LlamaAttention(config, weights, layerIndex, ropes),
        new LlamaMLP(config, weights, layerIndex),
        new RMSNorm(weights.require(p + INPUT_LAYERNORM), config.rmsNormEps()),
        new RMSNorm(weights.require(p + POST_ATTENTION_LAYERNORM), config.rmsNormEps()));
    }

    Tensor[] forward(
      final Tensor positions,
      final Tensor hiddenStates,
      final Tensor residual,
      final Context context
    ) {
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

  record LlamaModel(
    VocabParallelEmbedding embedTokens,
    List<LlamaDecoderLayer> layers,
    RMSNorm norm
  ) {
    LlamaModel(final Config.HfConfig config, final WeightBag weights) {
      this(assemble(config, weights));
    }

    private LlamaModel(final LlamaModel assembled) {
      this(assembled.embedTokens, assembled.layers, assembled.norm);
    }

    private static LlamaModel assemble(final Config.HfConfig config, final WeightBag weights) {
      Tables ropes = new Tables();
      List<LlamaDecoderLayer> built = new ArrayList<>(config.numHiddenLayers());
      for (int i = 0; i < config.numHiddenLayers(); i++) {
        built.add(new LlamaDecoderLayer(config, weights, i, ropes));
      }
      return new LlamaModel(
        new VocabParallelEmbedding(weights.require(EMBED_TOKENS)),
        List.copyOf(built),
        new RMSNorm(weights.require(MODEL_NORM), config.rmsNormEps()));
    }

    Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
      Tensor hiddenStates = this.embedTokens.forward(inputIds, context);
      Tensor residual = null;
      for (LlamaDecoderLayer layer : this.layers) {
        Tensor[] out = layer.forward(positions, hiddenStates, residual, context);
        hiddenStates = out[0];
        residual = out[1];
      }
      return this.norm.forward(hiddenStates, residual)[0];
    }
  }
}
