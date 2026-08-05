package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.models.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.WeightNames.QKV_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.WeightNames.selfAttn;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.utils.Json;
import java.util.ArrayList;
import java.util.List;

/**
 * Immutable Qwen3 causal LM. All weights are taken from {@link WeightBag} at construction.
 */
public record Qwen3ForCausalLM(Qwen3Model model, ParallelLMHead lmHead) implements CausalLM {

  public Qwen3ForCausalLM(Config.HfConfig config, WeightBag weights) {
    this(assemble(requireNonNull(config, "config"), requireNonNull(weights, "weights")));
  }

  private Qwen3ForCausalLM(Qwen3ForCausalLM assembled) {
    this(assembled.model, assembled.lmHead);
  }

  private static Qwen3ForCausalLM assemble(Config.HfConfig config, WeightBag weights) {
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
  public Tensor forward(Tensor inputIds, Tensor positions) {
    return this.model.forward(inputIds, positions);
  }

  @Override
  public Tensor computeLogits(Tensor hiddenStates) {
    return this.lmHead.forward(hiddenStates);
  }

  @Override
  public List<Attention> attentionLayers() {
    return this.model.layers().stream()
        .map(layer -> layer.selfAttn().attn())
        .toList();
  }

  @Override
  public boolean equals(Object other) {
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

    private Qwen3Attention(Qwen3Attention assembled) {
      this(
          assembled.qkvProj, assembled.oProj, assembled.rotaryEmb, assembled.attn,
          assembled.qNorm, assembled.kNorm, assembled.qkvBias,
          assembled.numHeads, assembled.numKvHeads, assembled.headDim,
          assembled.qSize, assembled.kvSize);
    }

    private static Qwen3Attention assemble(
        Config.HfConfig config,
        WeightBag weights,
        int layerIndex
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

    Tensor forward(Tensor positions, Tensor hiddenStates) {
      Tensor qkv = this.qkvProj.forward(hiddenStates);
      Tensor[] parts = Ops.splitLast(qkv, this.qSize, this.kvSize, this.kvSize);
      Tensor q = parts[0].reshape(parts[0].size(0), this.numHeads, this.headDim);
      Tensor k = parts[1].reshape(parts[1].size(0), this.numKvHeads, this.headDim);
      Tensor v = parts[2].reshape(parts[2].size(0), this.numKvHeads, this.headDim);
      if (!this.qkvBias) {
        q = this.normHeads(q, this.qNorm);
        k = this.normHeads(k, this.kNorm);
      }
      Tensor[] rotated = this.rotaryEmb.forward(positions, q, k);
      Tensor o = this.attn.forward(rotated[0], rotated[1], v);
      return this.oProj.forward(o.reshape(o.size(0), this.numHeads * this.headDim));
    }

    private Tensor normHeads(Tensor x, RMSNorm norm) {
      return norm.forward(x.reshape(x.size(0) * x.size(1), this.headDim))
          .reshape(x.size(0), x.size(1), this.headDim);
    }
  }

  record Qwen3MLP(Linear.Merged gateUpProj, Linear.Row downProj) {
    Qwen3MLP(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private Qwen3MLP(Qwen3MLP assembled) {
      this(assembled.gateUpProj, assembled.downProj);
    }

    private static Qwen3MLP assemble(Config.HfConfig config, WeightBag weights, int layerIndex) {
      if (!"silu".equals(config.effectiveActivation()) && !"silu".equals(config.hiddenAct())) {
        throw new IllegalArgumentException("only silu supported for Qwen3");
      }
      String p = mlp(layerIndex);
      return new Qwen3MLP(
          new Linear.Merged(weights.require(p + GATE_UP_PROJ_WEIGHT)),
          new Linear.Row(weights.require(p + DOWN_PROJ_WEIGHT)));
    }

    Tensor forward(Tensor x) {
      return this.downProj.forward(Ops.siluAndMul(this.gateUpProj.forward(x)));
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

    private Qwen3DecoderLayer(Qwen3DecoderLayer assembled) {
      this(
          assembled.selfAttn, assembled.mlp,
          assembled.inputLayernorm, assembled.postAttentionLayernorm);
    }

    private static Qwen3DecoderLayer assemble(
        Config.HfConfig config,
        WeightBag weights,
        int layerIndex
    ) {
      String p = layer(layerIndex);
      return new Qwen3DecoderLayer(
          new Qwen3Attention(config, weights, layerIndex),
          new Qwen3MLP(config, weights, layerIndex),
          new RMSNorm(weights.require(p + INPUT_LAYERNORM), config.rmsNormEps()),
          new RMSNorm(weights.require(p + POST_ATTENTION_LAYERNORM), config.rmsNormEps()));
    }

    Tensor[] forward(Tensor positions, Tensor hiddenStates, Tensor residual) {
      if (residual == null) {
        residual = hiddenStates;
        hiddenStates = this.inputLayernorm.forward(hiddenStates);
      } else {
        Tensor[] n = this.inputLayernorm.forward(hiddenStates, residual);
        hiddenStates = n[0];
        residual = n[1];
      }
      hiddenStates = this.selfAttn.forward(positions, hiddenStates);
      Tensor[] n = this.postAttentionLayernorm.forward(hiddenStates, residual);
      hiddenStates = this.mlp.forward(n[0]);
      return new Tensor[] {hiddenStates, n[1]};
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

    private Qwen3Model(Qwen3Model assembled) {
      this(assembled.embedTokens, assembled.layers, assembled.norm);
    }

    private static Qwen3Model assemble(Config.HfConfig config, WeightBag weights) {
      List<Qwen3DecoderLayer> built = new ArrayList<>(config.numHiddenLayers());
      for (int i = 0; i < config.numHiddenLayers(); i++) {
        built.add(new Qwen3DecoderLayer(config, weights, i));
      }
      return new Qwen3Model(
          new VocabParallelEmbedding(weights.require(EMBED_TOKENS)),
          List.copyOf(built),
          new RMSNorm(weights.require(MODEL_NORM), config.rmsNormEps()));
    }

    Tensor forward(Tensor inputIds, Tensor positions) {
      Tensor hiddenStates = this.embedTokens.forward(inputIds);
      Tensor residual = null;
      for (Qwen3DecoderLayer layer : this.layers) {
        Tensor[] out = layer.forward(positions, hiddenStates, residual);
        hiddenStates = out[0];
        residual = out[1];
      }
      return this.norm.forward(hiddenStates, residual)[0];
    }
  }
}
