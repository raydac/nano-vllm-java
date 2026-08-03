package com.igormaznitsa.nanollvm.models;

import com.igormaznitsa.nanollvm.Config;
import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding.ParallelLMHead;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Qwen3ForCausalLM implements CausalLM {

  public static final Map<String, Object[]> PACKED_MODULES_MAPPING = Map.of(
      "q_proj", new Object[] {"qkv_proj", "q"},
      "k_proj", new Object[] {"qkv_proj", "k"},
      "v_proj", new Object[] {"qkv_proj", "v"},
      "gate_proj", new Object[] {"gate_up_proj", 0},
      "up_proj", new Object[] {"gate_up_proj", 1}
  );

  private final Qwen3Model model;
  private final ParallelLMHead lmHead;
  private final Map<String, WeightSlot> parameters = new LinkedHashMap<>();

  public Qwen3ForCausalLM(Config.HfConfig config) {
    this.model = new Qwen3Model(config);
    this.lmHead = new ParallelLMHead(config.vocabSize(), config.hiddenSize());
    this.registerParameters();
    if (config.tieWordEmbeddings()) {
      this.lmHead.setWeight(this.model.embedTokens.weight());
    }
  }

  @Override
  public String architectureName() {
    return "qwen3";
  }

  @Override
  public Map<String, Object[]> packedModulesMapping() {
    return PACKED_MODULES_MAPPING;
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
    return this.model.layers.stream()
        .map(layer -> layer.selfAttn.attn)
        .toList();
  }

  @Override
  public WeightSlot getParameter(String name) {
    WeightSlot slot = this.parameters.get(name);
    if (slot == null) {
      throw new IllegalArgumentException("unknown parameter: " + name);
    }
    return slot;
  }

  @Override
  public boolean hasParameter(String name) {
    return this.parameters.containsKey(name);
  }

  @Override
  public void seal() {
    this.parameters.clear();
  }

  private void registerParameters() {
    this.parameters.put("model.embed_tokens.weight",
        WeightSlot.of(this.model.embedTokens::loadWeight));
    for (int i = 0; i < this.model.layers.size(); i++) {
      Qwen3DecoderLayer layer = this.model.layers.get(i);
      String p = "model.layers." + i + ".";
      this.parameters.put(p + "input_layernorm.weight",
          WeightSlot.of(layer.inputLayernorm::setWeight));
      this.parameters.put(p + "post_attention_layernorm.weight",
          WeightSlot.of(layer.postAttentionLayernorm::setWeight));
      this.parameters.put(p + "self_attn.qkv_proj.weight", WeightSlot.qkv(layer.selfAttn.qkvProj));
      this.parameters.put(p + "self_attn.o_proj.weight",
          WeightSlot.of(layer.selfAttn.oProj::loadWeight));
      if (layer.selfAttn.qNorm != null) {
        this.parameters.put(p + "self_attn.q_norm.weight",
            WeightSlot.of(layer.selfAttn.qNorm::setWeight));
        this.parameters.put(p + "self_attn.k_norm.weight",
            WeightSlot.of(layer.selfAttn.kNorm::setWeight));
      }
      this.parameters.put(p + "mlp.gate_up_proj.weight", WeightSlot.merged(layer.mlp.gateUpProj));
      this.parameters.put(p + "mlp.down_proj.weight",
          WeightSlot.of(layer.mlp.downProj::loadWeight));
    }
    this.parameters.put("model.norm.weight", WeightSlot.of(this.model.norm::setWeight));
    this.parameters.put("lm_head.weight", WeightSlot.of(this.lmHead::loadWeight));
  }

  static final class Qwen3Attention {
    final Linear.Qkv qkvProj;
    final Linear.Row oProj;
    final RotaryEmbedding rotaryEmb;
    final Attention attn;
    final RMSNorm qNorm;
    final RMSNorm kNorm;
    final boolean qkvBias;
    final int numHeads;
    final int numKvHeads;
    final int headDim;
    final int qSize;
    final int kvSize;

    Qwen3Attention(Config.HfConfig config, int layerIndex) {
      int tpSize = 1;
      this.numHeads = config.numAttentionHeads() / tpSize;
      this.numKvHeads = config.numKeyValueHeads() / tpSize;
      this.headDim = config.headDim();
      this.qSize = this.numHeads * this.headDim;
      this.kvSize = this.numKvHeads * this.headDim;
      float scaling = (float) Math.pow(this.headDim, -0.5);
      this.qkvBias = config.attentionBias();
      float ropeTheta = config.ropeTheta();
      if (config.ropeScaling() != null && config.ropeScaling().containsKey("rope_theta")) {
        ropeTheta =
            com.igormaznitsa.nanollvm.utils.Json.asFloat(config.ropeScaling().get("rope_theta"),
                ropeTheta);
      }
      this.qkvProj = new Linear.Qkv(
          config.hiddenSize(), this.headDim,
          config.numAttentionHeads(), config.numKeyValueHeads(), this.qkvBias
      );
      this.oProj =
          new Linear.Row(config.numAttentionHeads() * this.headDim, config.hiddenSize(), false);
      this.rotaryEmb =
          RotaryEmbedding.get(this.headDim, this.headDim, config.maxPositionEmbeddings(),
              ropeTheta);
      this.attn = new Attention(this.numHeads, this.headDim, scaling, this.numKvHeads, layerIndex);
      if (!this.qkvBias) {
        this.qNorm = new RMSNorm(this.headDim, config.rmsNormEps());
        this.kNorm = new RMSNorm(this.headDim, config.rmsNormEps());
      } else {
        this.qNorm = null;
        this.kNorm = null;
      }
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

  static final class Qwen3MLP {
    final Linear.Merged gateUpProj;
    final Linear.Row downProj;

    Qwen3MLP(Config.HfConfig config) {
      if (!"silu".equals(config.effectiveActivation()) && !"silu".equals(config.hiddenAct())) {
        throw new IllegalArgumentException("only silu supported for Qwen3");
      }
      this.gateUpProj = new Linear.Merged(
          config.hiddenSize(),
          new int[] {config.intermediateSize(), config.intermediateSize()},
          false
      );
      this.downProj = new Linear.Row(config.intermediateSize(), config.hiddenSize(), false);
    }

    Tensor forward(Tensor x) {
      return this.downProj.forward(Ops.siluAndMul(this.gateUpProj.forward(x)));
    }
  }

  static final class Qwen3DecoderLayer {
    final Qwen3Attention selfAttn;
    final Qwen3MLP mlp;
    final RMSNorm inputLayernorm;
    final RMSNorm postAttentionLayernorm;

    Qwen3DecoderLayer(Config.HfConfig config, int layerIndex) {
      this.selfAttn = new Qwen3Attention(config, layerIndex);
      this.mlp = new Qwen3MLP(config);
      this.inputLayernorm = new RMSNorm(config.hiddenSize(), config.rmsNormEps());
      this.postAttentionLayernorm = new RMSNorm(config.hiddenSize(), config.rmsNormEps());
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

  static final class Qwen3Model {
    final VocabParallelEmbedding embedTokens;
    final List<Qwen3DecoderLayer> layers = new ArrayList<>();
    final RMSNorm norm;

    Qwen3Model(Config.HfConfig config) {
      this.embedTokens = new VocabParallelEmbedding(config.vocabSize(), config.hiddenSize());
      for (int i = 0; i < config.numHiddenLayers(); i++) {
        this.layers.add(new Qwen3DecoderLayer(config, i));
      }
      this.norm = new RMSNorm(config.hiddenSize(), config.rmsNormEps());
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
