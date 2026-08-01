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

/**
 * Text-only Gemma 3 causal LM (e.g. google/gemma-3-270m).
 */
public final class Gemma3ForCausalLM implements CausalLM {

  public static final Map<String, Object[]> PACKED_MODULES_MAPPING = Map.of(
      "q_proj", new Object[] {"qkv_proj", "q"},
      "k_proj", new Object[] {"qkv_proj", "k"},
      "v_proj", new Object[] {"qkv_proj", "v"},
      "gate_proj", new Object[] {"gate_up_proj", 0},
      "up_proj", new Object[] {"gate_up_proj", 1}
  );

  private final Gemma3Model model;
  private final ParallelLMHead lmHead;
  private final Map<String, WeightSlot> parameters = new LinkedHashMap<>();

  public Gemma3ForCausalLM(Config.HfConfig config) {
    this.model = new Gemma3Model(config);
    this.lmHead = new ParallelLMHead(config.vocabSize(), config.hiddenSize());
    this.registerParameters();
    // Gemma3-270M has no lm_head in the checkpoint; HF ties embeddings (config often omits the flag).
    this.lmHead.setWeight(this.model.embedTokens.weight());
  }

  @Override
  public String architectureName() {
    return "gemma3";
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
    List<Attention> list = new ArrayList<>();
    for (Gemma3DecoderLayer layer : this.model.layers) {
      list.add(layer.selfAttn.attn);
    }
    return list;
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

  private void registerParameters() {
    this.parameters.put("model.embed_tokens.weight",
        WeightSlot.of(this.model.embedTokens::loadWeight));
    for (int i = 0; i < this.model.layers.size(); i++) {
      Gemma3DecoderLayer layer = this.model.layers.get(i);
      String p = "model.layers." + i + ".";
      this.parameters.put(p + "input_layernorm.weight",
          WeightSlot.of(layer.inputLayernorm::setWeight));
      this.parameters.put(p + "post_attention_layernorm.weight",
          WeightSlot.of(layer.postAttentionLayernorm::setWeight));
      this.parameters.put(p + "pre_feedforward_layernorm.weight",
          WeightSlot.of(layer.preFeedforwardLayernorm::setWeight));
      this.parameters.put(p + "post_feedforward_layernorm.weight",
          WeightSlot.of(layer.postFeedforwardLayernorm::setWeight));
      this.parameters.put(p + "self_attn.qkv_proj.weight", WeightSlot.qkv(layer.selfAttn.qkvProj));
      this.parameters.put(p + "self_attn.o_proj.weight",
          WeightSlot.of(layer.selfAttn.oProj::loadWeight));
      this.parameters.put(p + "self_attn.q_norm.weight",
          WeightSlot.of(layer.selfAttn.qNorm::setWeight));
      this.parameters.put(p + "self_attn.k_norm.weight",
          WeightSlot.of(layer.selfAttn.kNorm::setWeight));
      this.parameters.put(p + "mlp.gate_up_proj.weight", WeightSlot.merged(layer.mlp.gateUpProj));
      this.parameters.put(p + "mlp.down_proj.weight",
          WeightSlot.of(layer.mlp.downProj::loadWeight));
    }
    this.parameters.put("model.norm.weight", WeightSlot.of(this.model.norm::setWeight));
    this.parameters.put("lm_head.weight", WeightSlot.of(this.lmHead::loadWeight));
  }

  static final class Gemma3Attention {
    final Linear.Qkv qkvProj;
    final Linear.Row oProj;
    final RotaryEmbedding rotaryEmb;
    final Attention attn;
    final RMSNorm qNorm;
    final RMSNorm kNorm;
    final int numHeads;
    final int numKvHeads;
    final int headDim;
    final int qSize;
    final int kvSize;

    Gemma3Attention(Config.HfConfig config, int layerIndex) {
      this.numHeads = config.numAttentionHeads();
      this.numKvHeads = config.numKeyValueHeads();
      this.headDim = config.headDim();
      this.qSize = this.numHeads * this.headDim;
      this.kvSize = this.numKvHeads * this.headDim;
      boolean sliding = config.isSlidingLayer(layerIndex);
      float ropeBase = sliding ? config.ropeLocalBaseFreq() : config.ropeTheta();
      int window = sliding ? config.slidingWindow() : 0;
      this.qkvProj = new Linear.Qkv(
          config.hiddenSize(), this.headDim,
          config.numAttentionHeads(), config.numKeyValueHeads(), false
      );
      this.oProj =
          new Linear.Row(config.numAttentionHeads() * this.headDim, config.hiddenSize(), false);
      this.rotaryEmb =
          RotaryEmbedding.get(this.headDim, this.headDim, config.maxPositionEmbeddings(), ropeBase);
      this.attn =
          new Attention(this.numHeads, this.headDim, config.attentionScale(), this.numKvHeads,
              window);
      this.qNorm = new RMSNorm(this.headDim, config.rmsNormEps(), true);
      this.kNorm = new RMSNorm(this.headDim, config.rmsNormEps(), true);
    }

    Tensor forward(Tensor positions, Tensor hiddenStates) {
      Tensor qkv = this.qkvProj.forward(hiddenStates);
      Tensor[] parts = Ops.splitLast(qkv, this.qSize, this.kvSize, this.kvSize);
      Tensor q = parts[0].reshape(parts[0].size(0), this.numHeads, this.headDim);
      Tensor k = parts[1].reshape(parts[1].size(0), this.numKvHeads, this.headDim);
      Tensor v = parts[2].reshape(parts[2].size(0), this.numKvHeads, this.headDim);
      q = this.normHeads(q, this.qNorm);
      k = this.normHeads(k, this.kNorm);
      Tensor[] rotated = this.rotaryEmb.forward(positions, q, k);
      Tensor o = this.attn.forward(rotated[0], rotated[1], v);
      return this.oProj.forward(o.reshape(o.size(0), this.numHeads * this.headDim));
    }

    private Tensor normHeads(Tensor x, RMSNorm norm) {
      return norm.forward(x.reshape(x.size(0) * x.size(1), this.headDim))
          .reshape(x.size(0), x.size(1), this.headDim);
    }
  }

  static final class Gemma3MLP {
    final Linear.Merged gateUpProj;
    final Linear.Row downProj;

    Gemma3MLP(Config.HfConfig config) {
      String act = config.effectiveActivation().toLowerCase();
      if (!act.contains("gelu")) {
        throw new IllegalArgumentException("Gemma3 expects gelu_pytorch_tanh, got " + act);
      }
      this.gateUpProj = new Linear.Merged(
          config.hiddenSize(),
          new int[] {config.intermediateSize(), config.intermediateSize()},
          false
      );
      this.downProj = new Linear.Row(config.intermediateSize(), config.hiddenSize(), false);
    }

    Tensor forward(Tensor x) {
      return this.downProj.forward(Ops.geluPytorchTanhAndMul(this.gateUpProj.forward(x)));
    }
  }

  static final class Gemma3DecoderLayer {
    final Gemma3Attention selfAttn;
    final Gemma3MLP mlp;
    final RMSNorm inputLayernorm;
    final RMSNorm postAttentionLayernorm;
    final RMSNorm preFeedforwardLayernorm;
    final RMSNorm postFeedforwardLayernorm;

    Gemma3DecoderLayer(Config.HfConfig config, int layerIndex) {
      this.selfAttn = new Gemma3Attention(config, layerIndex);
      this.mlp = new Gemma3MLP(config);
      this.inputLayernorm = new RMSNorm(config.hiddenSize(), config.rmsNormEps(), true);
      this.postAttentionLayernorm = new RMSNorm(config.hiddenSize(), config.rmsNormEps(), true);
      this.preFeedforwardLayernorm = new RMSNorm(config.hiddenSize(), config.rmsNormEps(), true);
      this.postFeedforwardLayernorm = new RMSNorm(config.hiddenSize(), config.rmsNormEps(), true);
    }

    Tensor forward(Tensor positions, Tensor hiddenStates) {
      Tensor residual = hiddenStates;
      hiddenStates = this.selfAttn.forward(positions, this.inputLayernorm.forward(hiddenStates));
      hiddenStates = this.add(residual, this.postAttentionLayernorm.forward(hiddenStates));

      residual = hiddenStates;
      hiddenStates = this.mlp.forward(this.preFeedforwardLayernorm.forward(hiddenStates));
      return this.add(residual, this.postFeedforwardLayernorm.forward(hiddenStates));
    }

    private Tensor add(Tensor a, Tensor b) {
      Tensor out = Tensor.zeros(a.shape());
      float[] ad = a.data();
      float[] bd = b.data();
      float[] od = out.data();
      int n = a.numel();
      int aOff = a.offset();
      int bOff = b.offset();
      for (int i = 0; i < n; i++) {
        od[i] = ad[aOff + i] + bd[bOff + i];
      }
      return out;
    }
  }

  static final class Gemma3Model {
    final VocabParallelEmbedding embedTokens;
    final List<Gemma3DecoderLayer> layers = new ArrayList<>();
    final RMSNorm norm;
    final float embedScale;

    Gemma3Model(Config.HfConfig config) {
      this.embedTokens = new VocabParallelEmbedding(config.vocabSize(), config.hiddenSize());
      this.embedScale = (float) Math.sqrt(config.hiddenSize());
      for (int i = 0; i < config.numHiddenLayers(); i++) {
        this.layers.add(new Gemma3DecoderLayer(config, i));
      }
      this.norm = new RMSNorm(config.hiddenSize(), config.rmsNormEps(), true);
    }

    Tensor forward(Tensor inputIds, Tensor positions) {
      Tensor hiddenStates = this.scaleEmbed(this.embedTokens.forward(inputIds));
      for (Gemma3DecoderLayer layer : this.layers) {
        hiddenStates = layer.forward(positions, hiddenStates);
      }
      return this.norm.forward(hiddenStates);
    }

    private Tensor scaleEmbed(Tensor emb) {
      Tensor out = Tensor.zeros(emb.shape());
      float[] ed = emb.data();
      float[] od = out.data();
      int n = emb.numel();
      int off = emb.offset();
      for (int i = 0; i < n; i++) {
        od[i] = ed[off + i] * this.embedScale;
      }
      return out;
    }
  }
}
