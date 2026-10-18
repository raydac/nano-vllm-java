package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3_5;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.K_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.LM_HEAD;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.MODEL_NORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.O_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.POST_ATTENTION_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.Q_NORM_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.layer;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.linearAttn;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.mlp;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.selfAttn;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Attention;
import com.igormaznitsa.nanollvm.layers.GatedDeltaNet;
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
 * Causal LM for Qwen3.5 text: three Gated DeltaNet layers per full-attention layer.
 * Vision tensors in a multimodal checkpoint are not part of this graph.
 *
 * <p>Full attention uses a query-projection gate, Q/K RMSNorm with {@code (1 + w)}, and partial
 * RoPE on the leading fraction of the head. Text positions use that partial table; image and
 * video positions are not applied.
 *
 * @since 1.5.0
 */
public record Qwen35ForCausalLM(Qwen35Model model, ParallelLMHead lmHead) implements CausalLM {

  public Qwen35ForCausalLM(final Config.HfConfig config, final WeightBag weights) {
    this(assemble(requireNonNull(config, "config"), requireNonNull(weights, "weights")));
  }

  private Qwen35ForCausalLM(final Qwen35ForCausalLM assembled) {
    this(assembled.model, assembled.lmHead);
  }

  private static Qwen35ForCausalLM assemble(final Config.HfConfig config, final WeightBag weights) {
    Qwen35Model model = new Qwen35Model(config, weights);
    Tensor lmWeight = weights.find(LM_HEAD).orElseGet(() -> model.embedTokens().weight());
    return new Qwen35ForCausalLM(model, new ParallelLMHead(lmWeight));
  }

  private static Qwen35DecoderLayer layerAt(
    final Config.HfConfig config,
    final Config.Qwen35Text spec,
    final WeightBag weights,
    final int layerIndex,
    final RotaryEmbedding rotary
  ) {
    String prefix = layer(layerIndex);
    Qwen35Mlp feedForward = new Qwen35Mlp(
      new Linear.Merged(weights.require(mlp(layerIndex) + GATE_UP_PROJ_WEIGHT)),
      new Linear.Row(weights.require(mlp(layerIndex) + DOWN_PROJ_WEIGHT)));
    RMSNorm input =
      new RMSNorm(weights.require(prefix + INPUT_LAYERNORM), config.rmsNormEps(), true);
    RMSNorm post = new RMSNorm(
      weights.require(prefix + POST_ATTENTION_LAYERNORM), config.rmsNormEps(), true);
    if (config.isLinearAttentionLayer(layerIndex)) {
      return new Qwen35DecoderLayer(
        deltaNet(spec, weights, layerIndex, config.rmsNormEps()),
        null,
        feedForward,
        input,
        post);
    }
    return new Qwen35DecoderLayer(null, attention(config, weights, layerIndex, rotary), feedForward,
      input, post);
  }

  private static GatedDeltaNet deltaNet(
    final Config.Qwen35Text spec,
    final WeightBag weights,
    final int layerIndex,
    final float eps
  ) {
    String prefix = linearAttn(layerIndex);
    return new GatedDeltaNet(
      weights.require(prefix + "in_proj_qkv.weight"),
      weights.require(prefix + "in_proj_z.weight"),
      weights.require(prefix + "in_proj_b.weight"),
      weights.require(prefix + "in_proj_a.weight"),
      weights.require(prefix + "conv1d.weight"),
      weights.require(prefix + "A_log"),
      weights.require(prefix + "dt_bias"),
      weights.require(prefix + "norm.weight"),
      weights.require(prefix + "out_proj.weight"),
      spec.linearNumKeyHeads(),
      spec.linearKeyHeadDim(),
      spec.linearNumValueHeads(),
      spec.linearValueHeadDim(),
      eps,
      layerIndex);
  }

  private static Qwen35Attention attention(
    final Config.HfConfig config,
    final WeightBag weights,
    final int layerIndex,
    final RotaryEmbedding rotary
  ) {
    String prefix = selfAttn(layerIndex);
    int numHeads = config.numAttentionHeads();
    int numKvHeads = config.numKeyValueHeads();
    int headDim = config.headDim();
    return new Qwen35Attention(
      new Linear.Row(weights.require(prefix + "q_proj.weight")),
      new Linear.Row(weights.require(prefix + "k_proj.weight")),
      new Linear.Row(weights.require(prefix + "v_proj.weight")),
      new Linear.Row(weights.require(prefix + O_PROJ_WEIGHT)),
      new RMSNorm(weights.require(prefix + Q_NORM_WEIGHT), config.rmsNormEps(), true),
      new RMSNorm(weights.require(prefix + K_NORM_WEIGHT), config.rmsNormEps(), true),
      rotary,
      new Attention(numHeads, headDim, config.attentionScale(), numKvHeads, layerIndex),
      numHeads,
      numKvHeads,
      headDim);
  }

  private static Config.Qwen35Text requireSpec(final Config.HfConfig config) {
    Config.Qwen35Text spec = config.qwen35();
    if (spec == null) {
      throw new IllegalArgumentException("qwen3_5 config is missing linear-attention fields");
    }
    if (config.layerTypes().size() != config.numHiddenLayers()) {
      throw new IllegalArgumentException(
        "qwen3_5 layer_types length " + config.layerTypes().size()
          + " != num_hidden_layers " + config.numHiddenLayers());
    }
    return spec;
  }

  private static void requireSilu(final Config.HfConfig config) {
    if (!"silu".equals(config.effectiveActivation())) {
      throw new IllegalArgumentException(
        "qwen3_5 architecture expects silu, got " + config.effectiveActivation());
    }
  }

  @Override
  public String architectureName() {
    return ARCH_QWEN3_5;
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

  record Qwen35Attention(
    Linear.Row qProj,
    Linear.Row kProj,
    Linear.Row vProj,
    Linear.Row oProj,
    RMSNorm qNorm,
    RMSNorm kNorm,
    RotaryEmbedding rotary,
    Attention attn,
    int numHeads,
    int numKvHeads,
    int headDim
  ) {
    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      int tokens = hiddenStates.size(0);
      int width = this.numHeads * this.headDim;
      Tensor projected = this.qProj.forward(hiddenStates, context);
      float[] query = new float[tokens * width];
      float[] gate = new float[tokens * width];
      this.splitQueryGate(projected, tokens, width, query, gate);

      Tensor q = this.qNorm.forward(Tensor.of(query, tokens, this.numHeads, this.headDim));
      Tensor k = this.kNorm.forward(this.kProj.forward(hiddenStates, context)
        .reshape(tokens, this.numKvHeads, this.headDim));
      Tensor v = this.vProj.forward(hiddenStates, context)
        .reshape(tokens, this.numKvHeads, this.headDim);
      Tensor[] rotated = this.rotary.forward(positions, q, k, context);
      Tensor attended = this.attn.forward(rotated[0], rotated[1], v, context);
      this.applyGate(attended, gate);
      return this.oProj.forward(attended.reshape(tokens, width), context);
    }

    private void splitQueryGate(
      final Tensor projected,
      final int tokens,
      final int width,
      final float[] query,
      final float[] gate
    ) {
      float[] data = projected.data();
      int off = projected.offset();
      int row = width * 2;
      for (int t = 0; t < tokens; t++) {
        System.arraycopy(data, off + t * row, query, t * width, width);
        System.arraycopy(data, off + t * row + width, gate, t * width, width);
      }
    }

    private void applyGate(final Tensor attended, final float[] gate) {
      float[] data = attended.data();
      int off = attended.offset();
      for (int i = 0; i < gate.length; i++) {
        float g = gate[i];
        data[off + i] *= (float) (1.0 / (1.0 + Math.exp(-g)));
      }
    }
  }

  record Qwen35Mlp(Linear.Merged gateUpProj, Linear.Row downProj) {
    Tensor forward(final Tensor x, final Context context) {
      return this.downProj.forward(Ops.siluAndMul(this.gateUpProj.forward(x, context)), context);
    }
  }

  record Qwen35DecoderLayer(
    GatedDeltaNet linearAttn,
    Qwen35Attention attention,
    Qwen35Mlp mlp,
    RMSNorm inputLayernorm,
    RMSNorm postAttentionLayernorm
  ) {
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
        Tensor[] normed = this.inputLayernorm.forward(hiddenStates, residual);
        hidden = normed[0];
        resid = normed[1];
      }
      hidden = this.linearAttn != null
        ? this.linearAttn.forward(hidden, context)
        : this.attention.forward(positions, hidden, context);
      Tensor[] normed = this.postAttentionLayernorm.forward(hidden, resid);
      return new Tensor[] {this.mlp.forward(normed[0], context), normed[1]};
    }
  }

  record Qwen35Model(
    VocabParallelEmbedding embedTokens,
    List<Qwen35DecoderLayer> layers,
    RMSNorm norm
  ) {
    Qwen35Model(final Config.HfConfig config, final WeightBag weights) {
      this(assemble(config, weights));
    }

    private Qwen35Model(final Qwen35Model assembled) {
      this(assembled.embedTokens, assembled.layers, assembled.norm);
    }

    private static Qwen35Model assemble(final Config.HfConfig config, final WeightBag weights) {
      Config.Qwen35Text spec = requireSpec(config);
      requireSilu(config);
      int rotaryDim = (int) (config.headDim() * spec.partialRotaryFactor());
      if (rotaryDim <= 0 || rotaryDim > config.headDim() || (rotaryDim & 1) != 0) {
        throw new IllegalArgumentException(
          "qwen3_5 partial RoPE width must be a positive even head fraction, got " + rotaryDim);
      }
      Tables ropes = new Tables();
      RotaryEmbedding rotary = ropes.leading(
        config.headDim(), rotaryDim, config.maxPositionEmbeddings(), config.ropeTheta());
      List<Qwen35DecoderLayer> built = IntStream.range(0, config.numHiddenLayers())
        .mapToObj(i -> layerAt(config, spec, weights, i, rotary))
        .toList();
      return new Qwen35Model(
        new VocabParallelEmbedding(weights.require(EMBED_TOKENS)),
        built,
        new RMSNorm(weights.require(MODEL_NORM), config.rmsNormEps(), true));
    }

    Tensor forward(final Tensor inputIds, final Tensor positions, final Context context) {
      Tensor hiddenStates = this.embedTokens.forward(inputIds, context);
      Tensor residual = null;
      for (Qwen35DecoderLayer layer : this.layers) {
        Tensor[] out = layer.forward(positions, hiddenStates, residual, context);
        hiddenStates = out[0];
        residual = out[1];
      }
      return this.norm.forward(hiddenStates, residual)[0];
    }
  }
}
