package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3_EMBEDDING;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.DOWN_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBEDDING_DOWN_PROJ;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBEDDING_UP_PROJ;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.EMBED_TOKENS;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.GATE_UP_PROJ_WEIGHT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.INPUT_LAYERNORM;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.K_NORM_WEIGHT;
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
import static com.igormaznitsa.nanollvm.tensor.ActivationWorkspace.bind;
import static com.igormaznitsa.nanollvm.tensor.ActivationWorkspace.endLayer;
import static com.igormaznitsa.nanollvm.tensor.ActivationWorkspace.releaseUnless;
import static com.igormaznitsa.nanollvm.tensor.ActivationWorkspace.unbind;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.GroupedEncoderAttention;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding.Tables;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.ActivationWorkspace;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Bidirectional Gemma 3 text encoder plus the two dense maps that turn a mean-pooled sequence
 * into one L2-normalized vector. Causal Gemma 3 chat stays {@link Gemma3ForCausalLM}.
 *
 * @since 1.5.0
 */
public final class Gemma3ForEmbedding implements EmbeddingEncoder {

  private final Config.HfConfig config;
  private final VocabParallelEmbedding embedTokens;
  private final List<DecoderLayer> layers;
  private final RMSNorm norm;
  private final float embedScale;
  private final Linear.Row upProj;
  private final Linear.Row downProj;
  private final int embeddingDim;

  public Gemma3ForEmbedding(final Config.HfConfig config, final WeightBag weights) {
    requireNonNull(config, "config");
    requireNonNull(weights, "weights");
    if (!config.bidirectionalAttention()) {
      throw new IllegalArgumentException(
        "gemma3 embedding encoder requires use_bidirectional_attention");
    }
    Tables ropes = new Tables();
    Tensor down = weights.require(EMBEDDING_DOWN_PROJ);
    this.config = config;
    this.embedTokens = new VocabParallelEmbedding(weights.require(EMBED_TOKENS));
    this.layers = IntStream.range(0, config.numHiddenLayers())
      .mapToObj(i -> new DecoderLayer(config, weights, i, ropes))
      .toList();
    this.norm = new RMSNorm(weights.require(MODEL_NORM), config.rmsNormEps(), true);
    this.embedScale = (float) Math.sqrt(config.hiddenSize());
    this.upProj = new Linear.Row(weights.require(EMBEDDING_UP_PROJ));
    this.downProj = new Linear.Row(down);
    this.embeddingDim = down.size(0);
  }

  private static Tensor ids(final int[] tokenIds) {
    float[] data = new float[tokenIds.length];
    for (int i = 0; i < tokenIds.length; i++) {
      data[i] = tokenIds[i];
    }
    return Tensor.of(data, tokenIds.length);
  }

  private static Tensor positions(final int length) {
    float[] data = new float[length];
    for (int i = 0; i < length; i++) {
      data[i] = i;
    }
    return Tensor.of(data, length);
  }

  private static float[] meanPool(final Tensor hidden) {
    int tokens = hidden.size(0);
    int width = hidden.size(1);
    float[] data = hidden.data();
    int off = hidden.offset();
    double[] sum = new double[width];
    for (int token = 0; token < tokens; token++) {
      int row = off + token * width;
      for (int i = 0; i < width; i++) {
        sum[i] += data[row + i];
      }
    }
    float[] mean = new float[width];
    double inv = 1.0 / tokens;
    for (int i = 0; i < width; i++) {
      mean[i] = (float) (sum[i] * inv);
    }
    return mean;
  }

  private static float[] l2Normalize(final float[] vector) {
    double sumSq = 0.0;
    for (float value : vector) {
      sumSq += (double) value * value;
    }
    float inv = sumSq > 0.0 ? (float) (1.0 / Math.sqrt(sumSq)) : 0f;
    float[] out = new float[vector.length];
    for (int i = 0; i < vector.length; i++) {
      out[i] = vector[i] * inv;
    }
    return out;
  }

  @Override
  public String architectureName() {
    return ARCH_GEMMA3_EMBEDDING;
  }

  @Override
  public int embeddingDim() {
    return this.embeddingDim;
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
    Context context = new Context();
    context.bindMatmul(runtime == null ? MatmulRuntime.sequential() : runtime);
    bind(new ActivationWorkspace());
    try {
      Tensor normed = this.forwardTokens(tokenIds, context);
      Tensor projected = this.downProj.forward(
        this.upProj.forward(Tensor.of(meanPool(normed), 1, this.config.hiddenSize()), context),
        context);
      return l2Normalize(projected.toFloatArray());
    } finally {
      unbind();
    }
  }

  private Tensor forwardTokens(final int[] tokenIds, final Context context) {
    beginLayer();
    Tensor hidden = Ops.scale(this.embedTokens.forward(ids(tokenIds), context), this.embedScale);
    Tensor positions = positions(tokenIds.length);
    for (DecoderLayer decoderLayer : this.layers) {
      Tensor next = decoderLayer.forward(positions, hidden, context);
      endLayer(next);
      releaseUnless(hidden, next);
      hidden = next;
      beginLayer();
    }
    Tensor normed = this.norm.forward(hidden);
    endLayer(normed);
    releaseUnless(hidden, normed);
    return normed;
  }

  private record DecoderLayer(
    AttentionBlock selfAttn,
    MlpBlock mlp,
    RMSNorm inputLayernorm,
    RMSNorm postAttentionLayernorm,
    RMSNorm preFeedforwardLayernorm,
    RMSNorm postFeedforwardLayernorm
  ) {
    DecoderLayer(Config.HfConfig config, WeightBag weights, int layerIndex, Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private DecoderLayer(final DecoderLayer assembled) {
      this(
        assembled.selfAttn, assembled.mlp,
        assembled.inputLayernorm, assembled.postAttentionLayernorm,
        assembled.preFeedforwardLayernorm, assembled.postFeedforwardLayernorm);
    }

    private static DecoderLayer assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex,
      final Tables ropes
    ) {
      String prefix = layer(layerIndex);
      return new DecoderLayer(
        new AttentionBlock(config, weights, layerIndex, ropes),
        new MlpBlock(config, weights, layerIndex),
        new RMSNorm(weights.require(prefix + INPUT_LAYERNORM), config.rmsNormEps(), true),
        new RMSNorm(weights.require(prefix + POST_ATTENTION_LAYERNORM), config.rmsNormEps(), true),
        new RMSNorm(weights.require(prefix + PRE_FEEDFORWARD_LAYERNORM), config.rmsNormEps(), true),
        new RMSNorm(weights.require(prefix + POST_FEEDFORWARD_LAYERNORM), config.rmsNormEps(),
          true));
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor attended = this.selfAttn.forward(
        positions, this.inputLayernorm.forward(hiddenStates), context);
      Tensor mid = Ops.add(hiddenStates, this.postAttentionLayernorm.forward(attended));
      Tensor fed = this.mlp.forward(this.preFeedforwardLayernorm.forward(mid), context);
      return Ops.add(mid, this.postFeedforwardLayernorm.forward(fed));
    }
  }

  private record AttentionBlock(
    Linear.Qkv qkvProj,
    Linear.Row oProj,
    RotaryEmbedding rotaryEmb,
    GroupedEncoderAttention attn,
    RMSNorm qNorm,
    RMSNorm kNorm,
    int numHeads,
    int numKvHeads,
    int headDim,
    int qSize,
    int kvSize
  ) {
    AttentionBlock(Config.HfConfig config, WeightBag weights, int layerIndex, Tables ropes) {
      this(assemble(config, weights, layerIndex, ropes));
    }

    private AttentionBlock(final AttentionBlock assembled) {
      this(
        assembled.qkvProj, assembled.oProj, assembled.rotaryEmb, assembled.attn,
        assembled.qNorm, assembled.kNorm,
        assembled.numHeads, assembled.numKvHeads, assembled.headDim,
        assembled.qSize, assembled.kvSize);
    }

    private static AttentionBlock assemble(
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
      int window = sliding ? config.bidirectionalWindow() : 0;
      String prefix = selfAttn(layerIndex);
      return new AttentionBlock(
        new Linear.Qkv(weights.require(prefix + QKV_PROJ_WEIGHT)),
        new Linear.Row(weights.require(prefix + O_PROJ_WEIGHT)),
        ropes.get(headDim, headDim, config.maxPositionEmbeddings(), ropeBase),
        new GroupedEncoderAttention(
          numHeads, numKvHeads, headDim, config.attentionScale(), window),
        new RMSNorm(weights.require(prefix + Q_NORM_WEIGHT), config.rmsNormEps(), true),
        new RMSNorm(weights.require(prefix + K_NORM_WEIGHT), config.rmsNormEps(), true),
        numHeads,
        numKvHeads,
        headDim,
        numHeads * headDim,
        numKvHeads * headDim);
    }

    Tensor forward(final Tensor positions, final Tensor hiddenStates, final Context context) {
      Tensor qkv = this.qkvProj.forward(hiddenStates, context);
      Tensor[] parts = Ops.splitLast(qkv, this.qSize, this.kvSize, this.kvSize);
      Tensor query =
        this.qNorm.forward(parts[0].reshape(parts[0].size(0), this.numHeads, this.headDim));
      Tensor key =
        this.kNorm.forward(parts[1].reshape(parts[1].size(0), this.numKvHeads, this.headDim));
      Tensor value = parts[2].reshape(parts[2].size(0), this.numKvHeads, this.headDim);
      Tensor[] rotated = this.rotaryEmb.forward(positions, query, key, context);
      Tensor attended = this.attn.forward(rotated[0], rotated[1], value, context);
      return this.oProj.forward(
        attended.reshape(attended.size(0), this.numHeads * this.headDim), context);
    }
  }

  private record MlpBlock(Linear.Merged gateUpProj, Linear.Row downProj) {
    MlpBlock(Config.HfConfig config, WeightBag weights, int layerIndex) {
      this(assemble(config, weights, layerIndex));
    }

    private MlpBlock(final MlpBlock assembled) {
      this(assembled.gateUpProj, assembled.downProj);
    }

    private static MlpBlock assemble(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layerIndex
    ) {
      String act = config.effectiveActivation().toLowerCase(ROOT);
      if (!act.contains("gelu")) {
        throw new IllegalArgumentException(
          "gemma3 embedding encoder expects gelu_pytorch_tanh, got " + act);
      }
      String prefix = mlp(layerIndex);
      return new MlpBlock(
        new Linear.Merged(weights.require(prefix + GATE_UP_PROJ_WEIGHT)),
        new Linear.Row(weights.require(prefix + DOWN_PROJ_WEIGHT)));
    }

    Tensor forward(final Tensor hidden, final Context context) {
      return this.downProj.forward(
        Ops.geluPytorchTanhAndMul(this.gateUpProj.forward(hidden, context)), context);
    }
  }
}
