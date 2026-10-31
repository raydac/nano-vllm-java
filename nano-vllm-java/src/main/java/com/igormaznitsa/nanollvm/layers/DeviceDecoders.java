package com.igormaznitsa.nanollvm.layers;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Norms.RMSNorm;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Optional device implementation of one decoder layer. TornadoVM registers itself when that
 * backend is selected; every other backend leaves the step unset.
 */
public final class DeviceDecoders {

  private static volatile Step decoder;

  private DeviceDecoders() {
  }

  public static void register(final Step step) {
    decoder = step;
  }

  public static Step decoder() {
    FloatKernels.loadBackend();
    return decoder;
  }

  public static LayerBuilder layer() {
    return new LayerBuilder();
  }

  public enum Shape {
    HIGHWAY,
    NORM_BRANCH,
    PLAIN_BRANCH
  }

  public enum Activation {
    SILU,
    GELU
  }

  @FunctionalInterface
  public interface Step {
    Tensor[] forward(Layer layer);
  }

  public record Gain(float[] data, int offset, int width, boolean onePlus) {
    public static Gain from(final RMSNorm norm) {
      Tensor weight = norm.weight();
      if (weight == null) {
        return new Gain(null, 0, 0, false);
      }
      return new Gain(weight.data(), weight.offset(), weight.numel(), norm.isOnePlusWeight());
    }
  }

  public record Matrix(
    float[] data,
    int offset,
    int rows,
    int columns,
    float[] bias,
    float inputScale,
    float outputScale
  ) {
    public static Matrix of(final Tensor weight) {
      return new Matrix(
        weight.data(), weight.offset(), weight.size(0), weight.size(1), null, 0f, 0f);
    }

    public static Matrix of(final Linear linear) {
      Tensor weight = linear.residentMatrix();
      Tensor bias = linear.bias();
      float[] biasData = null;
      if (bias != null) {
        biasData = bias.offset() == 0 ? bias.data() : bias.toFloatArray();
      }
      return new Matrix(
        weight.data(), weight.offset(), weight.size(0), weight.size(1), biasData,
        linear.getInputActivationScale(), linear.getOutputActivationScale());
    }
  }

  public record Layer(
    Shape shape,
    Activation activation,
    boolean attention,
    boolean queryGate,
    boolean writeKv,
    Tensor positions,
    Tensor hidden,
    Tensor residual,
    Tensor perLayerInput,
    Context context,
    Matrix qkv,
    Matrix query,
    Matrix key,
    Matrix value,
    Gain queryNorm,
    Gain keyNorm,
    Gain valueNorm,
    RotaryEmbedding rope,
    Matrix output,
    Gain inputNorm,
    Gain postAttentionNorm,
    Gain preMlpNorm,
    Gain postMlpNorm,
    Matrix gateUp,
    Matrix gate,
    Matrix up,
    Matrix down,
    Matrix perLayerGate,
    Matrix perLayerProjection,
    Gain postPerLayerNorm,
    float layerScale,
    float epsilon,
    int numHeads,
    int numKvHeads,
    int headDim,
    float scoreScale,
    int window,
    Tensor keyCache,
    Tensor valueCache
  ) {
  }

  public static final class LayerBuilder {
    private Shape shape;
    private Activation activation = Activation.SILU;
    private boolean attention = true;
    private boolean queryGate;
    private boolean writeKv = true;
    private Tensor positions;
    private Tensor hidden;
    private Tensor residual;
    private Tensor perLayerInput;
    private Context context;
    private Matrix qkv;
    private Matrix query;
    private Matrix key;
    private Matrix value;
    private Gain queryNorm;
    private Gain keyNorm;
    private Gain valueNorm;
    private RotaryEmbedding rope;
    private Matrix output;
    private Gain inputNorm;
    private Gain postAttentionNorm;
    private Gain preMlpNorm;
    private Gain postMlpNorm;
    private Matrix gateUp;
    private Matrix gate;
    private Matrix up;
    private Matrix down;
    private Matrix perLayerGate;
    private Matrix perLayerProjection;
    private Gain postPerLayerNorm;
    private float layerScale = 1f;
    private float epsilon;
    private int numHeads;
    private int numKvHeads;
    private int headDim;
    private float scoreScale;
    private int window;
    private Tensor keyCache;
    private Tensor valueCache;

    public LayerBuilder highway() {
      this.shape = Shape.HIGHWAY;
      return this;
    }

    public LayerBuilder normBranch() {
      this.shape = Shape.NORM_BRANCH;
      return this;
    }

    public LayerBuilder plainBranch() {
      this.shape = Shape.PLAIN_BRANCH;
      return this;
    }

    public LayerBuilder silu() {
      this.activation = Activation.SILU;
      return this;
    }

    public LayerBuilder gelu() {
      this.activation = Activation.GELU;
      return this;
    }

    public LayerBuilder mlpOnly() {
      this.attention = false;
      this.writeKv = false;
      return this;
    }

    public LayerBuilder queryGate() {
      this.queryGate = true;
      return this;
    }

    public LayerBuilder writeKv(final boolean writeKv) {
      this.writeKv = writeKv;
      return this;
    }

    public LayerBuilder positions(final Tensor positions) {
      this.positions = positions;
      return this;
    }

    public LayerBuilder hidden(final Tensor hidden) {
      this.hidden = hidden;
      return this;
    }

    public LayerBuilder residual(final Tensor residual) {
      this.residual = residual;
      return this;
    }

    public LayerBuilder perLayerInput(final Tensor perLayerInput) {
      this.perLayerInput = perLayerInput;
      return this;
    }

    public LayerBuilder context(final Context context) {
      this.context = context;
      return this;
    }

    public LayerBuilder qkv(final Linear linear) {
      this.qkv = Matrix.of(linear);
      return this;
    }

    public LayerBuilder qkv(final Tensor weight) {
      this.qkv = Matrix.of(weight);
      return this;
    }

    public LayerBuilder query(final Linear linear) {
      this.query = Matrix.of(linear);
      return this;
    }

    public LayerBuilder query(final Tensor weight) {
      this.query = Matrix.of(weight);
      return this;
    }

    public LayerBuilder key(final Linear linear) {
      this.key = Matrix.of(linear);
      return this;
    }

    public LayerBuilder key(final Tensor weight) {
      this.key = Matrix.of(weight);
      return this;
    }

    public LayerBuilder value(final Linear linear) {
      this.value = Matrix.of(linear);
      return this;
    }

    public LayerBuilder value(final Tensor weight) {
      this.value = Matrix.of(weight);
      return this;
    }

    public LayerBuilder output(final Linear linear) {
      this.output = Matrix.of(linear);
      return this;
    }

    public LayerBuilder output(final Tensor weight) {
      this.output = Matrix.of(weight);
      return this;
    }

    public LayerBuilder gateUp(final Linear linear) {
      this.gateUp = Matrix.of(linear);
      return this;
    }

    public LayerBuilder gateUp(final Tensor weight) {
      this.gateUp = Matrix.of(weight);
      return this;
    }

    public LayerBuilder gate(final Linear linear) {
      this.gate = Matrix.of(linear);
      return this;
    }

    public LayerBuilder gate(final Tensor weight) {
      this.gate = Matrix.of(weight);
      return this;
    }

    public LayerBuilder up(final Linear linear) {
      this.up = Matrix.of(linear);
      return this;
    }

    public LayerBuilder up(final Tensor weight) {
      this.up = Matrix.of(weight);
      return this;
    }

    public LayerBuilder down(final Linear linear) {
      this.down = Matrix.of(linear);
      return this;
    }

    public LayerBuilder down(final Tensor weight) {
      this.down = Matrix.of(weight);
      return this;
    }

    public LayerBuilder perLayerGate(final Linear linear) {
      this.perLayerGate = Matrix.of(linear);
      return this;
    }

    public LayerBuilder perLayerProjection(final Linear linear) {
      this.perLayerProjection = Matrix.of(linear);
      return this;
    }

    public LayerBuilder inputNorm(final RMSNorm norm) {
      this.inputNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder inputNorm(final Tensor weight, final boolean onePlus) {
      this.inputNorm = new Gain(weight.data(), weight.offset(), weight.numel(), onePlus);
      return this;
    }

    public LayerBuilder postAttentionNorm(final RMSNorm norm) {
      this.postAttentionNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder postAttentionNorm(final Tensor weight, final boolean onePlus) {
      this.postAttentionNorm = new Gain(weight.data(), weight.offset(), weight.numel(), onePlus);
      return this;
    }

    public LayerBuilder preMlpNorm(final RMSNorm norm) {
      this.preMlpNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder preMlpNorm(final Tensor weight, final boolean onePlus) {
      this.preMlpNorm = new Gain(weight.data(), weight.offset(), weight.numel(), onePlus);
      return this;
    }

    public LayerBuilder postMlpNorm(final RMSNorm norm) {
      this.postMlpNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder postMlpNorm(final Tensor weight, final boolean onePlus) {
      this.postMlpNorm = new Gain(weight.data(), weight.offset(), weight.numel(), onePlus);
      return this;
    }

    public LayerBuilder postPerLayerNorm(final RMSNorm norm) {
      this.postPerLayerNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder queryNorm(final RMSNorm norm) {
      this.queryNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder queryNorm(final Tensor weight, final boolean onePlus) {
      this.queryNorm = new Gain(weight.data(), weight.offset(), weight.numel(), onePlus);
      return this;
    }

    public LayerBuilder keyNorm(final RMSNorm norm) {
      this.keyNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder keyNorm(final Tensor weight, final boolean onePlus) {
      this.keyNorm = new Gain(weight.data(), weight.offset(), weight.numel(), onePlus);
      return this;
    }

    public LayerBuilder valueNorm(final RMSNorm norm) {
      this.valueNorm = Gain.from(norm);
      return this;
    }

    public LayerBuilder rope(final RotaryEmbedding rope) {
      this.rope = rope;
      return this;
    }

    public LayerBuilder heads(final int numHeads, final int numKvHeads, final int headDim) {
      this.numHeads = numHeads;
      this.numKvHeads = numKvHeads;
      this.headDim = headDim;
      return this;
    }

    public LayerBuilder epsilon(final float epsilon) {
      this.epsilon = epsilon;
      return this;
    }

    public LayerBuilder scoreScale(final float scoreScale) {
      this.scoreScale = scoreScale;
      return this;
    }

    public LayerBuilder window(final int window) {
      this.window = window;
      return this;
    }

    public LayerBuilder layerScale(final float layerScale) {
      this.layerScale = layerScale;
      return this;
    }

    public LayerBuilder caches(final Tensor keyCache, final Tensor valueCache) {
      this.keyCache = keyCache;
      this.valueCache = valueCache;
      return this;
    }

    public Layer build() {
      if (this.shape == null || this.hidden == null) {
        throw new IllegalStateException("device layer needs a shape and a hidden tensor");
      }
      return new Layer(
        this.shape, this.activation, this.attention, this.queryGate, this.writeKv,
        this.positions, this.hidden, this.residual, this.perLayerInput, this.context,
        this.qkv, this.query, this.key, this.value,
        this.queryNorm, this.keyNorm, this.valueNorm, this.rope, this.output,
        this.inputNorm, this.postAttentionNorm, this.preMlpNorm, this.postMlpNorm,
        this.gateUp, this.gate, this.up, this.down,
        this.perLayerGate, this.perLayerProjection, this.postPerLayerNorm,
        this.layerScale, this.epsilon, this.numHeads, this.numKvHeads, this.headDim,
        this.scoreScale, this.window, this.keyCache, this.valueCache);
    }
  }
}
