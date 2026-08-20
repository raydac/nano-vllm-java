package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.models.internal.GemmaQat;
import com.igormaznitsa.nanollvm.models.internal.GemmaQatWeight;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.LinearKernel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.kernels.PackedLinearKernel;

/**
 * Immutable affine transform {@code y = x Wᵀ + b}. A {@link LinearKernel} is bound at construction
 * (dense F32, packed GGUF with a GGML-type-fixed dequant, or Gemma QAT).
 *
 * <p>Weight layout is {@code [out, in]} (Hugging Face / this port). The last axis of {@code x} must
 * be {@code in}; leading axes are treated as rows. Nested types keep vLLM names
 * ({@link Column} / {@link Row} / {@link Merged} / {@link Qkv}) so architecture graphs can tell
 * fused projections from output projections; this port is single-device (no tensor parallel).
 *
 * <p>Gemma QAT layers may apply {@link GemmaQat#applySrq(float, float) SRQ} to activations before
 * and after the matmul when the checkpoint stores non-zero scales.
 */
public class Linear {

  protected final Tensor weight;
  protected final PackedWeight packedWeight;
  protected final Tensor bias;
  private final LinearKernel kernel;
  private final float inputActivationScale;
  private final float outputActivationScale;

  /**
   * Dense float32 weight {@code [out, in]} with optional bias of length {@code out}.
   *
   * @param weight matrix {@code [out, in]}
   * @param bias   length {@code out}, or {@code null}
   */
  public Linear(final Tensor weight, final Tensor bias) {
    this(LinearKernel.of(requireNonNull(weight, "weight")), bias, 0f, 0f, weight, null);
  }

  /**
   * Packed GGUF weight. Float32 packs are materialized and the packed bytes released; other GGML
   * types stay packed and dequant on each forward.
   *
   * @param weight packed matrix {@code [out, in]}
   * @param bias   length {@code out}, or {@code null}
   */
  public Linear(final PackedWeight weight, final Tensor bias) {
    this(denseOrPacked(requireNonNull(weight, "weight"), bias));
  }

  private Linear(final Linear assembled) {
    this(
      assembled.kernel, assembled.bias, assembled.inputActivationScale,
      assembled.outputActivationScale, assembled.weight, assembled.packedWeight);
  }

  private static Linear denseOrPacked(final PackedWeight weight, final Tensor bias) {
    if (weight.isFloat32()) {
      Tensor dense = weight.materialize();
      weight.releasePackedBytes();
      return new Linear(LinearKernel.of(dense), bias, 0f, 0f, dense, null);
    }
    return new Linear(LinearKernel.of(weight), bias, 0f, 0f, null, weight);
  }

  /**
   * Packed GGUF weight with no bias.
   *
   * @param weight packed matrix {@code [out, in]}
   */
  public Linear(final PackedWeight weight) {
    this(weight, null);
  }

  /**
   * Gemma QAT packed weight (int2/4/8 + scales). Bias is unused; SRQ scales come from the weight.
   *
   * @param weight QAT matrix
   * @since 1.1.0
   */
  public Linear(final GemmaQatWeight weight) {
    this(
      PackedLinearKernel.of(
        requireNonNull(weight, "weight").cols(),
        weight.rows(),
        weight.name(),
        weight::dequantizeRow),
      null,
      weight.inputActivationScale(),
      weight.outputActivationScale(),
      null,
      null);
  }

  /**
   * Already-bound kernel (tests and custom graphs). No dense/packed table is stored.
   *
   * @param kernel affine kernel
   * @param bias   length {@code outFeatures()}, or {@code null}
   */
  public Linear(final LinearKernel kernel, final Tensor bias) {
    this(kernel, bias, 0f, 0f, null, null);
  }

  private Linear(
    final LinearKernel kernel,
    final Tensor bias,
    final float inputActivationScale,
    final float outputActivationScale,
    final Tensor weight,
    final PackedWeight packedWeight
  ) {
    this.kernel = requireNonNull(kernel, "kernel");
    this.bias = bias;
    this.inputActivationScale = inputActivationScale;
    this.outputActivationScale = outputActivationScale;
    this.weight = weight;
    this.packedWeight = packedWeight;
  }

  /**
   * Dense float32 table. Packed layers materialize a copy (expensive); QAT-only kernels have none.
   *
   * @return weight {@code [out, in]}
   * @throws IllegalStateException if this layer has no dense or packed table
   */
  public Tensor weight() {
    if (this.weight != null) {
      return this.weight;
    }
    if (this.packedWeight != null) {
      return this.packedWeight.materialize();
    }
    throw new IllegalStateException("linear has no dense weight table");
  }

  /**
   * Packed GGUF payload, or {@code null} for dense / QAT-kernel layers.
   *
   * @return packed weight, or {@code null}
   */
  public PackedWeight packedWeight() {
    return this.packedWeight;
  }

  /**
   * {@code true} when this layer still holds a GGUF packed payload (not yet a dense table).
   *
   * @return whether {@link #packedWeight()} is non-null
   */
  public boolean isPacked() {
    return this.packedWeight != null;
  }

  /**
   * Kernel bound at construction (dense, packed, or QAT).
   *
   * @return affine kernel
   */
  public LinearKernel kernel() {
    return this.kernel;
  }

  /**
   * Additive bias of length {@code out}, or {@code null}.
   *
   * @return bias tensor, or {@code null}
   */
  public Tensor bias() {
    return this.bias;
  }

  /**
   * {@code y = x Wᵀ (+ b)} along the last axis of {@code x}. Uses {@code context}'s
   * {@link MatmulRuntime}, or {@link MatmulRuntime#sequential()} when unbound / {@code null}.
   *
   * @param x       activations; last dim must equal {@code in}
   * @param context step context (matmul pool); {@code null} is sequential
   * @return tensor with last dim {@code out}, other axes preserved
   * @throws NullPointerException     if {@code x} is {@code null}
   * @throws IllegalArgumentException if {@code x} width is not a multiple of {@code in}
   */
  public Tensor forward(final Tensor x, final Context context) {
    requireNonNull(x, "x");
    MatmulRuntime matmul = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    return this.apply(x, matmul);
  }

  private Tensor apply(final Tensor x, final MatmulRuntime matmul) {
    int in = this.kernel.inFeatures();
    int out = this.kernel.outFeatures();
    int[] xs = x.shape();
    int rows = x.numel() / in;
    if (x.numel() % in != 0) {
      throw new IllegalArgumentException("x last dim mismatch");
    }
    Tensor y = Tensor.zeros(rows, out);
    float[] biasData = null;
    if (this.bias != null) {
      biasData = this.bias.offset() == 0 ? this.bias.data() : this.bias.toFloatArray();
    }
    float[] xData = x.data();
    int xOff = x.offset();
    if (this.inputActivationScale != 0f) {
      Tensor quantized = this.scaleActivations(x, this.inputActivationScale, matmul);
      xData = quantized.data();
      xOff = quantized.offset();
    }
    this.kernel.apply(xData, xOff, biasData, y.data(), 0, rows, matmul);
    if (this.outputActivationScale != 0f) {
      y = this.scaleActivations(y, this.outputActivationScale, matmul);
    }
    if (xs.length == 1) {
      return y.reshape(out);
    }
    if (xs.length == 2) {
      return y.reshape(xs[0], out);
    }
    int[] newShape = xs.clone();
    newShape[newShape.length - 1] = out;
    return y.reshape(newShape);
  }

  /**
   * Gemma QAT SRQ: round {@code value / scale} to int8 then rescale. No-op when {@code scale == 0}.
   */
  private Tensor scaleActivations(
    final Tensor values,
    final float scale,
    final MatmulRuntime matmul
  ) {
    Tensor out = Tensor.zeros(values.shape());
    float[] source = values.data();
    float[] dest = out.data();
    int off = values.offset();
    int n = values.numel();
    if (n < 4096) {
      this.scaleRange(source, off, dest, scale, 0, n);
      return out;
    }
    matmul.parallelRanges(n, (start, end) -> this.scaleRange(source, off, dest, scale, start, end));
    return out;
  }

  private void scaleRange(
    final float[] source,
    final int sourceOffset,
    final float[] dest,
    final float scale,
    final int start,
    final int end
  ) {
    for (int i = start; i < end; i++) {
      dest[i] = GemmaQat.applySrq(source[sourceOffset + i], scale);
    }
  }

  /**
   * Column-style projection (output dim is the packed axis): Q/K/V, gate+up, and similar
   * expansions. Marker type for architecture graphs; math matches {@link Linear}.
   */
  public static class Column extends Linear {

    /**
     * Dense float32 column projection.
     *
     * @param weight matrix {@code [out, in]}
     * @param bias   length {@code out}, or {@code null}
     */
    public Column(final Tensor weight, final Tensor bias) {
      super(weight, bias);
    }

    /**
     * Packed GGUF column projection.
     *
     * @param weight packed matrix {@code [out, in]}
     * @param bias   length {@code out}, or {@code null}
     */
    public Column(final PackedWeight weight, final Tensor bias) {
      super(weight, bias);
    }
  }

  /**
   * Row-style projection (input dim is the packed axis): {@code o_proj}, {@code down_proj},
   * and similar reductions. Marker type for architecture graphs; math matches {@link Linear}.
   */
  public static final class Row extends Linear {

    /**
     * Dense float32 row projection with no bias.
     *
     * @param weight matrix {@code [out, in]}
     */
    public Row(final Tensor weight) {
      super(weight, null);
    }

    /**
     * Dense float32 row projection.
     *
     * @param weight matrix {@code [out, in]}
     * @param bias   length {@code out}, or {@code null}
     */
    public Row(final Tensor weight, final Tensor bias) {
      super(weight, bias);
    }

    /**
     * Packed GGUF row projection with no bias.
     *
     * @param weight packed matrix {@code [out, in]}
     */
    public Row(final PackedWeight weight) {
      super(weight, null);
    }

    /**
     * Packed GGUF row projection.
     *
     * @param weight packed matrix {@code [out, in]}
     * @param bias   length {@code out}, or {@code null}
     */
    public Row(final PackedWeight weight, final Tensor bias) {
      super(weight, bias);
    }

    /**
     * Gemma QAT row projection.
     *
     * @param weight QAT matrix
     * @since 1.1.0
     */
    public Row(final GemmaQatWeight weight) {
      super(weight);
    }
  }

  /**
   * Packed gate+up projection; {@code weight} shape {@code [2*intermediate, hidden]}.
   * The MLP splits the last dim into gate and up halves after {@link Linear#forward}.
   */
  public static final class Merged extends Column {

    /**
     * Dense fused gate+up.
     *
     * @param weight matrix {@code [2*intermediate, hidden]}
     */
    public Merged(final Tensor weight) {
      super(weight, null);
    }

    /**
     * Packed fused gate+up.
     *
     * @param weight packed matrix {@code [2*intermediate, hidden]}
     */
    public Merged(final PackedWeight weight) {
      super(weight, null);
    }
  }

  /**
   * Packed Q/K/V projection; {@code weight} shape {@code [(nH+2nKV)*d, hidden]}.
   * The attention block splits the last dim into Q, K, and V after {@link Linear#forward}.
   */
  public static final class Qkv extends Column {

    /**
     * Dense fused QKV with optional bias.
     *
     * @param weight matrix {@code [(nH+2nKV)*d, hidden]}
     * @param bias   length {@code out}, or {@code null}
     */
    public Qkv(final Tensor weight, final Tensor bias) {
      super(weight, bias);
    }

    /**
     * Dense fused QKV with no bias.
     *
     * @param weight matrix {@code [(nH+2nKV)*d, hidden]}
     */
    public Qkv(final Tensor weight) {
      this(weight, null);
    }

    /**
     * Packed fused QKV with no bias.
     *
     * @param weight packed matrix {@code [(nH+2nKV)*d, hidden]}
     */
    public Qkv(final PackedWeight weight) {
      super(weight, null);
    }
  }
}
