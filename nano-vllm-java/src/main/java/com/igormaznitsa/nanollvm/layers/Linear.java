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
 * (dense F32, or packed with a GGML-type-fixed dequant).
 */
public class Linear {

  protected final Tensor weight;
  protected final PackedWeight packedWeight;
  protected final Tensor bias;
  private final LinearKernel kernel;
  private final float inputActivationScale;
  private final float outputActivationScale;

  public Linear(final Tensor weight, final Tensor bias) {
    this(LinearKernel.of(requireNonNull(weight, "weight")), bias, 0f, 0f, weight, null);
  }

  public Linear(final PackedWeight weight, final Tensor bias) {
    this(LinearKernel.of(requireNonNull(weight, "weight")), bias, 0f, 0f, null, weight);
  }

  public Linear(final PackedWeight weight) {
    this(weight, null);
  }

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

  public Tensor weight() {
    if (this.weight != null) {
      return this.weight;
    }
    if (this.packedWeight != null) {
      return this.packedWeight.materialize();
    }
    throw new IllegalStateException("linear has no dense weight table");
  }

  public PackedWeight packedWeight() {
    return this.packedWeight;
  }

  public boolean isPacked() {
    return this.packedWeight != null;
  }

  public LinearKernel kernel() {
    return this.kernel;
  }

  public Tensor bias() {
    return this.bias;
  }

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
      Tensor quantized = this.scaleActivations(x, this.inputActivationScale);
      xData = quantized.data();
      xOff = quantized.offset();
    }
    this.kernel.apply(xData, xOff, biasData, y.data(), 0, rows, matmul);
    if (this.outputActivationScale != 0f) {
      y = this.scaleActivations(y, this.outputActivationScale);
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

  private Tensor scaleActivations(final Tensor values, final float scale) {
    Tensor out = Tensor.zeros(values.shape());
    float[] source = values.data();
    float[] dest = out.data();
    int off = values.offset();
    int n = values.numel();
    for (int i = 0; i < n; i++) {
      dest[i] = GemmaQat.applySrq(source[off + i], scale);
    }
    return out;
  }

  public static class Column extends Linear {
    public Column(final Tensor weight, final Tensor bias) {
      super(weight, bias);
    }

    public Column(final PackedWeight weight, final Tensor bias) {
      super(weight, bias);
    }
  }

  public static final class Row extends Linear {
    public Row(final Tensor weight) {
      super(weight, null);
    }

    public Row(final Tensor weight, final Tensor bias) {
      super(weight, bias);
    }

    public Row(final PackedWeight weight) {
      super(weight, null);
    }

    public Row(final PackedWeight weight, final Tensor bias) {
      super(weight, bias);
    }

    public Row(final GemmaQatWeight weight) {
      super(weight);
    }
  }

  /**
   * Packed gate+up projection; {@code weight} shape {@code [2*intermediate, hidden]}.
   */
  public static final class Merged extends Column {
    public Merged(final Tensor weight) {
      super(weight, null);
    }

    public Merged(final PackedWeight weight) {
      super(weight, null);
    }
  }

  /**
   * Packed Q/K/V projection; {@code weight} shape {@code [(nH+2nKV)*d, hidden]}.
   */
  public static final class Qkv extends Column {
    public Qkv(final Tensor weight, final Tensor bias) {
      super(weight, bias);
    }

    public Qkv(final Tensor weight) {
      this(weight, null);
    }

    public Qkv(final PackedWeight weight) {
      super(weight, null);
    }
  }
}
