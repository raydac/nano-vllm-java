package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.LinearKernel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Immutable affine transform {@code y = x Wᵀ + b}. A {@link LinearKernel} is bound at construction
 * (dense F32, or packed with a GGML-type-fixed dequant).
 */
public class Linear {

  protected final Tensor weight;
  protected final PackedWeight packedWeight;
  protected final Tensor bias;
  private final LinearKernel kernel;

  public Linear(final Tensor weight, final Tensor bias) {
    this.weight = requireNonNull(weight, "weight");
    this.packedWeight = null;
    this.bias = bias;
    this.kernel = LinearKernel.of(weight);
  }

  public Linear(final PackedWeight weight, final Tensor bias) {
    this.weight = null;
    this.packedWeight = requireNonNull(weight, "weight");
    this.bias = bias;
    this.kernel = LinearKernel.of(weight);
  }

  public Linear(final PackedWeight weight) {
    this(weight, null);
  }

  public Tensor weight() {
    return this.weight != null ? this.weight : this.packedWeight.materialize();
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
    this.kernel.apply(x.data(), x.offset(), biasData, y.data(), 0, rows, matmul);
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
