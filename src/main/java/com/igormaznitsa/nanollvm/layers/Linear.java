package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Immutable affine transform {@code y = x Wᵀ + b}. Weight (and optional bias) are fixed at
 * construction; there is no load/set API.
 */
public class Linear {

  protected final Tensor weight;
  protected final Tensor bias;

  public Linear(Tensor weight, Tensor bias) {
    this.weight = requireNonNull(weight, "weight");
    this.bias = bias;
  }

  public Tensor weight() {
    return this.weight;
  }

  public Tensor bias() {
    return this.bias;
  }

  public Tensor forward(Tensor x) {
    return Ops.linear(x, this.weight, this.bias);
  }

  public static class Column extends Linear {
    public Column(Tensor weight, Tensor bias) {
      super(weight, bias);
    }
  }

  public static final class Row extends Linear {
    public Row(Tensor weight) {
      super(weight, null);
    }

    public Row(Tensor weight, Tensor bias) {
      super(weight, bias);
    }
  }

  /**
   * Packed gate+up projection; {@code weight} shape {@code [2*intermediate, hidden]}.
   */
  public static final class Merged extends Column {
    public Merged(Tensor weight) {
      super(weight, null);
    }
  }

  /** Packed Q/K/V projection; {@code weight} shape {@code [(nH+2nKV)*d, hidden]}. */
  public static final class Qkv extends Column {
    public Qkv(Tensor weight, Tensor bias) {
      super(weight, bias);
    }

    public Qkv(Tensor weight) {
      this(weight, null);
    }
  }
}
