package com.igormaznitsa.nanollvm.tensor.kernels;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.LinearKernel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Dense float32 {@code [out, in]} GEMM. Decode ({@code rows == 1}) uses a specialized path.
 */
public final class DenseF32LinearKernel implements LinearKernel {

  private final float[] weight;
  private final int weightOffset;
  private final int inFeatures;
  private final int outFeatures;

  private DenseF32LinearKernel(
    final float[] weight,
    final int weightOffset,
    final int inFeatures,
    final int outFeatures
  ) {
    this.weight = weight;
    this.weightOffset = weightOffset;
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
  }

  public static DenseF32LinearKernel of(final Tensor weight) {
    requireNonNull(weight, "weight");
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException("dense weight must be 2D");
    }
    return new DenseF32LinearKernel(
      weight.data(), weight.offset(), weight.size(1), weight.size(0));
  }

  @Override
  public int inFeatures() {
    return this.inFeatures;
  }

  @Override
  public int outFeatures() {
    return this.outFeatures;
  }

  @Override
  public String name() {
    return "dense-f32";
  }

  @Override
  public void apply(
    final float[] x, final int xOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int rows,
    final MatmulRuntime matmul
  ) {
    if (rows == 1) {
      matmul.linearDecode1(
        x, xOff, this.weight, this.weightOffset, bias, y, yOff, this.inFeatures, this.outFeatures);
      return;
    }
    matmul.linear(
      x, xOff, this.weight, this.weightOffset, bias, y, yOff, rows, this.inFeatures,
      this.outFeatures);
  }
}
