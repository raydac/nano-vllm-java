package com.igormaznitsa.nanollvm.tensor.kernels;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.LinearKernel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime.PackedRowDequant;

/**
 * Packed GEMM configured with a row-dequant bound at construction (GGML type fixed in the lambda).
 */
public final class PackedLinearKernel implements LinearKernel {

  private final int inFeatures;
  private final int outFeatures;
  private final String name;
  private final PackedRowDequant dequant;

  private PackedLinearKernel(
    final int inFeatures,
    final int outFeatures,
    final String name,
    final PackedRowDequant dequant
  ) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.name = name;
    this.dequant = dequant;
  }

  public static PackedLinearKernel of(
    final PackedWeight weight,
    final String name,
    final PackedRowDequant dequant
  ) {
    requireNonNull(weight, "weight");
    requireNonNull(name, "name");
    requireNonNull(dequant, "dequant");
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException("packed weight must be 2D");
    }
    return new PackedLinearKernel(weight.size(1), weight.size(0), name, dequant);
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
    return this.name;
  }

  @Override
  public void apply(
    final float[] x, final int xOff,
    final float[] bias,
    final float[] y, final int yOff,
    final int rows,
    final MatmulRuntime matmul
  ) {
    matmul.linearPackedRows(
      x, xOff, this.dequant, bias, y, yOff, rows, this.inFeatures, this.outFeatures);
  }
}
