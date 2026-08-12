package com.igormaznitsa.nanollvm.tensor;

import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_BF16;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_F16;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_F32;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_IQ4_NL;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q3_K;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q4_0;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q4_K;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q6_K;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q8_0;

import com.igormaznitsa.nanollvm.internal.GgufDequant;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.kernels.DenseF32LinearKernel;
import com.igormaznitsa.nanollvm.tensor.kernels.PackedLinearKernel;

/**
 * Affine map {@code y = x Wᵀ (+ bias)} bound to one weight matrix at layer construction.
 *
 * <p>Dense float32 and packed layouts use separate implementations; packed kernels fix the GGML
 * dequant in a lambda so the forward path does not switch on type every call.
 */
public sealed interface LinearKernel permits DenseF32LinearKernel, PackedLinearKernel {

  static LinearKernel of(final Tensor weight) {
    return DenseF32LinearKernel.of(weight);
  }

  static LinearKernel of(final PackedWeight weight) {
    return switch (weight.ggmlType()) {
      case TYPE_Q4_K -> packed(weight, "packed-q4_k", TYPE_Q4_K);
      case TYPE_Q3_K -> packed(weight, "packed-q3_k", TYPE_Q3_K);
      case TYPE_Q4_0 -> packed(weight, "packed-q4_0", TYPE_Q4_0);
      case TYPE_Q6_K -> packed(weight, "packed-q6_k", TYPE_Q6_K);
      case TYPE_Q8_0 -> packed(weight, "packed-q8_0", TYPE_Q8_0);
      case TYPE_IQ4_NL -> packed(weight, "packed-iq4_nl", TYPE_IQ4_NL);
      case TYPE_F16 -> packed(weight, "packed-f16", TYPE_F16);
      case TYPE_BF16 -> packed(weight, "packed-bf16", TYPE_BF16);
      case TYPE_F32 -> DenseF32LinearKernel.of(weight.materialize());
      default -> PackedLinearKernel.of(
        weight, "packed-ggml-" + weight.ggmlType(), weight::dequantizeRow);
    };
  }

  private static LinearKernel packed(
    final PackedWeight weight,
    final String name,
    final int ggmlType
  ) {
    int in = weight.size(1);
    int numel = weight.numel();
    return PackedLinearKernel.of(weight, name, (row, dst) ->
      GgufDequant.dequantizeRange(
        weight.rawPacked(), ggmlType, numel,
        Math.multiplyExact(row, in), in, dst, 0));
  }

  int inFeatures();

  int outFeatures();

  String name();

  /**
   * Writes {@code rows × out} outputs into {@code y} starting at {@code yOff}.
   *
   * @param bias length {@code outFeatures()}, or {@code null}
   */
  void apply(
    float[] x, int xOff,
    float[] bias,
    float[] y, int yOff,
    int rows,
    MatmulRuntime matmul
  );
}
