package com.igormaznitsa.nanollvm.tensor;

import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_BF16;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_F16;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_F32;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q4_0;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q4_K;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q6_K;
import static com.igormaznitsa.nanollvm.internal.GgufDequant.TYPE_Q8_0;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.GgufDequant;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.tensor.kernels.DenseEmbeddingKernel;
import com.igormaznitsa.nanollvm.tensor.kernels.PackedEmbeddingKernel;

/**
 * Token embedding gather bound to one table at construction (dense or packed with fixed dequant).
 */
public sealed interface EmbeddingKernel permits DenseEmbeddingKernel, PackedEmbeddingKernel {

  static EmbeddingKernel of(final Tensor weight) {
    return DenseEmbeddingKernel.of(weight);
  }

  static EmbeddingKernel of(final PackedWeight weight) {
    requireNonNull(weight, "weight");
    return switch (weight.ggmlType()) {
      case TYPE_Q4_K -> packed(weight, "packed-q4_k-embed", TYPE_Q4_K);
      case TYPE_Q4_0 -> packed(weight, "packed-q4_0-embed", TYPE_Q4_0);
      case TYPE_Q6_K -> packed(weight, "packed-q6_k-embed", TYPE_Q6_K);
      case TYPE_Q8_0 -> packed(weight, "packed-q8_0-embed", TYPE_Q8_0);
      case TYPE_F16 -> packed(weight, "packed-f16-embed", TYPE_F16);
      case TYPE_BF16 -> packed(weight, "packed-bf16-embed", TYPE_BF16);
      case TYPE_F32 -> DenseEmbeddingKernel.of(weight.materialize());
      default -> PackedEmbeddingKernel.of(weight, "packed-ggml-" + weight.ggmlType() + "-embed",
        weight::dequantizeRow);
    };
  }

  private static EmbeddingKernel packed(
    final PackedWeight weight,
    final String name,
    final int ggmlType
  ) {
    int dim = weight.size(1);
    int numel = weight.numel();
    return PackedEmbeddingKernel.of(weight, name, (row, dst) ->
      GgufDequant.dequantizeRange(
        weight.rawPacked(), ggmlType, numel,
        Math.multiplyExact(row, dim), dim, dst, 0));
  }

  int vocabSize();

  int embeddingDim();

  String name();

  /**
   * Gathers rows for each token id in {@code ids[idsOff .. idsOff+count)}.
   * Ids are stored as floats (engine convention); each is {@link Math#round(float) rounded}.
   */
  void gather(float[] ids, int idsOff, int count, float[] out, int outOff);
}
