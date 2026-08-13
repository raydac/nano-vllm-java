package com.igormaznitsa.nanollvm.internal;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.OnnxProtoReader.OnnxTensorProto;

/**
 * ONNX {@code TensorProto.DataType} catalog for Tier A weight import.
 *
 * <p>Codes follow the ONNX TensorProto enum. Loadable weights are converted to float32;
 * integer / bool / string / complex graph constants are skipped; float8 / nibble / unknown
 * types fail loudly so quantized or future exports are not silently ignored.
 *
 * @since 1.1.0
 */
public final class OnnxDataTypes {

  public static final int UNDEFINED = 0;
  public static final int FLOAT = 1;
  public static final int UINT8 = 2;
  public static final int INT8 = 3;
  public static final int UINT16 = 4;
  public static final int INT16 = 5;
  public static final int INT32 = 6;
  public static final int INT64 = 7;
  public static final int STRING = 8;
  public static final int BOOL = 9;
  public static final int FLOAT16 = 10;
  public static final int DOUBLE = 11;
  public static final int UINT32 = 12;
  public static final int UINT64 = 13;
  public static final int COMPLEX64 = 14;
  public static final int COMPLEX128 = 15;
  public static final int BFLOAT16 = 16;
  public static final int FLOAT8E4M3FN = 17;
  public static final int FLOAT8E4M3FNUZ = 18;
  public static final int FLOAT8E5M2 = 19;
  public static final int FLOAT8E5M2FNUZ = 20;
  public static final int UINT4 = 21;
  public static final int INT4 = 22;
  public static final int FLOAT4E2M1 = 23;

  private OnnxDataTypes() {
  }

  /**
   * Classifies a TensorProto type code for Tier A import policy.
   *
   * @since 1.1.0
   */
  public static Kind kind(final int dataType) {
    return switch (dataType) {
      case FLOAT, FLOAT16, BFLOAT16, DOUBLE -> Kind.LOADABLE_FLOAT;
      case UNDEFINED, UINT8, INT8, UINT16, INT16, INT32, INT64, STRING, BOOL,
           UINT32, UINT64, COMPLEX64, COMPLEX128 -> Kind.SKIP_GRAPH_CONSTANT;
      default -> Kind.UNSUPPORTED_WEIGHT;
    };
  }

  /**
   * Human-readable TensorProto type label (or {@code UNKNOWN(n)}).
   *
   * @since 1.1.0
   */
  public static String name(final int dataType) {
    return switch (dataType) {
      case UNDEFINED -> "UNDEFINED";
      case FLOAT -> "FLOAT";
      case UINT8 -> "UINT8";
      case INT8 -> "INT8";
      case UINT16 -> "UINT16";
      case INT16 -> "INT16";
      case INT32 -> "INT32";
      case INT64 -> "INT64";
      case STRING -> "STRING";
      case BOOL -> "BOOL";
      case FLOAT16 -> "FLOAT16";
      case DOUBLE -> "DOUBLE";
      case UINT32 -> "UINT32";
      case UINT64 -> "UINT64";
      case COMPLEX64 -> "COMPLEX64";
      case COMPLEX128 -> "COMPLEX128";
      case BFLOAT16 -> "BFLOAT16";
      case FLOAT8E4M3FN -> "FLOAT8E4M3FN";
      case FLOAT8E4M3FNUZ -> "FLOAT8E4M3FNUZ";
      case FLOAT8E5M2 -> "FLOAT8E5M2";
      case FLOAT8E5M2FNUZ -> "FLOAT8E5M2FNUZ";
      case UINT4 -> "UINT4";
      case INT4 -> "INT4";
      case FLOAT4E2M1 -> "FLOAT4E2M1";
      default -> "UNKNOWN(" + dataType + ")";
    };
  }

  /**
   * {@code true} for FLOAT / FLOAT16 / BFLOAT16 / DOUBLE.
   *
   * @since 1.1.0
   */
  public static boolean isLoadableFloatingWeight(final int dataType) {
    return kind(dataType) == Kind.LOADABLE_FLOAT;
  }

  /**
   * {@code true} for int / bool / string / complex / undefined graph constants.
   *
   * @since 1.1.0
   */
  public static boolean isSkippableGraphConstant(final int dataType) {
    return kind(dataType) == Kind.SKIP_GRAPH_CONSTANT;
  }

  /**
   * {@code true} when dims are present and non-empty (scalars are not weight tensors).
   *
   * @since 1.1.0
   */
  public static boolean hasWeightRank(final OnnxTensorProto proto) {
    return proto.dims() != null && proto.dims().length > 0;
  }

  /**
   * {@code true} when this initializer should be decoded into the weight bag.
   *
   * @since 1.1.0
   */
  public static boolean shouldLoadAsWeight(final OnnxTensorProto proto) {
    return proto.name() != null
      && !proto.name().isBlank()
      && hasWeightRank(proto)
      && isLoadableFloatingWeight(proto.dataType());
  }

  /**
   * Ensures a non-loadable initializer is a known skippable constant; otherwise fails fast.
   *
   * @since 1.1.0
   */
  public static void requireHandledOrSkip(final OnnxTensorProto proto) {
    if (shouldLoadAsWeight(proto) || !hasWeightRank(proto)) {
      return;
    }
    if (proto.name() == null || proto.name().isBlank()) {
      return;
    }
    if (isSkippableGraphConstant(proto.dataType())) {
      return;
    }
    throw new ModelLoadException(
      "ONNX initializer '%s' has unsupported data_type %s (%d); Tier A loads only FLOAT / FLOAT16 / BFLOAT16 / DOUBLE weights"
        .formatted(proto.name(), name(proto.dataType()), proto.dataType()));
  }

  /**
   * How Tier A treats a {@code TensorProto.DataType} code.
   *
   * @since 1.1.0
   */
  public enum Kind {
    /**
     * FLOAT / FLOAT16 / BFLOAT16 / DOUBLE → decode to float32 weights.
     */
    LOADABLE_FLOAT,
    /** Int / bool / string / complex / undefined graph constants — ignore. */
    SKIP_GRAPH_CONSTANT,
    /** Float8 / nibble / unknown — fail loud; never silent-drop a weight. */
    UNSUPPORTED_WEIGHT
  }
}
