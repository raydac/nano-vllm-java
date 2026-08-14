package com.igormaznitsa.nanollvm.internal;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Dequantizes GGML block layouts into float32 (ported from ggml row dequant kernels).
 *
 * <p>{@link #dequantizeRange} writes into the caller buffer: full blocks go straight to
 * {@code dst}; only partial block edges use a small one-block scratch (no per-row full-tensor
 * allocation).
 */
public final class GgufDequant {

  public static final int TYPE_F32 = 0;
  public static final int TYPE_F16 = 1;
  public static final int TYPE_Q4_0 = 2;
  public static final int TYPE_Q4_1 = 3;
  public static final int TYPE_Q5_0 = 6;
  public static final int TYPE_Q5_1 = 7;
  public static final int TYPE_Q8_0 = 8;
  public static final int TYPE_Q8_1 = 9;
  public static final int TYPE_Q2_K = 10;
  public static final int TYPE_Q3_K = 11;
  public static final int TYPE_Q4_K = 12;
  public static final int TYPE_Q5_K = 13;
  public static final int TYPE_Q6_K = 14;
  public static final int TYPE_Q8_K = 15;
  public static final int TYPE_IQ2_XXS = 16;
  public static final int TYPE_IQ2_XS = 17;
  public static final int TYPE_IQ3_XXS = 18;
  public static final int TYPE_IQ1_S = 19;
  public static final int TYPE_IQ4_NL = 20;
  public static final int TYPE_IQ3_S = 21;
  public static final int TYPE_IQ2_S = 22;
  public static final int TYPE_IQ4_XS = 23;
  public static final int TYPE_I8 = 24;
  public static final int TYPE_I16 = 25;
  public static final int TYPE_I32 = 26;
  public static final int TYPE_I64 = 27;
  public static final int TYPE_F64 = 28;
  public static final int TYPE_IQ1_M = 29;
  public static final int TYPE_BF16 = 30;
  public static final int TYPE_TQ1_0 = 34;
  public static final int TYPE_TQ2_0 = 35;
  public static final int TYPE_MXFP4 = 39;
  public static final int TYPE_NVFP4 = 40;
  public static final int TYPE_Q1_0 = 41;
  public static final int TYPE_Q2_0 = 42;

  public static final int QK1_0 = 128;
  public static final int QK2_0 = 64;
  public static final int QK4_0 = 32;
  public static final int QK4_1 = 32;
  public static final int QK5_0 = 32;
  public static final int QK5_1 = 32;
  public static final int QK8_0 = 32;
  public static final int QK8_1 = 32;
  public static final int QK4_NL = 32;
  public static final int QK_MXFP4 = 32;
  public static final int QK_NVFP4 = 64;
  public static final int QK_NVFP4_SUB = 16;
  public static final int QK_K = 256;
  public static final int BLOCK_Q1_0 = 2 + QK1_0 / 8;
  public static final int BLOCK_Q2_0 = 2 + QK2_0 / 4;
  public static final int BLOCK_Q4_0 = 2 + QK4_0 / 2;
  public static final int BLOCK_Q4_1 = 4 + QK4_1 / 2;
  public static final int BLOCK_Q5_0 = 2 + 4 + QK5_0 / 2;
  public static final int BLOCK_Q5_1 = 4 + 4 + QK5_1 / 2;
  public static final int BLOCK_Q8_0 = 2 + QK8_0;
  public static final int BLOCK_Q8_1 = 4 + QK8_1;
  public static final int BLOCK_MXFP4 = 1 + QK_MXFP4 / 2;
  public static final int BLOCK_NVFP4 = QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2;
  public static final int BLOCK_Q2_K = 4 + QK_K / 16 + QK_K / 4;
  public static final int BLOCK_Q3_K = QK_K / 8 + QK_K / 4 + 12 + 2;
  public static final int BLOCK_Q4_K = 2 + 2 + 12 + QK_K / 2;
  public static final int BLOCK_Q5_K = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
  public static final int BLOCK_Q6_K = 2 + QK_K / 16 + 3 * QK_K / 4;
  public static final int BLOCK_Q8_K = 4 + QK_K + QK_K / 8;
  public static final int BLOCK_IQ2_XXS = 2 + QK_K / 8 * 2;
  public static final int BLOCK_IQ2_XS = 2 + QK_K / 8 * 2 + QK_K / 32;
  public static final int BLOCK_IQ2_S = 2 + QK_K / 4 + QK_K / 16;
  public static final int BLOCK_IQ3_XXS = 2 + 3 * (QK_K / 8);
  public static final int BLOCK_IQ3_S = 2 + 13 * (QK_K / 32) + QK_K / 64;
  public static final int BLOCK_IQ1_S = 2 + QK_K / 8 + QK_K / 16;
  public static final int BLOCK_IQ1_M = QK_K / 8 + QK_K / 16 + QK_K / 32;
  public static final int BLOCK_IQ4_NL = 2 + QK4_NL / 2;
  public static final int BLOCK_IQ4_XS = 2 + 2 + QK_K / 64 + QK_K / 2;
  public static final int BLOCK_TQ1_0 = 2 + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5;
  public static final int BLOCK_TQ2_0 = 2 + QK_K / 4;

  private static final int[] KVALUES_IQ4NL = {
      -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
  };

  private GgufDequant() {
  }

  public static float[] dequantize(final ByteBuffer src, final int ggmlType, final long numel) {
    int n = requireInt(numel);
    if (isUnitStride(ggmlType)) {
      return switch (ggmlType) {
        case TYPE_F32 -> readF32(src, n);
        case TYPE_F16 -> readF16(src, n);
        case TYPE_BF16 -> readBf16(src, n);
        case TYPE_I8 -> readI8(src, n);
        case TYPE_I16 -> readI16(src, n);
        case TYPE_I32 -> readI32(src, n);
        case TYPE_I64 -> readI64(src, n);
        case TYPE_F64 -> readF64(src, n);
        default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
      };
    }
    byte[] packed = copyRemaining(src);
    float[] out = new float[n];
    dequantizeRange(packed, ggmlType, n, 0, n, out, 0);
    return out;
  }

  public static int typeBlockSize(final int ggmlType) {
    return switch (ggmlType) {
      case TYPE_F32, TYPE_I32 -> 4;
      case TYPE_F16, TYPE_BF16, TYPE_I16 -> 2;
      case TYPE_I8 -> 1;
      case TYPE_I64, TYPE_F64 -> 8;
      case TYPE_Q1_0 -> BLOCK_Q1_0;
      case TYPE_Q2_0 -> BLOCK_Q2_0;
      case TYPE_Q4_0 -> BLOCK_Q4_0;
      case TYPE_Q4_1 -> BLOCK_Q4_1;
      case TYPE_Q5_0 -> BLOCK_Q5_0;
      case TYPE_Q5_1 -> BLOCK_Q5_1;
      case TYPE_Q8_0 -> BLOCK_Q8_0;
      case TYPE_Q8_1 -> BLOCK_Q8_1;
      case TYPE_MXFP4 -> BLOCK_MXFP4;
      case TYPE_NVFP4 -> BLOCK_NVFP4;
      case TYPE_Q2_K -> BLOCK_Q2_K;
      case TYPE_Q3_K -> BLOCK_Q3_K;
      case TYPE_Q4_K -> BLOCK_Q4_K;
      case TYPE_Q5_K -> BLOCK_Q5_K;
      case TYPE_Q6_K -> BLOCK_Q6_K;
      case TYPE_Q8_K -> BLOCK_Q8_K;
      case TYPE_IQ2_XXS -> BLOCK_IQ2_XXS;
      case TYPE_IQ2_XS -> BLOCK_IQ2_XS;
      case TYPE_IQ2_S -> BLOCK_IQ2_S;
      case TYPE_IQ3_XXS -> BLOCK_IQ3_XXS;
      case TYPE_IQ3_S -> BLOCK_IQ3_S;
      case TYPE_IQ1_S -> BLOCK_IQ1_S;
      case TYPE_IQ1_M -> BLOCK_IQ1_M;
      case TYPE_IQ4_NL -> BLOCK_IQ4_NL;
      case TYPE_IQ4_XS -> BLOCK_IQ4_XS;
      case TYPE_TQ1_0 -> BLOCK_TQ1_0;
      case TYPE_TQ2_0 -> BLOCK_TQ2_0;
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    };
  }

  public static int typeBlockElems(final int ggmlType) {
    return switch (ggmlType) {
      case TYPE_F32, TYPE_F16, TYPE_BF16, TYPE_I8, TYPE_I16, TYPE_I32, TYPE_I64, TYPE_F64 -> 1;
      case TYPE_Q1_0 -> QK1_0;
      case TYPE_Q2_0 -> QK2_0;
      case TYPE_Q4_0, TYPE_Q4_1, TYPE_Q5_0, TYPE_Q5_1, TYPE_Q8_0, TYPE_Q8_1, TYPE_IQ4_NL,
           TYPE_MXFP4 -> QK4_0;
      case TYPE_NVFP4 -> QK_NVFP4;
      case TYPE_Q2_K, TYPE_Q3_K, TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_K,
           TYPE_IQ2_XXS, TYPE_IQ2_XS, TYPE_IQ2_S, TYPE_IQ3_XXS, TYPE_IQ3_S,
           TYPE_IQ1_S, TYPE_IQ1_M, TYPE_IQ4_XS, TYPE_TQ1_0, TYPE_TQ2_0 -> QK_K;
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    };
  }

  public static long packedByteLength(final int ggmlType, final long numel) {
    int blockElems = typeBlockElems(ggmlType);
    int blockBytes = typeBlockSize(ggmlType);
    long blocks = (numel + blockElems - 1) / blockElems;
    return Math.multiplyExact(blocks, blockBytes);
  }

  /**
   * Dequantizes {@code elemCount} elements starting at {@code elemStart} into {@code dst}.
   * Packed layout matches full-tensor {@link #dequantize} order (GGML storage / HF row-major
   * after {@code GgufReader} shape remap).
   */
  public static void dequantizeRange(
    final byte[] packed,
    final int ggmlType,
    final int numel,
    final int elemStart,
    final int elemCount,
    final float[] dst,
    final int dstOff) {
    if (elemCount < 0 || elemStart < 0 || elemStart > numel - elemCount) {
      throw new IndexOutOfBoundsException(
        "range [" + elemStart + "," + (elemStart + elemCount) + ") vs numel " + numel);
    }
    if (dstOff < 0 || dstOff > dst.length - elemCount) {
      throw new IndexOutOfBoundsException("dst range");
    }
    if (elemCount == 0) {
      return;
    }

    switch (ggmlType) {
      case TYPE_F32, TYPE_F16, TYPE_BF16, TYPE_I8, TYPE_I16, TYPE_I32, TYPE_I64, TYPE_F64 ->
        dequantizeUnitStride(packed, ggmlType, elemStart, elemCount, dst, dstOff);
      case TYPE_Q4_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK4_0, BLOCK_Q4_0, GgufDequant::dequantOneQ4_0);
      case TYPE_Q4_1 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK4_1, BLOCK_Q4_1, GgufDequant::dequantOneQ4_1);
      case TYPE_Q5_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK5_0, BLOCK_Q5_0, GgufDequant::dequantOneQ5_0);
      case TYPE_Q5_1 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK5_1, BLOCK_Q5_1, GgufDequant::dequantOneQ5_1);
      case TYPE_Q8_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK8_0, BLOCK_Q8_0, GgufDequant::dequantOneQ8_0);
      case TYPE_Q8_1 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK8_1, BLOCK_Q8_1, GgufDequant::dequantOneQ8_1);
      case TYPE_Q2_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_Q2_K, GgufDequant::dequantOneQ2_K);
      case TYPE_Q3_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
          QK_K, BLOCK_Q3_K, GgufDequant::dequantOneQ3_K);
      case TYPE_Q4_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_Q4_K, GgufDequant::dequantOneQ4_K);
      case TYPE_Q5_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_Q5_K, GgufDequant::dequantOneQ5_K);
      case TYPE_Q6_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_Q6_K, GgufDequant::dequantOneQ6_K);
      case TYPE_Q8_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_Q8_K, GgufDequant::dequantOneQ8_K);
      case TYPE_IQ4_NL -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
          QK4_NL, BLOCK_IQ4_NL, GgufDequant::dequantOneIq4Nl);
      case TYPE_IQ4_XS -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ4_XS, GgufIqDequant::dequantOneIq4Xs);
      case TYPE_IQ2_XXS -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ2_XXS, GgufIqDequant::dequantOneIq2Xxs);
      case TYPE_IQ2_XS -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ2_XS, GgufIqDequant::dequantOneIq2Xs);
      case TYPE_IQ2_S -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ2_S, GgufIqDequant::dequantOneIq2S);
      case TYPE_IQ3_XXS -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ3_XXS, GgufIqDequant::dequantOneIq3Xxs);
      case TYPE_IQ3_S -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ3_S, GgufIqDequant::dequantOneIq3S);
      case TYPE_IQ1_S -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ1_S, GgufIqDequant::dequantOneIq1S);
      case TYPE_IQ1_M -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_IQ1_M, GgufIqDequant::dequantOneIq1M);
      case TYPE_TQ1_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_TQ1_0, GgufIqDequant::dequantOneTq1_0);
      case TYPE_TQ2_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_TQ2_0, GgufIqDequant::dequantOneTq2_0);
      case TYPE_MXFP4 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_MXFP4, BLOCK_MXFP4, GgufIqDequant::dequantOneMxfp4);
      case TYPE_NVFP4 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_NVFP4, BLOCK_NVFP4, GgufIqDequant::dequantOneNvfp4);
      case TYPE_Q1_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK1_0, BLOCK_Q1_0, GgufIqDequant::dequantOneQ1_0);
      case TYPE_Q2_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK2_0, BLOCK_Q2_0, GgufIqDequant::dequantOneQ2_0);
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    }
  }

  private static void dequantizeBlockedRange(
    final byte[] packed,
    final int numel,
    final int elemStart,
    final int elemCount,
    final float[] dst,
    final int dstOff,
    final int blockElems,
    final int blockBytes,
    final BlockDequant blockDequant
  ) {
    if (numel % blockElems != 0) {
      throw new IllegalArgumentException("numel must be multiple of " + blockElems);
    }

    float[] scratch = null;
    int elem = elemStart;
    int remaining = elemCount;
    int out = dstOff;
    while (remaining > 0) {
      int inBlock = elem % blockElems;
      int take = Math.min(remaining, blockElems - inBlock);
      int byteOff = Math.multiplyExact(elem / blockElems, blockBytes);
      if (inBlock == 0 && take == blockElems) {
        blockDequant.dequant(packed, byteOff, dst, out);
      } else {
        if (scratch == null) {
          scratch = new float[blockElems];
        }
        blockDequant.dequant(packed, byteOff, scratch, 0);
        System.arraycopy(scratch, inBlock, dst, out, take);
      }
      elem += take;
      out += take;
      remaining -= take;
    }
  }

  private static void dequantizeUnitStride(
    final byte[] packed,
    final int ggmlType,
    final int elemStart,
    final int elemCount,
    final float[] dst,
    final int dstOff) {
    switch (ggmlType) {
      case TYPE_F32 -> {
        int byteOff = Math.multiplyExact(elemStart, 4);
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = Float.intBitsToFloat(i32LE(packed, byteOff + i * 4));
        }
      }
      case TYPE_F16 -> {
        int byteOff = Math.multiplyExact(elemStart, 2);
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = Float.float16ToFloat((short) u16LE(packed, byteOff + i * 2));
        }
      }
      case TYPE_BF16 -> {
        int byteOff = Math.multiplyExact(elemStart, 2);
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = Float.intBitsToFloat(u16LE(packed, byteOff + i * 2) << 16);
        }
      }
      case TYPE_I8 -> {
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = packed[elemStart + i];
        }
      }
      case TYPE_I16 -> {
        int byteOff = Math.multiplyExact(elemStart, 2);
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = (short) u16LE(packed, byteOff + i * 2);
        }
      }
      case TYPE_I32 -> {
        int byteOff = Math.multiplyExact(elemStart, 4);
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = i32LE(packed, byteOff + i * 4);
        }
      }
      case TYPE_I64 -> {
        int byteOff = Math.multiplyExact(elemStart, 8);
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = i64LE(packed, byteOff + i * 8);
        }
      }
      case TYPE_F64 -> {
        int byteOff = Math.multiplyExact(elemStart, 8);
        for (int i = 0; i < elemCount; i++) {
          dst[dstOff + i] = (float) Double.longBitsToDouble(i64LE(packed, byteOff + i * 8));
        }
      }
      default -> throw new UnsupportedOperationException("unit-stride type " + ggmlType);
    }
  }

  private static boolean isUnitStride(final int ggmlType) {
    return ggmlType == TYPE_F32 || ggmlType == TYPE_F16 || ggmlType == TYPE_BF16
      || ggmlType == TYPE_I8 || ggmlType == TYPE_I16 || ggmlType == TYPE_I32
      || ggmlType == TYPE_I64 || ggmlType == TYPE_F64;
  }

  private static float[] readF32(final ByteBuffer src, final long numel) {
    float[] out = new float[requireInt(numel)];
    src.asFloatBuffer().get(out);
    return out;
  }

  private static float[] readF16(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = Float.float16ToFloat(src.getShort());
    }
    return out;
  }

  private static float[] readBf16(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = Float.intBitsToFloat((src.getShort() & 0xFFFF) << 16);
    }
    return out;
  }

  private static float[] readI8(final ByteBuffer src, final int n) {
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = src.get();
    }
    return out;
  }

  private static float[] readI16(final ByteBuffer src, final int n) {
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = src.getShort();
    }
    return out;
  }

  private static float[] readI32(final ByteBuffer src, final int n) {
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = src.getInt();
    }
    return out;
  }

  private static float[] readI64(final ByteBuffer src, final int n) {
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = src.getLong();
    }
    return out;
  }

  private static float[] readF64(final ByteBuffer src, final int n) {
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = (float) src.getDouble();
    }
    return out;
  }

  private static void dequantOneQ4_0(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    int qsOff = byteOff + 2;
    for (int j = 0; j < QK4_0 / 2; j++) {
      int qs = packed[qsOff + j] & 0xFF;
      dst[dstOff + j] = ((qs & 0x0F) - 8) * d;
      dst[dstOff + j + QK4_0 / 2] = ((qs >> 4) - 8) * d;
    }
  }

  private static void dequantOneQ8_0(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    int qsOff = byteOff + 2;
    for (int j = 0; j < QK8_0; j++) {
      dst[dstOff + j] = packed[qsOff + j] * d;
    }
  }

  private static void dequantOneQ4_1(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    float m = Float.float16ToFloat((short) u16LE(packed, byteOff + 2));
    int qsOff = byteOff + 4;
    for (int j = 0; j < QK4_1 / 2; j++) {
      int qs = packed[qsOff + j] & 0xFF;
      dst[dstOff + j] = (qs & 0x0F) * d + m;
      dst[dstOff + j + QK4_1 / 2] = (qs >> 4) * d + m;
    }
  }

  private static void dequantOneQ5_0(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    int qh = i32LE(packed, byteOff + 2);
    int qsOff = byteOff + 6;
    for (int j = 0; j < QK5_0 / 2; j++) {
      int qs = packed[qsOff + j] & 0xFF;
      int xh0 = ((qh >>> j) << 4) & 0x10;
      int xh1 = (qh >>> (j + 12)) & 0x10;
      dst[dstOff + j] = (((qs & 0x0F) | xh0) - 16) * d;
      dst[dstOff + j + QK5_0 / 2] = (((qs >> 4) | xh1) - 16) * d;
    }
  }

  private static void dequantOneQ5_1(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    float m = Float.float16ToFloat((short) u16LE(packed, byteOff + 2));
    int qh = i32LE(packed, byteOff + 4);
    int qsOff = byteOff + 8;
    for (int j = 0; j < QK5_1 / 2; j++) {
      int qs = packed[qsOff + j] & 0xFF;
      int xh0 = ((qh >>> j) << 4) & 0x10;
      int xh1 = (qh >>> (j + 12)) & 0x10;
      dst[dstOff + j] = ((qs & 0x0F) | xh0) * d + m;
      dst[dstOff + j + QK5_1 / 2] = ((qs >> 4) | xh1) * d + m;
    }
  }

  private static void dequantOneQ8_1(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    int qsOff = byteOff + 4;
    for (int j = 0; j < QK8_1; j++) {
      dst[dstOff + j] = packed[qsOff + j] * d;
    }
  }

  private static void dequantOneQ2_K(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    int scalesOff = byteOff;
    int qsOff = byteOff + QK_K / 16;
    float d = Float.float16ToFloat((short) u16LE(packed, qsOff + QK_K / 4));
    float min = Float.float16ToFloat((short) u16LE(packed, qsOff + QK_K / 4 + 2));
    int y = dstOff;
    int q = qsOff;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; j++) {
        int sc = packed[scalesOff + is++] & 0xFF;
        float dl = d * (sc & 0x0F);
        float ml = min * (sc >> 4);
        for (int l = 0; l < 16; l++) {
          dst[y++] = dl * ((packed[q + l] >> shift) & 3) - ml;
        }
        sc = packed[scalesOff + is++] & 0xFF;
        dl = d * (sc & 0x0F);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; l++) {
          dst[y++] = dl * ((packed[q + 16 + l] >> shift) & 3) - ml;
        }
        shift += 2;
      }
      q += 32;
    }
  }

  private static void dequantOneQ5_K(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    float min = Float.float16ToFloat((short) u16LE(packed, byteOff + 2));
    int scalesOff = byteOff + 4;
    int qhOff = byteOff + 16;
    int qsOff = byteOff + 16 + QK_K / 8;
    int y = dstOff;
    int ql = 0;
    int is = 0;
    int u1 = 1;
    int u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
      int sc0 = scaleMinK4(is, packed, scalesOff);
      int m0 = scaleMinK4Min(is, packed, scalesOff);
      float d1 = d * sc0;
      float m1 = min * m0;
      int sc1 = scaleMinK4(is + 1, packed, scalesOff);
      int m1b = scaleMinK4Min(is + 1, packed, scalesOff);
      float d2 = d * sc1;
      float m2 = min * m1b;
      for (int l = 0; l < 32; l++) {
        dst[y++] = d1 * ((packed[qsOff + ql + l] & 0x0F) + ((packed[qhOff + l] & u1) != 0 ? 16 : 0))
          - m1;
      }
      for (int l = 0; l < 32; l++) {
        dst[y++] = d2 * (((packed[qsOff + ql + l] & 0xFF) >> 4)
          + ((packed[qhOff + l] & u2) != 0 ? 16 : 0)) - m2;
      }
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }

  private static void dequantOneQ8_K(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.intBitsToFloat(i32LE(packed, byteOff));
    int qsOff = byteOff + 4;
    for (int j = 0; j < QK_K; j++) {
      dst[dstOff + j] = d * packed[qsOff + j];
    }
  }

  private static void dequantOneQ3_K(
      final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    int hmaskOff = byteOff;
    int qsOff = byteOff + QK_K / 8;
    int scalesOff = qsOff + QK_K / 4;
    float dAll = Float.float16ToFloat((short) u16LE(packed, scalesOff + 12));

    int[] aux = new int[4];
    for (int i = 0; i < 3; i++) {
      aux[i] = i32LE(packed, scalesOff + i * 4);
    }
    final int kmask1 = 0x03030303;
    final int kmask2 = 0x0f0f0f0f;
    int tmp = aux[2];
    aux[2] = ((aux[0] >>> 4) & kmask2) | (((tmp >>> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >>> 4) & kmask2) | (((tmp >>> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >>> 2) & kmask1) << 4);
    byte[] scales = new byte[16];
    for (int i = 0; i < 4; i++) {
      scales[i * 4] = (byte) (aux[i] & 0xFF);
      scales[i * 4 + 1] = (byte) ((aux[i] >>> 8) & 0xFF);
      scales[i * 4 + 2] = (byte) ((aux[i] >>> 16) & 0xFF);
      scales[i * 4 + 3] = (byte) ((aux[i] >>> 24) & 0xFF);
    }

    int y = dstOff;
    int qBase = qsOff;
    int m = 1;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; j++) {
        float dl = dAll * (scales[is++] - 32);
        for (int l = 0; l < 16; l++) {
          int q = (packed[qBase + l] >> shift) & 3;
          int hm = (packed[hmaskOff + l] & m) != 0 ? 0 : 4;
          dst[y++] = dl * (q - hm);
        }
        dl = dAll * (scales[is++] - 32);
        for (int l = 0; l < 16; l++) {
          int q = (packed[qBase + 16 + l] >> shift) & 3;
          int hm = (packed[hmaskOff + 16 + l] & m) != 0 ? 0 : 4;
          dst[y++] = dl * (q - hm);
        }
        shift += 2;
        m <<= 1;
      }
      qBase += 32;
    }
  }

  private static void dequantOneIq4Nl(
      final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    int qsOff = byteOff + 2;
    for (int j = 0; j < QK4_NL / 2; j++) {
      int qs = packed[qsOff + j] & 0xFF;
      dst[dstOff + j] = d * KVALUES_IQ4NL[qs & 0x0F];
      dst[dstOff + j + QK4_NL / 2] = d * KVALUES_IQ4NL[qs >> 4];
    }
  }

  private static void dequantOneQ4_K(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = Float.float16ToFloat((short) u16LE(packed, byteOff));
    float min = Float.float16ToFloat((short) u16LE(packed, byteOff + 2));
    int scalesOff = byteOff + 4;
    int qsOff = byteOff + 16;
    int y = dstOff;
    int qOff = 0;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
      int sc0 = scaleMinK4(is, packed, scalesOff);
      int m0 = scaleMinK4Min(is, packed, scalesOff);
      float d1 = d * sc0;
      float m1 = min * m0;
      int sc1 = scaleMinK4(is + 1, packed, scalesOff);
      int m1b = scaleMinK4Min(is + 1, packed, scalesOff);
      float d2 = d * sc1;
      float m2 = min * m1b;
      for (int l = 0; l < 32; l++) {
        dst[y++] = d1 * (packed[qsOff + qOff + l] & 0x0F) - m1;
      }
      for (int l = 0; l < 32; l++) {
        dst[y++] = d2 * ((packed[qsOff + qOff + l] & 0xFF) >> 4) - m2;
      }
      qOff += 32;
      is += 2;
    }
  }

  private static void dequantOneQ6_K(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    int qlOff = byteOff;
    int qhOff = byteOff + QK_K / 2;
    int scOff = byteOff + QK_K / 2 + QK_K / 4;
    float d = Float.float16ToFloat((short) u16LE(packed, scOff + QK_K / 16));
    int y = dstOff;
    int ql = 0;
    int qh = 0;
    int sc = 0;
    for (int block = 0; block < QK_K; block += 128) {
      for (int l = 0; l < 32; l++) {
        int is = l / 16;
        int q1 = (packed[qlOff + ql + l] & 0x0F) | ((packed[qhOff + qh + l] & 3) << 4);
        int q2 = (packed[qlOff + ql + l + 32] & 0x0F) | (((packed[qhOff + qh + l] >> 2) & 3) << 4);
        int q3 =
          ((packed[qlOff + ql + l] & 0xFF) >> 4) | (((packed[qhOff + qh + l] >> 4) & 3) << 4);
        int q4 = ((packed[qlOff + ql + l + 32] & 0xFF) >> 4)
          | (((packed[qhOff + qh + l] >> 6) & 3) << 4);
        dst[y + l] = d * packed[scOff + sc + is] * (q1 - 32);
        dst[y + l + 32] = d * packed[scOff + sc + is + 2] * (q2 - 32);
        dst[y + l + 64] = d * packed[scOff + sc + is + 4] * (q3 - 32);
        dst[y + l + 96] = d * packed[scOff + sc + is + 6] * (q4 - 32);
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }

  private static int scaleMinK4(final int j, final byte[] q, final int off) {
    if (j < 4) {
      return q[off + j] & 63;
    }
    return (q[off + j + 4] & 0x0F) | (((q[off + j - 4] & 0xFF) >> 6) << 4);
  }

  private static int scaleMinK4Min(final int j, final byte[] q, final int off) {
    if (j < 4) {
      return q[off + j + 4] & 63;
    }
    return ((q[off + j + 4] & 0xFF) >> 4) | (((q[off + j] & 0xFF) >> 6) << 4);
  }

  private static long i64LE(final byte[] b, final int off) {
    return (b[off] & 0xFFL)
      | ((b[off + 1] & 0xFFL) << 8)
      | ((b[off + 2] & 0xFFL) << 16)
      | ((b[off + 3] & 0xFFL) << 24)
      | ((b[off + 4] & 0xFFL) << 32)
      | ((b[off + 5] & 0xFFL) << 40)
      | ((b[off + 6] & 0xFFL) << 48)
      | ((b[off + 7] & 0xFFL) << 56);
  }

  private static int u16LE(final byte[] b, final int off) {
    return (b[off] & 0xFF) | ((b[off + 1] & 0xFF) << 8);
  }

  private static int i32LE(final byte[] b, final int off) {
    return (b[off] & 0xFF)
      | ((b[off + 1] & 0xFF) << 8)
      | ((b[off + 2] & 0xFF) << 16)
      | ((b[off + 3] & 0xFF) << 24);
  }

  private static byte[] copyRemaining(final ByteBuffer src) {
    byte[] packed = new byte[src.remaining()];
    src.get(packed);
    return packed;
  }

  static ByteBuffer littleEndianSlice(final byte[] packed, final int position, final int length) {
    return littleEndianSlice(ByteBuffer.wrap(packed), position, length);
  }

  private static int requireInt(final long numel) {
    if (numel < 0 || numel > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("numel out of range: " + numel);
    }
    return (int) numel;
  }

  static ByteBuffer littleEndianSlice(final ByteBuffer map, final int position, final int length) {
    ByteBuffer slice = map.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    slice.position(position);
    slice.limit(position + length);
    return slice.slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  @FunctionalInterface
  private interface BlockDequant {
    void dequant(byte[] packed, int byteOff, float[] dst, int dstOff);
  }
}
