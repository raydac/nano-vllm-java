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
  public static final int TYPE_Q8_0 = 8;
  public static final int TYPE_Q4_K = 12;
  public static final int TYPE_Q6_K = 14;
  public static final int TYPE_BF16 = 30;

  public static final int QK4_0 = 32;
  public static final int QK8_0 = 32;
  public static final int QK_K = 256;
  public static final int BLOCK_Q4_0 = 2 + QK4_0 / 2;
  public static final int BLOCK_Q8_0 = 2 + QK8_0;
  public static final int BLOCK_Q4_K = 2 + 2 + 12 + QK_K / 2;
  public static final int BLOCK_Q6_K = 2 + QK_K / 16 + 3 * QK_K / 4;

  private GgufDequant() {
  }

  public static float[] dequantize(final ByteBuffer src, final int ggmlType, final long numel) {
    return switch (ggmlType) {
      case TYPE_F32 -> readF32(src, numel);
      case TYPE_F16 -> readF16(src, numel);
      case TYPE_Q4_0 -> dequantQ4_0(src, numel);
      case TYPE_Q8_0 -> dequantQ8_0(src, numel);
      case TYPE_Q4_K -> dequantQ4_K(src, numel);
      case TYPE_Q6_K -> dequantQ6_K(src, numel);
      case TYPE_BF16 -> readBf16(src, numel);
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    };
  }

  public static int typeBlockSize(final int ggmlType) {
    return switch (ggmlType) {
      case TYPE_F32 -> 4;
      case TYPE_F16, TYPE_BF16 -> 2;
      case TYPE_Q4_0 -> BLOCK_Q4_0;
      case TYPE_Q8_0 -> BLOCK_Q8_0;
      case TYPE_Q4_K -> BLOCK_Q4_K;
      case TYPE_Q6_K -> BLOCK_Q6_K;
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    };
  }

  public static int typeBlockElems(final int ggmlType) {
    return switch (ggmlType) {
      case TYPE_F32, TYPE_F16, TYPE_BF16 -> 1;
      case TYPE_Q4_0 -> QK4_0;
      case TYPE_Q8_0 -> QK8_0;
      case TYPE_Q4_K, TYPE_Q6_K -> QK_K;
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
      case TYPE_F32, TYPE_F16, TYPE_BF16 ->
        dequantizeUnitStride(packed, ggmlType, elemStart, elemCount, dst, dstOff);
      case TYPE_Q4_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK4_0, BLOCK_Q4_0, GgufDequant::dequantOneQ4_0);
      case TYPE_Q8_0 -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK8_0, BLOCK_Q8_0, GgufDequant::dequantOneQ8_0);
      case TYPE_Q4_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_Q4_K, GgufDequant::dequantOneQ4_K);
      case TYPE_Q6_K -> dequantizeBlockedRange(packed, numel, elemStart, elemCount, dst, dstOff,
        QK_K, BLOCK_Q6_K, GgufDequant::dequantOneQ6_K);
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
      default -> throw new UnsupportedOperationException("unit-stride type " + ggmlType);
    }
  }

  private static float[] dequantQ4_0(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK4_0 != 0) {
      throw new IllegalArgumentException("Q4_0 numel must be multiple of " + QK4_0);
    }
    byte[] packed = copyRemaining(src);
    float[] out = new float[n];
    dequantizeRange(packed, TYPE_Q4_0, n, 0, n, out, 0);
    return out;
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

  private static float[] dequantQ8_0(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK8_0 != 0) {
      throw new IllegalArgumentException("Q8_0 numel must be multiple of " + QK8_0);
    }
    byte[] packed = copyRemaining(src);
    float[] out = new float[n];
    dequantizeRange(packed, TYPE_Q8_0, n, 0, n, out, 0);
    return out;
  }

  private static float[] dequantQ4_K(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK_K != 0) {
      throw new IllegalArgumentException("Q4_K numel must be multiple of " + QK_K);
    }
    byte[] packed = copyRemaining(src);
    float[] out = new float[n];
    dequantizeRange(packed, TYPE_Q4_K, n, 0, n, out, 0);
    return out;
  }

  private static float[] dequantQ6_K(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK_K != 0) {
      throw new IllegalArgumentException("Q6_K numel must be multiple of " + QK_K);
    }
    byte[] packed = copyRemaining(src);
    float[] out = new float[n];
    dequantizeRange(packed, TYPE_Q6_K, n, 0, n, out, 0);
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
