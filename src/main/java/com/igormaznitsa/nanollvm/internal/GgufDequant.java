package com.igormaznitsa.nanollvm.internal;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Dequantizes GGML block layouts into float32 (ported from ggml row dequant kernels).
 */
public final class GgufDequant {

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
      case 0 -> readF32(src, numel);
      case 1 -> readF16(src, numel);
      case 2 -> dequantQ4_0(src, numel);
      case 8 -> dequantQ8_0(src, numel);
      case 12 -> dequantQ4_K(src, numel);
      case 14 -> dequantQ6_K(src, numel);
      case 30 -> readBf16(src, numel);
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    };
  }

  public static int typeBlockSize(final int ggmlType) {
    return switch (ggmlType) {
      case 0 -> 4;
      case 1, 30 -> 2;
      case 2 -> BLOCK_Q4_0;
      case 8 -> BLOCK_Q8_0;
      case 12 -> BLOCK_Q4_K;
      case 14 -> BLOCK_Q6_K;
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    };
  }

  public static int typeBlockElems(final int ggmlType) {
    return switch (ggmlType) {
      case 0, 1, 30 -> 1;
      case 2 -> QK4_0;
      case 8 -> QK8_0;
      case 12, 14 -> QK_K;
      default -> throw new UnsupportedOperationException("unsupported GGML type " + ggmlType);
    };
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

  private static float[] dequantQ4_0(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK4_0 != 0) {
      throw new IllegalArgumentException("Q4_0 numel must be multiple of " + QK4_0);
    }
    float[] out = new float[n];
    int nb = n / QK4_0;
    for (int i = 0; i < nb; i++) {
      float d = Float.float16ToFloat(src.getShort());
      for (int j = 0; j < QK4_0 / 2; j++) {
        int qs = src.get() & 0xFF;
        out[i * QK4_0 + j] = ((qs & 0x0F) - 8) * d;
        out[i * QK4_0 + j + QK4_0 / 2] = ((qs >> 4) - 8) * d;
      }
    }
    return out;
  }

  private static float[] dequantQ8_0(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK8_0 != 0) {
      throw new IllegalArgumentException("Q8_0 numel must be multiple of " + QK8_0);
    }
    float[] out = new float[n];
    int nb = n / QK8_0;
    for (int i = 0; i < nb; i++) {
      float d = Float.float16ToFloat(src.getShort());
      for (int j = 0; j < QK8_0; j++) {
        out[i * QK8_0 + j] = src.get() * d;
      }
    }
    return out;
  }

  private static float[] dequantQ4_K(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK_K != 0) {
      throw new IllegalArgumentException("Q4_K numel must be multiple of " + QK_K);
    }
    float[] out = new float[n];
    int nb = n / QK_K;
    byte[] scales = new byte[12];
    byte[] qs = new byte[QK_K / 2];
    int y = 0;
    for (int i = 0; i < nb; i++) {
      float d = Float.float16ToFloat(src.getShort());
      float min = Float.float16ToFloat(src.getShort());
      src.get(scales);
      src.get(qs);
      int qOff = 0;
      int is = 0;
      for (int j = 0; j < QK_K; j += 64) {
        int sc0 = scaleMinK4(is, scales);
        int m0 = scaleMinK4Min(is, scales);
        float d1 = d * sc0;
        float m1 = min * m0;
        int sc1 = scaleMinK4(is + 1, scales);
        int m1b = scaleMinK4Min(is + 1, scales);
        float d2 = d * sc1;
        float m2 = min * m1b;
        for (int l = 0; l < 32; l++) {
          out[y++] = d1 * (qs[qOff + l] & 0x0F) - m1;
        }
        for (int l = 0; l < 32; l++) {
          out[y++] = d2 * ((qs[qOff + l] & 0xFF) >> 4) - m2;
        }
        qOff += 32;
        is += 2;
      }
    }
    return out;
  }

  private static float[] dequantQ6_K(final ByteBuffer src, final long numel) {
    int n = requireInt(numel);
    if (n % QK_K != 0) {
      throw new IllegalArgumentException("Q6_K numel must be multiple of " + QK_K);
    }
    float[] out = new float[n];
    int nb = n / QK_K;
    byte[] ql = new byte[QK_K / 2];
    byte[] qh = new byte[QK_K / 4];
    byte[] sc = new byte[QK_K / 16];
    int y = 0;
    for (int i = 0; i < nb; i++) {
      src.get(ql);
      src.get(qh);
      src.get(sc);
      float d = Float.float16ToFloat(src.getShort());
      int qlOff = 0;
      int qhOff = 0;
      int scOff = 0;
      for (int block = 0; block < QK_K; block += 128) {
        for (int l = 0; l < 32; l++) {
          int is = l / 16;
          int q1 = (ql[qlOff + l] & 0x0F) | ((qh[qhOff + l] & 3) << 4);
          int q2 = (ql[qlOff + l + 32] & 0x0F) | (((qh[qhOff + l] >> 2) & 3) << 4);
          int q3 = ((ql[qlOff + l] & 0xFF) >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4);
          int q4 = ((ql[qlOff + l + 32] & 0xFF) >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4);
          out[y + l] = d * sc[scOff + is] * (q1 - 32);
          out[y + l + 32] = d * sc[scOff + is + 2] * (q2 - 32);
          out[y + l + 64] = d * sc[scOff + is + 4] * (q3 - 32);
          out[y + l + 96] = d * sc[scOff + is + 6] * (q4 - 32);
        }
        y += 128;
        qlOff += 64;
        qhOff += 32;
        scOff += 8;
      }
    }
    return out;
  }

  private static int scaleMinK4(final int j, final byte[] q) {
    if (j < 4) {
      return q[j] & 63;
    }
    return (q[j + 4] & 0x0F) | (((q[j - 4] & 0xFF) >> 6) << 4);
  }

  private static int scaleMinK4Min(final int j, final byte[] q) {
    if (j < 4) {
      return q[j + 4] & 63;
    }
    return ((q[j + 4] & 0xFF) >> 4) | (((q[j] & 0xFF) >> 6) << 4);
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
}
