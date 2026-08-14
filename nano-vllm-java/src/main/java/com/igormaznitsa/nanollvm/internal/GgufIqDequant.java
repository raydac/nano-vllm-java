package com.igormaznitsa.nanollvm.internal;

import static com.igormaznitsa.nanollvm.internal.GgufDequant.QK_K;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.iq1s_grid;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.iq2s_grid;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.iq2xs_grid;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.iq2xxs_grid;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.iq3s_grid;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.iq3xxs_grid;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.kmask_iq2xs;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.ksigns_iq2xs;
import static com.igormaznitsa.nanollvm.internal.GgufIqTables.kvalues_fp4;

/**
 * GGML IQ / ternary / MXFP / NVFP / Q1_0 / Q2_0 row dequant (llama.cpp {@code ggml-quants.c}).
 */
final class GgufIqDequant {

  private static final float IQ1S_DELTA = 0.125f;
  private static final int[] KVALUES_IQ4NL = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
  };
  private static final int[] POW3 = {1, 3, 9, 27, 81, 243};

  private GgufIqDequant() {
  }

  static void dequantOneQ1_0(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int qsOff = byteOff + 2;
    for (int j = 0; j < GgufDequant.QK1_0; j++) {
      int bit = (packed[qsOff + j / 8] >> (j % 8)) & 1;
      dst[dstOff + j] = bit != 0 ? d : -d;
    }
  }

  static void dequantOneQ2_0(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int qsOff = byteOff + 2;
    for (int j = 0; j < GgufDequant.QK2_0; j++) {
      int q = (packed[qsOff + j / 4] >> ((j % 4) * 2)) & 3;
      dst[dstOff + j] = (q - 1) * d;
    }
  }

  static void dequantOneMxfp4(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = e8m0Half(packed[byteOff] & 0xFF);
    int qsOff = byteOff + 1;
    for (int j = 0; j < GgufDequant.QK_MXFP4 / 2; j++) {
      int qs = packed[qsOff + j] & 0xFF;
      dst[dstOff + j] = kvalues_fp4[qs & 0x0F] * d;
      dst[dstOff + j + GgufDequant.QK_MXFP4 / 2] = kvalues_fp4[qs >> 4] * d;
    }
  }

  static void dequantOneNvfp4(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    int qsOff = byteOff + 4;
    for (int s = 0; s < 4; s++) {
      float d = ue4m3ToFp32(packed[byteOff + s] & 0xFF);
      int y = dstOff + s * GgufDequant.QK_NVFP4_SUB;
      int q = qsOff + s * (GgufDequant.QK_NVFP4_SUB / 2);
      for (int j = 0; j < GgufDequant.QK_NVFP4_SUB / 2; j++) {
        int qs = packed[q + j] & 0xFF;
        dst[y + j] = kvalues_fp4[qs & 0x0F] * d;
        dst[y + j + GgufDequant.QK_NVFP4_SUB / 2] = kvalues_fp4[qs >> 4] * d;
      }
    }
  }

  static void dequantOneTq1_0(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff + 52);
    int y = dstOff;
    for (int j = 0; j < 32; j += 32) {
      for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 32; m++) {
          int q = ((packed[byteOff + j + m] & 0xFF) * POW3[n]) & 0xFF;
          int xi = (q * 3) >> 8;
          dst[y++] = (xi - 1) * d;
        }
      }
    }
    for (int j = 32; j < 48; j += 16) {
      for (int n = 0; n < 5; n++) {
        for (int m = 0; m < 16; m++) {
          int q = ((packed[byteOff + j + m] & 0xFF) * POW3[n]) & 0xFF;
          int xi = (q * 3) >> 8;
          dst[y++] = (xi - 1) * d;
        }
      }
    }
    int qhOff = byteOff + 48;
    for (int n = 0; n < 4; n++) {
      for (int j = 0; j < 4; j++) {
        int q = ((packed[qhOff + j] & 0xFF) * POW3[n]) & 0xFF;
        int xi = (q * 3) >> 8;
        dst[y++] = (xi - 1) * d;
      }
    }
  }

  static void dequantOneTq2_0(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff + QK_K / 4);
    int y = dstOff;
    for (int j = 0; j < QK_K / 4; j += 32) {
      for (int l = 0; l < 4; l++) {
        for (int m = 0; m < 32; m++) {
          int q = (packed[byteOff + j + m] >> (l * 2)) & 3;
          dst[y++] = (q - 1) * d;
        }
      }
    }
  }

  static void dequantOneIq2Xxs(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int y = dstOff;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
      int qs = byteOff + 2 + ib32 * 8;
      int aux1 = i32LE(packed, qs + 4);
      float db = d * (0.5f + (aux1 >>> 28)) * 0.25f;
      for (int l = 0; l < 4; l++) {
        applySignedGrid8(iq2xxs_grid[packed[qs + l] & 0xFF], db,
          ksigns_iq2xs[(aux1 >>> (7 * l)) & 127],
          dst, y);
        y += 8;
      }
    }
  }

  static void dequantOneIq2Xs(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int qsOff = byteOff + 2;
    int scalesOff = qsOff + QK_K / 8 * 2;
    int y = dstOff;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
      int sc = packed[scalesOff + ib32] & 0xFF;
      float db0 = d * (0.5f + (sc & 0x0F)) * 0.25f;
      float db1 = d * (0.5f + (sc >> 4)) * 0.25f;
      for (int l = 0; l < 4; l++) {
        int q = u16LE(packed, qsOff + (4 * ib32 + l) * 2);
        applySignedGrid8(iq2xs_grid[q & 511], l < 2 ? db0 : db1, ksigns_iq2xs[q >>> 9], dst, y);
        y += 8;
      }
    }
  }

  static void dequantOneIq2S(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int qsOff = byteOff + 2;
    int qhOff = qsOff + QK_K / 4;
    int scalesOff = qhOff + QK_K / 32;
    int signsOff = qsOff + QK_K / 8;
    int y = dstOff;
    int qs = qsOff;
    int signs = signsOff;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
      int sc = packed[scalesOff + ib32] & 0xFF;
      float db0 = d * (0.5f + (sc & 0x0F)) * 0.25f;
      float db1 = d * (0.5f + (sc >> 4)) * 0.25f;
      int qh = packed[qhOff + ib32] & 0xFF;
      for (int l = 0; l < 4; l++) {
        int idx = (packed[qs + l] & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
        applySignedGrid8(iq2s_grid[idx], l < 2 ? db0 : db1, packed[signs + l], dst, y);
        y += 8;
      }
      qs += 4;
      signs += 4;
    }
  }

  static void dequantOneIq3Xxs(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int qsOff = byteOff + 2;
    int scaleOff = qsOff + QK_K / 4;
    int y = dstOff;
    int qs = qsOff;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
      int aux = i32LE(packed, scaleOff + 4 * ib32);
      float db = d * (0.5f + (aux >>> 28)) * 0.5f;
      for (int l = 0; l < 4; l++) {
        int signs = ksigns_iq2xs[(aux >>> (7 * l)) & 127] & 0xFF;
        applyIq3Pair(iq3xxs_grid[packed[qs + 2 * l] & 0xFF],
          iq3xxs_grid[packed[qs + 2 * l + 1] & 0xFF],
          db, signs, dst, y);
        y += 8;
      }
      qs += 8;
    }
  }

  static void dequantOneIq3S(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int qsOff = byteOff + 2;
    int qhOff = qsOff + QK_K / 4;
    int signsOff = qhOff + QK_K / 32;
    int scalesOff = signsOff + QK_K / 8;
    int y = dstOff;
    int qs = qsOff;
    int signs = signsOff;
    int qh = qhOff;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      int sc = packed[scalesOff + ib32 / 2] & 0xFF;
      float db1 = d * (1 + 2 * (sc & 0x0F));
      float db2 = d * (1 + 2 * (sc >> 4));
      y = emitIq3SGroup(packed, qs, qh, signs, db1, dst, y);
      qs += 8;
      signs += 4;
      y = emitIq3SGroup(packed, qs, qh + 1, signs, db2, dst, y);
      qh += 2;
      qs += 8;
      signs += 4;
    }
  }

  static void dequantOneIq1S(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int qsOff = byteOff + 2;
    int qhOff = qsOff + QK_K / 8;
    int y = dstOff;
    int qs = qsOff;
    for (int ib = 0; ib < QK_K / 32; ib++) {
      int qh = u16LE(packed, qhOff + ib * 2);
      float dl = d * (2 * ((qh >>> 12) & 7) + 1);
      float delta = (qh & 0x8000) != 0 ? -IQ1S_DELTA : IQ1S_DELTA;
      for (int l = 0; l < 4; l++) {
        int idx = (packed[qs + l] & 0xFF) | (((qh >>> (3 * l)) & 7) << 8);
        applyIq1Grid(iq1s_grid[idx], dl, delta, dst, y);
        y += 8;
      }
      qs += 4;
    }
  }

  static void dequantOneIq1M(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    int qsOff = byteOff;
    int qhOff = qsOff + QK_K / 8;
    int scOff = qhOff + QK_K / 16;
    int scaleBits = (u16LE(packed, scOff) >>> 12)
      | ((u16LE(packed, scOff + 2) >>> 8) & 0x00F0)
      | ((u16LE(packed, scOff + 4) >>> 4) & 0x0F00)
      | (u16LE(packed, scOff + 6) & 0xF000);
    float d = Float.float16ToFloat((short) scaleBits);
    int y = dstOff;
    int qs = qsOff;
    int qh = qhOff;
    for (int ib = 0; ib < QK_K / 32; ib++) {
      int sc = u16LE(packed, scOff + (ib / 2) * 2);
      int shift = 6 * (ib % 2);
      float dl1 = d * (2 * ((sc >>> shift) & 7) + 1);
      float dl2 = d * (2 * ((sc >>> (shift + 3)) & 7) + 1);
      int qh0 = packed[qh] & 0xFF;
      int qh1 = packed[qh + 1] & 0xFF;
      int[] idx = {
        (packed[qs] & 0xFF) | ((qh0 << 8) & 0x700),
        (packed[qs + 1] & 0xFF) | ((qh0 << 4) & 0x700),
        (packed[qs + 2] & 0xFF) | ((qh1 << 8) & 0x700),
        (packed[qs + 3] & 0xFF) | ((qh1 << 4) & 0x700)
      };
      float[] delta = {
        (qh0 & 0x08) != 0 ? -IQ1S_DELTA : IQ1S_DELTA,
        (qh0 & 0x80) != 0 ? -IQ1S_DELTA : IQ1S_DELTA,
        (qh1 & 0x08) != 0 ? -IQ1S_DELTA : IQ1S_DELTA,
        (qh1 & 0x80) != 0 ? -IQ1S_DELTA : IQ1S_DELTA
      };
      for (int l = 0; l < 2; l++) {
        applyIq1Grid(iq1s_grid[idx[l]], dl1, delta[l], dst, y);
        y += 8;
      }
      for (int l = 2; l < 4; l++) {
        applyIq1Grid(iq1s_grid[idx[l]], dl2, delta[l], dst, y);
        y += 8;
      }
      qs += 4;
      qh += 2;
    }
  }

  static void dequantOneIq4Xs(
    final byte[] packed, final int byteOff, final float[] dst, final int dstOff
  ) {
    float d = fp16(packed, byteOff);
    int scalesH = u16LE(packed, byteOff + 2);
    int scalesL = byteOff + 4;
    int qsOff = byteOff + 8;
    int y = dstOff;
    int qs = qsOff;
    for (int ib = 0; ib < QK_K / 32; ib++) {
      int ls =
        ((packed[scalesL + ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scalesH >> (2 * ib)) & 3) << 4);
      float dl = d * (ls - 32);
      for (int j = 0; j < 16; j++) {
        int q = packed[qs + j] & 0xFF;
        dst[y + j] = dl * KVALUES_IQ4NL[q & 0x0F];
        dst[y + j + 16] = dl * KVALUES_IQ4NL[q >> 4];
      }
      y += 32;
      qs += 16;
    }
  }

  private static int emitIq3SGroup(
    final byte[] packed,
    final int qs,
    final int qhOff,
    final int signsOff,
    final float db,
    final float[] dst,
    int y
  ) {
    int qh = packed[qhOff] & 0xFF;
    for (int l = 0; l < 4; l++) {
      int idx1 = (packed[qs + 2 * l] & 0xFF) | ((qh << (8 - 2 * l)) & 256);
      int idx2 = (packed[qs + 2 * l + 1] & 0xFF) | ((qh << (7 - 2 * l)) & 256);
      applyIq3Pair(iq3s_grid[idx1], iq3s_grid[idx2], db, packed[signsOff + l] & 0xFF, dst, y);
      y += 8;
    }
    return y;
  }

  private static void applySignedGrid8(
    final long grid,
    final float scale,
    final int signs,
    final float[] dst,
    final int dstOff
  ) {
    int s = signs & 0xFF;
    for (int j = 0; j < 8; j++) {
      int g = (int) ((grid >>> (8 * j)) & 0xFF);
      dst[dstOff + j] = scale * g * ((s & (kmask_iq2xs[j] & 0xFF)) != 0 ? -1f : 1f);
    }
  }

  private static void applyIq3Pair(
    final int grid1,
    final int grid2,
    final float scale,
    final int signs,
    final float[] dst,
    final int dstOff
  ) {
    for (int j = 0; j < 4; j++) {
      dst[dstOff + j] = scale * ((grid1 >>> (8 * j)) & 0xFF)
        * ((signs & (kmask_iq2xs[j] & 0xFF)) != 0 ? -1f : 1f);
      dst[dstOff + j + 4] = scale * ((grid2 >>> (8 * j)) & 0xFF)
        * ((signs & (kmask_iq2xs[j + 4] & 0xFF)) != 0 ? -1f : 1f);
    }
  }

  private static void applyIq1Grid(
    final long grid,
    final float scale,
    final float delta,
    final float[] dst,
    final int dstOff
  ) {
    for (int j = 0; j < 8; j++) {
      dst[dstOff + j] = scale * (((byte) ((grid >>> (8 * j)) & 0xFF)) + delta);
    }
  }

  private static float fp16(final byte[] packed, final int off) {
    return Float.float16ToFloat((short) u16LE(packed, off));
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

  static float e8m0Half(final int e) {
    int bits = e < 2 ? (0x00200000 << e) : (e - 1) << 23;
    return Float.intBitsToFloat(bits);
  }

  static float ue4m3ToFp32(final int x) {
    if (x == 0 || x == 0x7F) {
      return 0f;
    }
    int exp = (x >>> 3) & 0xF;
    int man = x & 0x7;
    float raw = exp == 0
      ? Math.scalb((float) man, -9)
      : Math.scalb(1.0f + man / 8.0f, exp - 7);
    return raw * 0.5f;
  }
}
