package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

/**
 * Gemma QAT pack/unpack (Hugging Face {@code gemma_quant}): int2/int4 nibbles in uint8, int8 as
 * signed bytes, then per-row or per-block float scales.
 */
public final class GemmaQat {

  private GemmaQat() {
  }

  public static int packedWidth(final int cols, final int bits) {
    if (bits == 2) {
      return (cols + 3) / 4;
    }
    if (bits == 4) {
      return (cols + 1) / 2;
    }
    if (bits == 8) {
      return cols;
    }
    throw new IllegalArgumentException("unsupported QAT bit width: " + bits);
  }

  public static int inferBits(final int packedWidth, final int cols) {
    if (packedWidth == cols) {
      return 8;
    }
    if (packedWidth == (cols + 1) / 2) {
      return 4;
    }
    if (packedWidth == (cols + 3) / 4) {
      return 2;
    }
    throw new IllegalArgumentException(
      "cannot infer QAT bits from packed width %d and unpacked cols %d".formatted(
        packedWidth, cols));
  }

  public static void unpackRow(
    final byte[] packed,
    final int packedRowOffset,
    final int packedWidth,
    final int bits,
    final float[] scales,
    final int scaleOffset,
    final int scaleCols,
    final float[] dst,
    final int cols
  ) {
    requireNonNull(packed, "packed");
    requireNonNull(dst, "dst");
    if (bits == 2) {
      unpackInt2(packed, packedRowOffset, packedWidth, dst, cols);
    } else if (bits == 4) {
      unpackInt4(packed, packedRowOffset, packedWidth, dst, cols);
    } else if (bits == 8) {
      unpackInt8(packed, packedRowOffset, dst, cols);
    } else {
      throw new IllegalArgumentException("unsupported QAT bit width: " + bits);
    }
    applyScales(dst, cols, scales, scaleOffset, scaleCols);
  }

  public static float applySrq(final float value, final float scale) {
    if (scale == 0f) {
      return value;
    }
    float quantized = Math.round(value / scale);
    return Math.clamp(quantized, -128f, 127f) * scale;
  }

  static void unpackInt4(
    final byte[] packed,
    final int rowOffset,
    final int packedWidth,
    final float[] dst,
    final int cols
  ) {
    int o = 0;
    for (int i = 0; i < packedWidth && o < cols; i++) {
      int b = packed[rowOffset + i] & 0xFF;
      dst[o++] = (b & 0x0F) - 8;
      if (o < cols) {
        dst[o++] = (b >> 4) - 8;
      }
    }
  }

  static void unpackInt2(
    final byte[] packed,
    final int rowOffset,
    final int packedWidth,
    final float[] dst,
    final int cols
  ) {
    int o = 0;
    for (int i = 0; i < packedWidth && o < cols; i++) {
      int b = packed[rowOffset + i] & 0xFF;
      dst[o++] = (b & 0x03) - 2;
      if (o < cols) {
        dst[o++] = ((b >> 2) & 0x03) - 2;
      }
      if (o < cols) {
        dst[o++] = ((b >> 4) & 0x03) - 2;
      }
      if (o < cols) {
        dst[o++] = (b >> 6) - 2;
      }
    }
  }

  static void unpackInt8(final byte[] packed, final int rowOffset, final float[] dst,
                         final int cols) {
    for (int i = 0; i < cols; i++) {
      dst[i] = packed[rowOffset + i];
    }
  }

  private static void applyScales(
    final float[] dst,
    final int cols,
    final float[] scales,
    final int scaleOffset,
    final int scaleCols
  ) {
    requireNonNull(scales, "scales");
    if (scaleCols <= 0 || cols % scaleCols != 0) {
      throw new IllegalArgumentException(
        "scaleCols %d does not divide unpacked width %d".formatted(scaleCols, cols));
    }
    int block = cols / scaleCols;
    for (int i = 0; i < cols; i++) {
      dst[i] *= scales[scaleOffset + i / block];
    }
  }
}
