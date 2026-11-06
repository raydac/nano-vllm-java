package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Channel scaling and elementwise steps shared by the YOLO26 blocks.
 */
final class YoloMath {

  private YoloMath() {
  }

  static int channels(final int nominal, final float width, final int maxChannels) {
    int scaled = (int) (Math.min(nominal, maxChannels) * width);
    return (int) Math.ceil(scaled / 8.0) * 8;
  }

  static int repeats(final int count, final float depth) {
    return count > 1 ? Math.max(1, Math.round(count * depth)) : count;
  }

  static Tensor silu(final Tensor input) {
    float[] src = input.data();
    int off = input.offset();
    float[] dest = new float[input.numel()];
    for (int i = 0; i < dest.length; i++) {
      float x = src[off + i];
      dest[i] = x / (1f + (float) Math.exp(-x));
    }
    return Tensor.of(dest, input.shape());
  }

  static Tensor sigmoid(final Tensor input) {
    float[] src = input.data();
    int off = input.offset();
    float[] dest = new float[input.numel()];
    for (int i = 0; i < dest.length; i++) {
      dest[i] = 1f / (1f + (float) Math.exp(-src[off + i]));
    }
    return Tensor.of(dest, input.shape());
  }

  static Tensor add(final Tensor left, final Tensor right) {
    requireSame(left, right);
    float[] a = left.data();
    float[] b = right.data();
    int aOff = left.offset();
    int bOff = right.offset();
    float[] dest = new float[left.numel()];
    for (int i = 0; i < dest.length; i++) {
      dest[i] = a[aOff + i] + b[bOff + i];
    }
    return Tensor.of(dest, left.shape());
  }

  static Tensor concatChannels(final Tensor... parts) {
    int channels = 0;
    int height = parts[0].size(1);
    int width = parts[0].size(2);
    for (Tensor part : parts) {
      if (part.size(1) != height || part.size(2) != width) {
        throw new IllegalArgumentException("concat spatial sizes differ");
      }
      channels += part.size(0);
    }
    float[] dest = new float[channels * height * width];
    int written = 0;
    for (Tensor part : parts) {
      int count = part.numel();
      System.arraycopy(part.data(), part.offset(), dest, written, count);
      written += count;
    }
    return Tensor.of(dest, channels, height, width);
  }

  static Tensor sliceChannels(final Tensor input, final int from, final int count) {
    int height = input.size(1);
    int width = input.size(2);
    float[] dest = new float[count * height * width];
    System.arraycopy(
      input.data(),
      input.offset() + from * height * width,
      dest,
      0,
      dest.length);
    return Tensor.of(dest, count, height, width);
  }

  static Tensor nearestTwice(final Tensor input) {
    int channels = input.size(0);
    int height = input.size(1);
    int width = input.size(2);
    int outH = height * 2;
    int outW = width * 2;
    float[] src = input.data();
    int off = input.offset();
    float[] dest = new float[channels * outH * outW];
    for (int c = 0; c < channels; c++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          float value = src[off + (c * height + y) * width + x];
          int base = (c * outH + y * 2) * outW + x * 2;
          dest[base] = value;
          dest[base + 1] = value;
          dest[base + outW] = value;
          dest[base + outW + 1] = value;
        }
      }
    }
    return Tensor.of(dest, channels, outH, outW);
  }

  static Tensor bilinear(
    final Tensor input,
    final int outHeight,
    final int outWidth,
    final boolean alignCorners
  ) {
    int channels = input.size(0);
    int height = input.size(1);
    int width = input.size(2);
    float[] src = input.data();
    int off = input.offset();
    float[] dest = new float[channels * outHeight * outWidth];
    float scaleY = alignCorners && outHeight > 1
      ? (height - 1f) / (outHeight - 1f)
      : height / (float) outHeight;
    float scaleX = alignCorners && outWidth > 1
      ? (width - 1f) / (outWidth - 1f)
      : width / (float) outWidth;
    for (int c = 0; c < channels; c++) {
      for (int y = 0; y < outHeight; y++) {
        float srcY = alignCorners ? y * scaleY : (y + 0.5f) * scaleY - 0.5f;
        int y0 = (int) Math.floor(srcY);
        int y1 = Math.min(y0 + 1, height - 1);
        y0 = Math.clamp(y0, 0, height - 1);
        float fy = srcY - (float) Math.floor(srcY);
        for (int x = 0; x < outWidth; x++) {
          float srcX = alignCorners ? x * scaleX : (x + 0.5f) * scaleX - 0.5f;
          int x0 = (int) Math.floor(srcX);
          int x1 = Math.min(x0 + 1, width - 1);
          x0 = Math.clamp(x0, 0, width - 1);
          float fx = srcX - (float) Math.floor(srcX);
          float v00 = src[off + (c * height + y0) * width + x0];
          float v01 = src[off + (c * height + y0) * width + x1];
          float v10 = src[off + (c * height + y1) * width + x0];
          float v11 = src[off + (c * height + y1) * width + x1];
          float top = v00 + (v01 - v00) * fx;
          float bottom = v10 + (v11 - v10) * fx;
          dest[(c * outHeight + y) * outWidth + x] = top + (bottom - top) * fy;
        }
      }
    }
    return Tensor.of(dest, channels, outHeight, outWidth);
  }

  static Tensor nearestScale(final Tensor input, final int scale) {
    Tensor current = input;
    int left = scale;
    while (left > 1) {
      current = nearestTwice(current);
      left /= 2;
    }
    return current;
  }

  private static void requireSame(final Tensor left, final Tensor right) {
    int[] a = left.shape();
    int[] b = right.shape();
    if (a.length != b.length) {
      throw new IllegalArgumentException("tensor ranks differ");
    }
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) {
        throw new IllegalArgumentException("tensor shapes differ");
      }
    }
  }
}
