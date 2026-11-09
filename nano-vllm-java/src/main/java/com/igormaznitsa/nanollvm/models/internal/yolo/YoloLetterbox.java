package com.igormaznitsa.nanollvm.models.internal.yolo;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.Point2D;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;

/**
 * Letterbox to 640×640: keep aspect, pad with gray 114, store RGB in {@code [0, 1]}.
 */
final class YoloLetterbox {

  static final int SIZE = 640;
  private static final float PAD = 114f / 255f;

  final Tensor tensor;
  final int sourceWidth;
  final int sourceHeight;
  final float scale;
  final float padX;
  final float padY;

  private YoloLetterbox(
    final Tensor tensor,
    final int sourceWidth,
    final int sourceHeight,
    final float scale,
    final float padX,
    final float padY
  ) {
    this.tensor = tensor;
    this.sourceWidth = sourceWidth;
    this.sourceHeight = sourceHeight;
    this.scale = scale;
    this.padX = padX;
    this.padY = padY;
  }

  static YoloLetterbox of(final ArgbImage image) {
    return of(image, SIZE);
  }

  static YoloLetterbox of(final ArgbImage image, final int size) {
    requireNonNull(image, "image");
    if (size < 32 || size % 32 != 0) {
      throw new IllegalArgumentException("imageSize must be a multiple of 32 and at least 32");
    }
    int sourceWidth = image.width();
    int sourceHeight = image.height();
    float scale = Math.min(size / (float) sourceWidth, size / (float) sourceHeight);
    int resizedWidth = Math.max(1, Math.round(sourceWidth * scale));
    int resizedHeight = Math.max(1, Math.round(sourceHeight * scale));
    float padX = (size - resizedWidth) / 2f;
    float padY = (size - resizedHeight) / 2f;
    float[] nchw = new float[3 * size * size];
    Arrays.fill(nchw, PAD);
    int[] rgb = image.pixels();
    for (int y = 0; y < resizedHeight; y++) {
      int destY = y + (int) padY;
      float srcY = (y + 0.5f) * sourceHeight / (float) resizedHeight - 0.5f;
      int y0 = Math.clamp((int) Math.floor(srcY), 0, sourceHeight - 1);
      int y1 = Math.min(y0 + 1, sourceHeight - 1);
      float fy = Math.clamp(srcY - y0, 0f, 1f);
      for (int x = 0; x < resizedWidth; x++) {
        int destX = x + (int) padX;
        float srcX = (x + 0.5f) * sourceWidth / (float) resizedWidth - 0.5f;
        int x0 = Math.clamp((int) Math.floor(srcX), 0, sourceWidth - 1);
        int x1 = Math.min(x0 + 1, sourceWidth - 1);
        float fx = Math.clamp(srcX - x0, 0f, 1f);
        for (int channel = 0; channel < 3; channel++) {
          int shift = (2 - channel) * 8;
          float v00 = ((rgb[y0 * sourceWidth + x0] >> shift) & 0xff) / 255f;
          float v01 = ((rgb[y0 * sourceWidth + x1] >> shift) & 0xff) / 255f;
          float v10 = ((rgb[y1 * sourceWidth + x0] >> shift) & 0xff) / 255f;
          float v11 = ((rgb[y1 * sourceWidth + x1] >> shift) & 0xff) / 255f;
          float top = v00 + (v01 - v00) * fx;
          float bottom = v10 + (v11 - v10) * fx;
          nchw[(channel * size + destY) * size + destX] = top + (bottom - top) * fy;
        }
      }
    }
    return new YoloLetterbox(
      Tensor.of(nchw, 3, size, size), sourceWidth, sourceHeight, scale, padX, padY);
  }

  static YoloLetterbox direct(final Tensor tensor, final int sourceWidth, final int sourceHeight) {
    requireNonNull(tensor, "tensor");
    if (tensor.shape().length != 3 || tensor.size(0) != 3) {
      throw new IllegalArgumentException("letterbox tensor must be [3, height, width]");
    }
    return new YoloLetterbox(tensor, sourceWidth, sourceHeight, 1f, 0f, 0f);
  }

  Point2D toSource(final float letterX, final float letterY) {
    return new Point2D(
      Math.clamp((letterX - this.padX) / this.scale, 0f, this.sourceWidth),
      Math.clamp((letterY - this.padY) / this.scale, 0f, this.sourceHeight));
  }
}
