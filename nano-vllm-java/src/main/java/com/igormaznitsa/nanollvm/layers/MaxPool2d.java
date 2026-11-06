package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 2-D max pool over {@code [channels, height, width]}. Windows that fall off the edge are ignored.
 *
 * @since 1.5.0
 */
public final class MaxPool2d {

  private final int kernel;
  private final int stride;
  private final int padding;

  /**
   * @param kernel  window size ({@code >= 1})
   * @param stride  spatial stride ({@code >= 1})
   * @param padding zeros on each side ({@code >= 0})
   */
  public MaxPool2d(final int kernel, final int stride, final int padding) {
    if (kernel < 1 || stride < 1 || padding < 0) {
      throw new IllegalArgumentException("kernel and stride must be >= 1 and padding >= 0");
    }
    this.kernel = kernel;
    this.stride = stride;
    this.padding = padding;
  }

  /**
   * @param input {@code [channels, height, width]}
   * @return pooled image of the same channel count
   */
  public Tensor forward(final Tensor input) {
    requireNonNull(input, "input");
    if (input.shape().length != 3) {
      throw new IllegalArgumentException("maxpool input must be [channels, height, width]");
    }
    int channels = input.size(0);
    int height = input.size(1);
    int width = input.size(2);
    int outHeight = (height + 2 * this.padding - this.kernel) / this.stride + 1;
    int outWidth = (width + 2 * this.padding - this.kernel) / this.stride + 1;
    if (outHeight < 1 || outWidth < 1) {
      throw new IllegalArgumentException("maxpool output spatial size is < 1");
    }
    Tensor output = Tensor.zeros(channels, outHeight, outWidth);
    float[] src = input.data();
    int srcOff = input.offset();
    float[] dest = output.data();
    int kernel = this.kernel;
    int stride = this.stride;
    int pad = this.padding;
    for (int c = 0; c < channels; c++) {
      int inBase = srcOff + c * height * width;
      int outBase = c * outHeight * outWidth;
      for (int y = 0; y < outHeight; y++) {
        int originY = y * stride - pad;
        for (int x = 0; x < outWidth; x++) {
          int originX = x * stride - pad;
          float max = Float.NEGATIVE_INFINITY;
          for (int ky = 0; ky < kernel; ky++) {
            int srcY = originY + ky;
            if (srcY < 0 || srcY >= height) {
              continue;
            }
            for (int kx = 0; kx < kernel; kx++) {
              int srcX = originX + kx;
              if (srcX >= 0 && srcX < width) {
                max = Math.max(max, src[inBase + srcY * width + srcX]);
              }
            }
          }
          dest[outBase + y * outWidth + x] = max;
        }
      }
    }
    return output;
  }
}
