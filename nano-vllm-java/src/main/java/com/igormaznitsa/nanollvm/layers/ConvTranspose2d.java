package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 2-D transposed convolution. Weight layout is PyTorch {@code [inChannels, outChannels, kernel,
 * kernel]}.
 *
 * @since 1.5.0
 */
public final class ConvTranspose2d {

  private final Tensor weight;
  private final Tensor bias;
  private final int stride;
  private final int padding;
  private final int inChannels;
  private final int outChannels;
  private final int kernel;

  /**
   * @param weight  {@code [inChannels, outChannels, kernel, kernel]}
   * @param bias    length {@code outChannels}, or {@code null}
   * @param stride  spatial stride ({@code >= 1})
   * @param padding zeros removed from each side ({@code >= 0})
   */
  public ConvTranspose2d(
    final Tensor weight,
    final Tensor bias,
    final int stride,
    final int padding
  ) {
    this.weight = requireNonNull(weight, "weight");
    this.bias = bias;
    if (weight.shape().length != 4 || weight.size(2) != weight.size(3)) {
      throw new IllegalArgumentException(
        "convtranspose2d weight must be [in, out, kernel, kernel]");
    }
    if (stride < 1 || padding < 0) {
      throw new IllegalArgumentException("stride must be >= 1 and padding >= 0");
    }
    this.stride = stride;
    this.padding = padding;
    this.inChannels = weight.size(0);
    this.outChannels = weight.size(1);
    this.kernel = weight.size(2);
    if (bias != null && bias.numel() != this.outChannels) {
      throw new IllegalArgumentException("convtranspose2d bias length must equal outChannels");
    }
  }

  /**
   * @param input {@code [inChannels, height, width]}
   * @return {@code [outChannels, outHeight, outWidth]}
   */
  public Tensor forward(final Tensor input) {
    requireNonNull(input, "input");
    if (input.shape().length != 3 || input.size(0) != this.inChannels) {
      throw new IllegalArgumentException(
        "convtranspose2d input must be [inChannels, height, width]");
    }
    int height = input.size(1);
    int width = input.size(2);
    int outHeight = (height - 1) * this.stride - 2 * this.padding + this.kernel;
    int outWidth = (width - 1) * this.stride - 2 * this.padding + this.kernel;
    if (outHeight < 1 || outWidth < 1) {
      throw new IllegalArgumentException("convtranspose2d output spatial size is < 1");
    }
    Tensor output = Tensor.zeros(this.outChannels, outHeight, outWidth);
    float[] weights = this.weight.data();
    int weightOff = this.weight.offset();
    float[] src = input.data();
    int srcOff = input.offset();
    float[] dest = output.data();
    float[] biasData = this.bias == null ? null : this.bias.data();
    int biasOff = this.bias == null ? 0 : this.bias.offset();
    if (biasData != null) {
      for (int oc = 0; oc < this.outChannels; oc++) {
        float b = biasData[biasOff + oc];
        int base = oc * outHeight * outWidth;
        for (int i = 0; i < outHeight * outWidth; i++) {
          dest[base + i] = b;
        }
      }
    }
    int kernel = this.kernel;
    int stride = this.stride;
    int pad = this.padding;
    int weightChannel = this.outChannels * kernel * kernel;
    for (int ic = 0; ic < this.inChannels; ic++) {
      int inBase = srcOff + ic * height * width;
      int wBase = weightOff + ic * weightChannel;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          float value = src[inBase + y * width + x];
          if (value == 0f) {
            continue;
          }
          for (int oc = 0; oc < this.outChannels; oc++) {
            int wRow = wBase + oc * kernel * kernel;
            int outBase = oc * outHeight * outWidth;
            for (int ky = 0; ky < kernel; ky++) {
              int dstY = y * stride - pad + ky;
              if (dstY < 0 || dstY >= outHeight) {
                continue;
              }
              for (int kx = 0; kx < kernel; kx++) {
                int dstX = x * stride - pad + kx;
                if (dstX >= 0 && dstX < outWidth) {
                  dest[outBase + dstY * outWidth + dstX] +=
                    value * weights[wRow + ky * kernel + kx];
                }
              }
            }
          }
        }
      }
    }
    return output;
  }
}
