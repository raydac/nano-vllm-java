package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 2-D convolution over a channel-major image {@code [inChannels, height, width]}. Weight layout is
 * PyTorch {@code [outChannels, inChannels/groups, kernel, kernel]}. Edges use zero padding.
 *
 * @since 1.5.0
 */
public final class Conv2d {

  private final Tensor weight;
  private final Tensor bias;
  private final int stride;
  private final int padding;
  private final int groups;
  private final int outChannels;
  private final int inChannels;
  private final int kernel;

  /**
   * @param weight  {@code [outChannels, inChannels/groups, kernel, kernel]}
   * @param bias    length {@code outChannels}, or {@code null}
   * @param stride  spatial stride ({@code >= 1})
   * @param padding zeros on each side ({@code >= 0})
   * @param groups  channel groups ({@code >= 1})
   */
  public Conv2d(
    final Tensor weight,
    final Tensor bias,
    final int stride,
    final int padding,
    final int groups
  ) {
    this.weight = requireNonNull(weight, "weight");
    this.bias = bias;
    if (weight.shape().length != 4 || weight.size(2) != weight.size(3)) {
      throw new IllegalArgumentException("conv2d weight must be [out, in/groups, kernel, kernel]");
    }
    if (stride < 1 || padding < 0 || groups < 1) {
      throw new IllegalArgumentException("stride and groups must be >= 1 and padding >= 0");
    }
    this.stride = stride;
    this.padding = padding;
    this.groups = groups;
    this.outChannels = weight.size(0);
    this.inChannels = weight.size(1) * groups;
    this.kernel = weight.size(2);
    if (this.outChannels % groups != 0) {
      throw new IllegalArgumentException("outChannels must be divisible by groups");
    }
    if (bias != null && bias.numel() != this.outChannels) {
      throw new IllegalArgumentException("conv2d bias length must equal outChannels");
    }
  }

  /**
   * Applies the filter on the calling thread.
   *
   * @param input {@code [inChannels, height, width]}
   * @return {@code [outChannels, outHeight, outWidth]}
   */
  public Tensor forward(final Tensor input) {
    return this.forward(input, MatmulRuntime.sequential());
  }

  /**
   * Applies the filter, splitting output channels across {@code runtime}.
   *
   * @param input   {@code [inChannels, height, width]}
   * @param runtime dense kernel runtime; {@code null} is sequential
   * @return {@code [outChannels, outHeight, outWidth]}
   */
  public Tensor forward(final Tensor input, final MatmulRuntime runtime) {
    requireNonNull(input, "input");
    if (input.shape().length != 3 || input.size(0) != this.inChannels) {
      throw new IllegalArgumentException(
        "conv2d input must be [inChannels, height, width], inChannels=" + this.inChannels);
    }
    int height = input.size(1);
    int width = input.size(2);
    int outHeight = (height + 2 * this.padding - this.kernel) / this.stride + 1;
    int outWidth = (width + 2 * this.padding - this.kernel) / this.stride + 1;
    if (outHeight < 1 || outWidth < 1) {
      throw new IllegalArgumentException("conv2d output spatial size is < 1");
    }
    Tensor output = Tensor.zeros(this.outChannels, outHeight, outWidth);
    float[] weights = this.weight.data();
    int weightOff = this.weight.offset();
    float[] activations = input.data();
    int inputOff = input.offset();
    float[] dest = output.data();
    float[] biasData = this.bias == null ? null : this.bias.data();
    int biasOff = this.bias == null ? 0 : this.bias.offset();
    int inPerGroup = this.inChannels / this.groups;
    int outPerGroup = this.outChannels / this.groups;
    int kernel = this.kernel;
    int stride = this.stride;
    int pad = this.padding;
    int weightChannel = inPerGroup * kernel * kernel;
    MatmulRuntime matmul = runtime == null ? MatmulRuntime.sequential() : runtime;
    matmul.parallelRanges(this.outChannels, (oc0, oc1) -> {
      for (int oc = oc0; oc < oc1; oc++) {
        float b = biasData == null ? 0f : biasData[biasOff + oc];
        int group = oc / outPerGroup;
        int wBase = weightOff + oc * weightChannel;
        int yBase = oc * outHeight * outWidth;
        int inStart = group * inPerGroup;
        for (int y = 0; y < outHeight; y++) {
          int originY = y * stride - pad;
          for (int x = 0; x < outWidth; x++) {
            int originX = x * stride - pad;
            float sum = b;
            for (int ic = 0; ic < inPerGroup; ic++) {
              int xBase = inputOff + (inStart + ic) * height * width;
              int wRow = wBase + ic * kernel * kernel;
              for (int ky = 0; ky < kernel; ky++) {
                int srcY = originY + ky;
                if (srcY < 0 || srcY >= height) {
                  continue;
                }
                for (int kx = 0; kx < kernel; kx++) {
                  int srcX = originX + kx;
                  if (srcX >= 0 && srcX < width) {
                    sum +=
                      weights[wRow + ky * kernel + kx] * activations[xBase + srcY * width + srcX];
                  }
                }
              }
            }
            dest[yBase + y * outWidth + x] = sum;
          }
        }
      }
    });
    return output;
  }
}
