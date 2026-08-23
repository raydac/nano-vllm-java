package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 1-D convolution over a channel-major sequence {@code [inChannels, length]}. Weight layout is
 * Hugging Face / PyTorch {@code [outChannels, inChannels, kernel]}. Edges use zero padding.
 *
 * @since 1.3.0
 */
public final class Conv1d {

  private final Tensor weight;
  private final Tensor bias;
  private final int stride;
  private final int padding;
  private final int outChannels;
  private final int inChannels;
  private final int kernel;

  /**
   * Convolution with optional bias.
   *
   * @param weight  {@code [outChannels, inChannels, kernel]}
   * @param bias    length {@code outChannels}, or {@code null}
   * @param stride  temporal stride ({@code >= 1})
   * @param padding zeros on each side ({@code >= 0})
   */
  public Conv1d(final Tensor weight, final Tensor bias, final int stride, final int padding) {
    this.weight = requireNonNull(weight, "weight");
    this.bias = bias;
    if (weight.shape().length != 3) {
      throw new IllegalArgumentException("conv1d weight must be [out, in, kernel]");
    }
    if (stride < 1 || padding < 0) {
      throw new IllegalArgumentException("stride must be >= 1 and padding >= 0");
    }
    this.stride = stride;
    this.padding = padding;
    this.outChannels = weight.size(0);
    this.inChannels = weight.size(1);
    this.kernel = weight.size(2);
    if (bias != null && bias.numel() != this.outChannels) {
      throw new IllegalArgumentException("conv1d bias length must equal outChannels");
    }
  }

  /**
   * Applies the filter to {@code x} of shape {@code [inChannels, length]}.
   *
   * @param x channel-major input
   * @return {@code [outChannels, outLength]}
   */
  public Tensor forward(final Tensor x) {
    requireNonNull(x, "x");
    if (x.shape().length != 2 || x.size(0) != this.inChannels) {
      throw new IllegalArgumentException(
        "conv1d input must be [inChannels, length], inChannels=" + this.inChannels);
    }
    int inLength = x.size(1);
    int outLength = (inLength + 2 * this.padding - this.kernel) / this.stride + 1;
    if (outLength < 1) {
      throw new IllegalArgumentException("conv1d output length is < 1");
    }
    Tensor y = Tensor.zeros(this.outChannels, outLength);
    float[] wd = this.weight.data();
    int wOff = this.weight.offset();
    float[] xd = x.data();
    int xOff = x.offset();
    float[] yd = y.data();
    float[] biasData = this.bias == null ? null : this.bias.data();
    int biasOff = this.bias == null ? 0 : this.bias.offset();
    int inC = this.inChannels;
    int k = this.kernel;
    int stride = this.stride;
    int pad = this.padding;
    for (int oc = 0; oc < this.outChannels; oc++) {
      float b = biasData == null ? 0f : biasData[biasOff + oc];
      int wBase = wOff + oc * inC * k;
      int yBase = oc * outLength;
      for (int t = 0; t < outLength; t++) {
        float sum = b;
        int origin = t * stride - pad;
        for (int ic = 0; ic < inC; ic++) {
          int xBase = xOff + ic * inLength;
          int wRow = wBase + ic * k;
          for (int tap = 0; tap < k; tap++) {
            int src = origin + tap;
            if (src >= 0 && src < inLength) {
              sum += wd[wRow + tap] * xd[xBase + src];
            }
          }
        }
        yd[yBase + t] = sum;
      }
    }
    return y;
  }
}
