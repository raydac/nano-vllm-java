package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 1-D convolution over a channel-major sequence {@code [inChannels, length]}. Weight layout is
 * Hugging Face / PyTorch {@code [outChannels, inChannels, kernel]}. Edges use zero padding.
 *
 * <p>Independent output channels may run on a {@link MatmulRuntime} pool. Nested calls from a
 * pool worker stay sequential so a fixed pool cannot deadlock.
 *
 * @since 1.3.0
 */
public final class Conv1d {

  private final Tensor weight;
  private final Tensor bias;
  private final int stride;
  private final int padding;
  private final int dilation;
  private final int groups;
  private final int outChannels;
  private final int inChannels;
  private final int kernel;

  /**
   * Convolution with optional bias, stride 1 dilation, one group.
   *
   * @param weight  {@code [outChannels, inChannels, kernel]}
   * @param bias    length {@code outChannels}, or {@code null}
   * @param stride  temporal stride ({@code >= 1})
   * @param padding zeros on each side ({@code >= 0})
   */
  public Conv1d(final Tensor weight, final Tensor bias, final int stride, final int padding) {
    this(weight, bias, stride, padding, 1, 1);
  }

  /**
   * Convolution with dilation and groups (PyTorch Conv1d layout).
   *
   * @param weight   {@code [outChannels, inChannels/groups, kernel]}
   * @param bias     length {@code outChannels}, or {@code null}
   * @param stride   temporal stride ({@code >= 1})
   * @param padding  zeros on each side ({@code >= 0})
   * @param dilation kernel tap spacing ({@code >= 1})
   * @param groups   channel groups ({@code >= 1})
   * @since 1.3.0
   */
  public Conv1d(
    final Tensor weight,
    final Tensor bias,
    final int stride,
    final int padding,
    final int dilation,
    final int groups
  ) {
    this.weight = requireNonNull(weight, "weight");
    this.bias = bias;
    if (weight.shape().length != 3) {
      throw new IllegalArgumentException("conv1d weight must be [out, in/groups, kernel]");
    }
    if (stride < 1 || padding < 0 || dilation < 1 || groups < 1) {
      throw new IllegalArgumentException("stride, dilation, groups must be >= 1 and padding >= 0");
    }
    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;
    this.outChannels = weight.size(0);
    this.inChannels = weight.size(1) * groups;
    this.kernel = weight.size(2);
    if (this.outChannels % groups != 0) {
      throw new IllegalArgumentException("outChannels must be divisible by groups");
    }
    if (bias != null && bias.numel() != this.outChannels) {
      throw new IllegalArgumentException("conv1d bias length must equal outChannels");
    }
  }

  /**
   * Applies the filter to {@code input} of shape {@code [inChannels, length]} on the calling
   * thread.
   *
   * @param input channel-major activations
   * @return {@code [outChannels, outLength]}
   */
  public Tensor forward(final Tensor input) {
    return this.forward(input, MatmulRuntime.sequential());
  }

  /**
   * Same as {@link #forward(Tensor)} using the matmul runtime bound on {@code context}.
   *
   * @param input   channel-major activations
   * @param context step context (matmul pool); {@code null} is sequential
   * @return {@code [outChannels, outLength]}
   */
  public Tensor forward(final Tensor input, final Context context) {
    MatmulRuntime runtime = context != null && context.matmul() != null
      ? context.matmul()
      : MatmulRuntime.sequential();
    return this.forward(input, runtime);
  }

  /**
   * Applies the filter, splitting independent output channels across {@code runtime}.
   *
   * @param input   channel-major activations
   * @param runtime dense kernel runtime; {@code null} is sequential
   * @return {@code [outChannels, outLength]}
   */
  public Tensor forward(final Tensor input, final MatmulRuntime runtime) {
    requireNonNull(input, "input");
    if (input.shape().length != 2 || input.size(0) != this.inChannels) {
      throw new IllegalArgumentException(
        "conv1d input must be [inChannels, length], inChannels=" + this.inChannels);
    }
    int inLength = input.size(1);
    int effectiveKernel = this.dilation * (this.kernel - 1) + 1;
    int outLength = (inLength + 2 * this.padding - effectiveKernel) / this.stride + 1;
    if (outLength < 1) {
      throw new IllegalArgumentException("conv1d output length is < 1");
    }
    Tensor output = Tensor.zeros(this.outChannels, outLength);
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
    int dil = this.dilation;
    MatmulRuntime matmul = runtime == null ? MatmulRuntime.sequential() : runtime;
    matmul.parallelRanges(this.outChannels, (oc0, oc1) -> {
      for (int oc = oc0; oc < oc1; oc++) {
        float b = biasData == null ? 0f : biasData[biasOff + oc];
        int group = oc / outPerGroup;
        int wBase = weightOff + oc * inPerGroup * kernel;
        int yBase = oc * outLength;
        int inStart = group * inPerGroup;
        for (int t = 0; t < outLength; t++) {
          float sum = b;
          int origin = t * stride - pad;
          for (int ic = 0; ic < inPerGroup; ic++) {
            int xBase = inputOff + (inStart + ic) * inLength;
            int wRow = wBase + ic * kernel;
            for (int tap = 0; tap < kernel; tap++) {
              int src = origin + tap * dil;
              if (src >= 0 && src < inLength) {
                sum += weights[wRow + tap] * activations[xBase + src];
              }
            }
          }
          dest[yBase + t] = sum;
        }
      }
    });
    return output;
  }

  public int outChannels() {
    return this.outChannels;
  }
}
