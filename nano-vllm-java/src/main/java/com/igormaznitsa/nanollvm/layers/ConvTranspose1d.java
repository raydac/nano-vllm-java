package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 1-D transposed convolution over {@code [inChannels, length]}. Weight layout is PyTorch
 * {@code [inChannels, outChannels/groups, kernel]}.
 *
 * <p>Independent output channels may run on a {@link MatmulRuntime} pool. Nested calls from a
 * pool worker stay sequential so a fixed pool cannot deadlock.
 *
 * @since 1.3.0
 */
public final class ConvTranspose1d {

  private final Tensor weight;
  private final Tensor bias;
  private final int stride;
  private final int padding;
  private final int outputPadding;
  private final int dilation;
  private final int groups;
  private final int inChannels;
  private final int outChannels;
  private final int kernel;

  /**
   * Transposed convolution with optional bias, no extra output padding, stride-1 dilation, one
   * group.
   *
   * @param weight  {@code [inChannels, outChannels, kernel]}
   * @param bias    length {@code outChannels}, or {@code null}
   * @param stride  temporal stride ({@code >= 1})
   * @param padding zeros cropped from each side ({@code >= 0})
   */
  public ConvTranspose1d(
    final Tensor weight,
    final Tensor bias,
    final int stride,
    final int padding
  ) {
    this(weight, bias, stride, padding, 0, 1, 1);
  }

  /**
   * Transposed convolution with output padding, dilation, and groups (PyTorch ConvTranspose1d
   * layout).
   *
   * @param weight        {@code [inChannels, outChannels/groups, kernel]}
   * @param bias          length {@code outChannels}, or {@code null}
   * @param stride        temporal stride ({@code >= 1})
   * @param padding       zeros cropped from each side ({@code >= 0})
   * @param outputPadding extra samples on the right ({@code >= 0})
   * @param dilation      kernel tap spacing ({@code >= 1})
   * @param groups        channel groups ({@code >= 1})
   */
  public ConvTranspose1d(
    final Tensor weight,
    final Tensor bias,
    final int stride,
    final int padding,
    final int outputPadding,
    final int dilation,
    final int groups
  ) {
    this.weight = requireNonNull(weight, "weight");
    this.bias = bias;
    if (weight.shape().length != 3) {
      throw new IllegalArgumentException("convTranspose1d weight must be [in, out/groups, kernel]");
    }
    if (stride < 1 || padding < 0 || outputPadding < 0 || dilation < 1 || groups < 1) {
      throw new IllegalArgumentException("stride, dilation, groups must be >= 1; padding >= 0");
    }
    this.stride = stride;
    this.padding = padding;
    this.outputPadding = outputPadding;
    this.dilation = dilation;
    this.groups = groups;
    this.inChannels = weight.size(0);
    this.outChannels = weight.size(1) * groups;
    this.kernel = weight.size(2);
    if (this.inChannels % groups != 0) {
      throw new IllegalArgumentException("inChannels must be divisible by groups");
    }
    if (bias != null && bias.numel() != this.outChannels) {
      throw new IllegalArgumentException("convTranspose1d bias length must equal outChannels");
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
        "convTranspose1d input must be [inChannels, length], inChannels=" + this.inChannels);
    }
    int inLength = input.size(1);
    int outLength = (inLength - 1) * this.stride - 2 * this.padding
      + this.dilation * (this.kernel - 1) + this.outputPadding + 1;
    if (outLength < 1) {
      throw new IllegalArgumentException("convTranspose1d output length is < 1");
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
        int group = oc / outPerGroup;
        int ocInGroup = oc % outPerGroup;
        int yBase = oc * outLength;
        float b = biasData == null ? 0f : biasData[biasOff + oc];
        if (b != 0f) {
          for (int t = 0; t < outLength; t++) {
            dest[yBase + t] = b;
          }
        }
        int inStart = group * inPerGroup;
        for (int ic = 0; ic < inPerGroup; ic++) {
          int globalIc = inStart + ic;
          int wRow = weightOff + globalIc * outPerGroup * kernel + ocInGroup * kernel;
          int xBase = inputOff + globalIc * inLength;
          for (int t = 0; t < inLength; t++) {
            float xv = activations[xBase + t];
            if (xv == 0f) {
              continue;
            }
            int origin = t * stride - pad;
            for (int tap = 0; tap < kernel; tap++) {
              int dst = origin + tap * dil;
              if (dst >= 0 && dst < outLength) {
                dest[yBase + dst] += weights[wRow + tap] * xv;
              }
            }
          }
        }
      }
    });
    return output;
  }
}
