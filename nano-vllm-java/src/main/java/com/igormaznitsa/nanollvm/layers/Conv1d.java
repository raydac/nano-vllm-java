package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.vector.VectorConv1d;

/**
 * 1-D convolution over a channel-major sequence {@code [inChannels, length]}. Weight layout is
 * Hugging Face / PyTorch {@code [outChannels, inChannels, kernel]}. Edges use zero padding.
 *
 * <p>A single-group filter with a wide patch is gathered into rows and multiplied with
 * {@link MatmulRuntime#linear}, the same kernels as chat. A depthwise filter uses the Vector API
 * along the waveform when that module is loaded. Other shapes split output channels across the
 * pool. Nested calls from a pool worker stay sequential so a fixed pool cannot deadlock.
 *
 * @since 1.3.0
 */
public final class Conv1d {

  private static final int PACK_PATCH = 16;
  private static final int PACK_OUT = 8;
  private static final int PACK_TILE = 512;
  private static final int DEPTHWISE_LENGTH = 16;

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
   * Applies the filter. A wide single-group layer uses matmul; a long depthwise layer uses SIMD
   * along time when the Vector API is loaded; every other shape splits output channels across
   * {@code runtime}.
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
    MatmulRuntime matmul = runtime == null ? MatmulRuntime.sequential() : runtime;
    if (this.usesPackedMatmul()) {
      this.forwardPacked(input, inLength, output, outLength, matmul);
    } else if (this.usesVectorDepthwise(outLength)) {
      this.forwardDepthwise(input, inLength, output, outLength, matmul);
    } else {
      this.forwardDirect(input, inLength, output, outLength, matmul);
    }
    return output;
  }

  private boolean usesPackedMatmul() {
    return this.groups == 1
        && this.inChannels * this.kernel >= PACK_PATCH
        && this.outChannels >= PACK_OUT;
  }

  private boolean usesVectorDepthwise(final int outLength) {
    return this.groups == this.inChannels
        && this.outChannels == this.inChannels
        && outLength >= DEPTHWISE_LENGTH
        && FloatKernelsFactory.isVectorApiAvailable();
  }

  private void forwardPacked(
      final Tensor input,
      final int inLength,
      final Tensor output,
      final int outLength,
      final MatmulRuntime matmul
  ) {
    int patch = this.inChannels * this.kernel;
    float[] biasArg = this.biasVector();
    int tile = Math.min(outLength, PACK_TILE);
    float[] columns = new float[tile * patch];
    float[] rows = new float[tile * this.outChannels];
    float[] activations = input.data();
    int inputOff = input.offset();
    float[] weights = this.weight.data();
    int weightOff = this.weight.offset();
    float[] dest = output.data();
    for (int time = 0; time < outLength; time += tile) {
      int count = Math.min(tile, outLength - time);
      this.gatherColumns(activations, inputOff, inLength, columns, time, count, patch);
      matmul.linear(
          columns, 0, weights, weightOff, biasArg, rows, 0, count, patch, this.outChannels);
      this.scatterRows(dest, outLength, rows, time, count);
    }
  }

  private void gatherColumns(
      final float[] activations,
      final int inputOff,
      final int inLength,
      final float[] columns,
      final int time0,
      final int count,
      final int patch
  ) {
    int kernel = this.kernel;
    int stride = this.stride;
    int pad = this.padding;
    int dil = this.dilation;
    int channels = this.inChannels;
    for (int i = 0; i < count; i++) {
      int origin = (time0 + i) * stride - pad;
      int col = i * patch;
      for (int channel = 0; channel < channels; channel++) {
        int activationBase = inputOff + channel * inLength;
        for (int tap = 0; tap < kernel; tap++) {
          int src = origin + tap * dil;
          columns[col++] = src >= 0 && src < inLength ? activations[activationBase + src] : 0f;
        }
      }
    }
  }

  private void scatterRows(
      final float[] dest, final int outLength, final float[] rows, final int time0, final int count
  ) {
    int channels = this.outChannels;
    for (int i = 0; i < count; i++) {
      int row = i * channels;
      int time = time0 + i;
      for (int channel = 0; channel < channels; channel++) {
        dest[channel * outLength + time] = rows[row + channel];
      }
    }
  }

  private void forwardDepthwise(
      final Tensor input,
      final int inLength,
      final Tensor output,
      final int outLength,
      final MatmulRuntime matmul
  ) {
    float[] weights = this.weight.data();
    int weightOff = this.weight.offset();
    float[] biasData = this.bias == null ? null : this.bias.data();
    int biasOff = this.bias == null ? 0 : this.bias.offset();
    float[] activations = input.data();
    int inputOff = input.offset();
    float[] dest = output.data();
    int kernel = this.kernel;
    int stride = this.stride;
    int pad = this.padding;
    int dil = this.dilation;
    if (outLength >= 16384) {
      matmul.parallelRanges(this.outChannels, (channel0, channel1) -> VectorConv1d.channels(
          weights, weightOff, biasData, biasOff, activations, inputOff, dest,
          channel0, channel1, inLength, outLength, kernel, stride, pad, dil));
      return;
    }
    VectorConv1d.channels(
        weights, weightOff, biasData, biasOff, activations, inputOff, dest,
        0, this.outChannels, inLength, outLength, kernel, stride, pad, dil);
  }

  private void forwardDirect(
      final Tensor input,
      final int inLength,
      final Tensor output,
      final int outLength,
      final MatmulRuntime matmul
  ) {
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
    matmul.parallelRanges(this.outChannels, (oc0, oc1) -> {
      for (int oc = oc0; oc < oc1; oc++) {
        float shift = biasData == null ? 0f : biasData[biasOff + oc];
        int group = oc / outPerGroup;
        int weightBase = weightOff + oc * inPerGroup * kernel;
        int outputBase = oc * outLength;
        int inStart = group * inPerGroup;
        for (int time = 0; time < outLength; time++) {
          float sum = shift;
          int origin = time * stride - pad;
          for (int channel = 0; channel < inPerGroup; channel++) {
            int activationBase = inputOff + (inStart + channel) * inLength;
            int weightRow = weightBase + channel * kernel;
            for (int tap = 0; tap < kernel; tap++) {
              int src = origin + tap * dil;
              if (src >= 0 && src < inLength) {
                sum += weights[weightRow + tap] * activations[activationBase + src];
              }
            }
          }
          dest[outputBase + time] = sum;
        }
      }
    });
  }

  private float[] biasVector() {
    if (this.bias == null) {
      return null;
    }
    return this.bias.offset() == 0 ? this.bias.data() : this.bias.toFloatArray();
  }

  public int outChannels() {
    return this.outChannels;
  }
}
