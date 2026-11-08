package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 1-D transposed convolution over {@code [inChannels, length]}. Weight layout is PyTorch
 * {@code [inChannels, outChannels/groups, kernel]}.
 *
 * <p>A single-group filter with a wide patch is multiplied with {@link MatmulRuntime#linear}, the
 * same kernels as chat, then scattered back onto the waveform. Other shapes split output channels
 * across the pool. Nested calls from a pool worker stay sequential so a fixed pool cannot deadlock.
 *
 * @since 1.3.0
 */
public final class ConvTranspose1d {

  private static final int PACK_IN = 8;
  private static final int PACK_PATCH = 16;
  private static final int PACK_TILE = 512;

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
   * Applies the filter. A wide single-group layer uses matmul and scatters the taps; every other
   * shape splits output channels across {@code runtime}.
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
    MatmulRuntime matmul = runtime == null ? MatmulRuntime.sequential() : runtime;
    if (this.usesPackedMatmul()) {
      this.forwardPacked(input, inLength, output, outLength, matmul);
    } else {
      this.forwardDirect(input, inLength, output, outLength, matmul);
    }
    return output;
  }

  private boolean usesPackedMatmul() {
    return this.groups == 1
      && this.inChannels >= PACK_IN
      && this.outChannels * this.kernel >= PACK_PATCH;
  }

  private void forwardPacked(
    final Tensor input,
    final int inLength,
    final Tensor output,
    final int outLength,
    final MatmulRuntime matmul
  ) {
    int patch = this.outChannels * this.kernel;
    float[] packedWeight = this.packWeight();
    int tile = Math.min(inLength, PACK_TILE);
    float[] rows = new float[tile * this.inChannels];
    float[] taps = new float[tile * patch];
    float[] activations = input.data();
    int inputOff = input.offset();
    float[] dest = output.data();
    this.fillBias(dest, outLength);
    for (int time = 0; time < inLength; time += tile) {
      int count = Math.min(tile, inLength - time);
      this.gatherRows(activations, inputOff, inLength, rows, time, count);
      matmul.linear(
        rows, 0, packedWeight, 0, null, taps, 0, count, this.inChannels, patch);
      this.scatterTaps(dest, outLength, taps, time, count, patch);
    }
  }

  private float[] packWeight() {
    int patch = this.outChannels * this.kernel;
    float[] packed = new float[patch * this.inChannels];
    float[] weights = this.weight.data();
    int weightOff = this.weight.offset();
    int kernel = this.kernel;
    int outChannels = this.outChannels;
    for (int channel = 0; channel < this.inChannels; channel++) {
      int srcBase = weightOff + channel * outChannels * kernel;
      for (int oc = 0; oc < outChannels; oc++) {
        for (int tap = 0; tap < kernel; tap++) {
          int row = oc * kernel + tap;
          packed[row * this.inChannels + channel] = weights[srcBase + oc * kernel + tap];
        }
      }
    }
    return packed;
  }

  private void fillBias(final float[] dest, final int outLength) {
    if (this.bias == null) {
      return;
    }
    float[] biasData = this.bias.data();
    int biasOff = this.bias.offset();
    for (int oc = 0; oc < this.outChannels; oc++) {
      float shift = biasData[biasOff + oc];
      if (shift == 0f) {
        continue;
      }
      int outputBase = oc * outLength;
      for (int time = 0; time < outLength; time++) {
        dest[outputBase + time] = shift;
      }
    }
  }

  private void gatherRows(
    final float[] activations,
    final int inputOff,
    final int inLength,
    final float[] rows,
    final int time0,
    final int count
  ) {
    int channels = this.inChannels;
    for (int i = 0; i < count; i++) {
      int row = i * channels;
      for (int channel = 0; channel < channels; channel++) {
        rows[row + channel] = activations[inputOff + channel * inLength + time0 + i];
      }
    }
  }

  private void scatterTaps(
    final float[] dest,
    final int outLength,
    final float[] taps,
    final int time0,
    final int count,
    final int patch
  ) {
    int kernel = this.kernel;
    int stride = this.stride;
    int pad = this.padding;
    int dil = this.dilation;
    int outChannels = this.outChannels;
    for (int i = 0; i < count; i++) {
      int row = i * patch;
      int origin = (time0 + i) * stride - pad;
      for (int oc = 0; oc < outChannels; oc++) {
        int outputBase = oc * outLength;
        int tapBase = row + oc * kernel;
        for (int tap = 0; tap < kernel; tap++) {
          int dst = origin + tap * dil;
          if (dst >= 0 && dst < outLength) {
            dest[outputBase + dst] += taps[tapBase + tap];
          }
        }
      }
    }
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
        int group = oc / outPerGroup;
        int ocInGroup = oc % outPerGroup;
        int outputBase = oc * outLength;
        float shift = biasData == null ? 0f : biasData[biasOff + oc];
        if (shift != 0f) {
          for (int time = 0; time < outLength; time++) {
            dest[outputBase + time] = shift;
          }
        }
        int inStart = group * inPerGroup;
        for (int channel = 0; channel < inPerGroup; channel++) {
          int globalChannel = inStart + channel;
          int weightRow = weightOff + globalChannel * outPerGroup * kernel + ocInGroup * kernel;
          int activationBase = inputOff + globalChannel * inLength;
          for (int time = 0; time < inLength; time++) {
            float sample = activations[activationBase + time];
            if (sample == 0f) {
              continue;
            }
            int origin = time * stride - pad;
            for (int tap = 0; tap < kernel; tap++) {
              int dst = origin + tap * dil;
              if (dst >= 0 && dst < outLength) {
                dest[outputBase + dst] += weights[weightRow + tap] * sample;
              }
            }
          }
        }
      }
    });
  }
}
