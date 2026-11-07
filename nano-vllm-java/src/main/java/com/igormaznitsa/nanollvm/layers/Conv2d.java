package com.igormaznitsa.nanollvm.layers;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * 2-D convolution over a channel-major image {@code [inChannels, height, width]}. Weight layout is
 * PyTorch {@code [outChannels, inChannels/groups, kernel, kernel]}. Edges use zero padding.
 *
 * <p>A single-group filter with a patch of at least {@value #MIN_PACKED_PATCH} floats and at least
 * {@value #MIN_PACKED_OUTPUTS} outputs is packed once and multiplied with
 * {@link MatmulRuntime#linear}, so it uses the same vector GEMV as a chat layer.
 * When the process backend is TornadoVM, the filter runs on the device from the picture itself:
 * weights stay resident, and SiLU can be applied in that same task. Depthwise and smaller filters
 * on a CPU backend stay on the direct channel loop.
 *
 * @since 1.5.0
 */
public final class Conv2d {

  private static final int MIN_PACKED_PATCH = 16;
  private static final int MIN_PACKED_OUTPUTS = 8;
  private static final ThreadLocal<ImageMatmulScratch> SCRATCH =
    ThreadLocal.withInitial(ImageMatmulScratch::new);

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
    return this.forward(input, MatmulRuntime.sequential(), false);
  }

  /**
   * Applies the filter, splitting output channels across {@code runtime}.
   *
   * @param input   {@code [inChannels, height, width]}
   * @param runtime dense kernel runtime; {@code null} is sequential
   * @return {@code [outChannels, outHeight, outWidth]}
   */
  public Tensor forward(final Tensor input, final MatmulRuntime runtime) {
    return this.forward(input, runtime, false);
  }

  /**
   * Applies the filter. When {@code fuseSilu} is set, the returned picture is
   * {@code silu(conv + bias)}. A TornadoVM backend does that in the convolution task.
   *
   * @param input    {@code [inChannels, height, width]}
   * @param runtime  dense kernel runtime; {@code null} is sequential
   * @param fuseSilu apply SiLU to the convolution
   * @return {@code [outChannels, outHeight, outWidth]}
   */
  public Tensor forward(final Tensor input, final MatmulRuntime runtime, final boolean fuseSilu) {
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
    MatmulRuntime matmul = runtime == null ? MatmulRuntime.sequential() : runtime;
    if (this.onDevice(input, output, height, width, matmul, fuseSilu)) {
      return output;
    }
    if (this.usesPackedMatmul(outHeight * outWidth)) {
      this.forwardPacked(input, output, height, width, outHeight, outWidth, matmul);
    } else {
      this.forwardDirect(input, output, height, width, outHeight, outWidth, matmul);
    }
    if (fuseSilu) {
      this.applySilu(output, matmul);
    }
    return output;
  }

  private boolean onDevice(
      final Tensor input, final Tensor output, final int height, final int width,
      final MatmulRuntime runtime, final boolean fuseSilu
  ) {
    return runtime.conv2d(
        input.data(), input.offset(),
        this.weight.data(), this.weight.offset(),
        this.bias == null ? null : this.bias.data(),
        this.bias == null ? 0 : this.bias.offset(),
        output.data(), output.offset(),
        this.inChannels, height, width,
        this.outChannels, this.kernel, this.stride, this.padding, this.groups,
        fuseSilu);
  }

  private void applySilu(final Tensor output, final MatmulRuntime runtime) {
    float[] data = output.data();
    int offset = output.offset();
    runtime.parallelRanges(output.numel(), (start, end) -> {
      for (int i = start; i < end; i++) {
        float value = data[offset + i];
        data[offset + i] = value / (1f + (float) Math.exp(-value));
      }
    });
  }

  private boolean usesPackedMatmul(final int spatial) {
    long patch = (long) this.inChannels * this.kernel * this.kernel;
    if (this.groups != 1 || patch < MIN_PACKED_PATCH || this.outChannels < MIN_PACKED_OUTPUTS) {
      return false;
    }
    return (long) spatial * patch <= Integer.MAX_VALUE
      && (long) spatial * this.outChannels <= Integer.MAX_VALUE;
  }

  private void forwardPacked(
    final Tensor input,
    final Tensor output,
    final int height,
    final int width,
    final int outHeight,
    final int outWidth,
    final MatmulRuntime runtime
  ) {
    int spatial = outHeight * outWidth;
    int patch = this.inChannels * this.kernel * this.kernel;
    ImageMatmulScratch scratch = SCRATCH.get();
    float[] patches = scratch.patches(spatial * patch);
    float[] rows = scratch.rows(spatial * this.outChannels);
    runtime.parallelRanges(outHeight, (y0, y1) -> this.packRows(
      input.data(), input.offset(), height, width, outWidth, y0, y1, patches));
    runtime.linear(
      patches, 0,
      this.weight.data(), this.weight.offset(),
      null,
      rows, 0,
      spatial, patch, this.outChannels);
    this.scatterChannels(output, rows, spatial, runtime);
  }

  private void packRows(
    final float[] activations,
    final int inputOff,
    final int height,
    final int width,
    final int outWidth,
    final int y0,
    final int y1,
    final float[] patches
  ) {
    if (this.kernel == 1 && this.stride == 1 && this.padding == 0) {
      this.packPointwise(activations, inputOff, height, width, y0, y1, patches);
      return;
    }
    if (this.kernel == 3 && this.padding == 1 && (this.stride == 1 || this.stride == 2)) {
      this.pack3x3(activations, inputOff, height, width, outWidth, y0, y1, patches);
      return;
    }
    int patch = this.inChannels * this.kernel * this.kernel;
    for (int y = y0; y < y1; y++) {
      int originY = y * this.stride - this.padding;
      for (int x = 0; x < outWidth; x++) {
        int originX = x * this.stride - this.padding;
        int cursor = (y * outWidth + x) * patch;
        for (int ic = 0; ic < this.inChannels; ic++) {
          int plane = inputOff + ic * height * width;
          for (int ky = 0; ky < this.kernel; ky++) {
            int srcY = originY + ky;
            boolean insideRow = srcY >= 0 && srcY < height;
            int row = insideRow ? plane + srcY * width : 0;
            for (int kx = 0; kx < this.kernel; kx++) {
              int srcX = originX + kx;
              patches[cursor++] = insideRow && srcX >= 0 && srcX < width
                ? activations[row + srcX]
                : 0f;
            }
          }
        }
      }
    }
  }

  private void pack3x3(
    final float[] activations,
    final int inputOff,
    final int height,
    final int width,
    final int outWidth,
    final int y0,
    final int y1,
    final float[] patches
  ) {
    int patch = this.inChannels * 9;
    int stride = this.stride;
    for (int y = y0; y < y1; y++) {
      int originY = y * stride - 1;
      boolean rowInside = originY >= 0 && originY + 2 < height;
      for (int x = 0; x < outWidth; x++) {
        int originX = x * stride - 1;
        int cursor = (y * outWidth + x) * patch;
        if (rowInside && originX >= 0 && originX + 2 < width) {
          for (int ic = 0; ic < this.inChannels; ic++) {
            int row0 = inputOff + (ic * height + originY) * width + originX;
            int row1 = row0 + width;
            int row2 = row1 + width;
            patches[cursor] = activations[row0];
            patches[cursor + 1] = activations[row0 + 1];
            patches[cursor + 2] = activations[row0 + 2];
            patches[cursor + 3] = activations[row1];
            patches[cursor + 4] = activations[row1 + 1];
            patches[cursor + 5] = activations[row1 + 2];
            patches[cursor + 6] = activations[row2];
            patches[cursor + 7] = activations[row2 + 1];
            patches[cursor + 8] = activations[row2 + 2];
            cursor += 9;
          }
          continue;
        }
        for (int ic = 0; ic < this.inChannels; ic++) {
          int plane = inputOff + ic * height * width;
          for (int ky = 0; ky < 3; ky++) {
            int srcY = originY + ky;
            boolean insideRow = srcY >= 0 && srcY < height;
            int row = insideRow ? plane + srcY * width : 0;
            for (int kx = 0; kx < 3; kx++) {
              int srcX = originX + kx;
              patches[cursor++] = insideRow && srcX >= 0 && srcX < width
                ? activations[row + srcX]
                : 0f;
            }
          }
        }
      }
    }
  }

  private void packPointwise(
    final float[] activations,
    final int inputOff,
    final int height,
    final int width,
    final int y0,
    final int y1,
    final float[] patches
  ) {
    int span = height * width;
    for (int y = y0; y < y1; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = y * width + x;
        int dst = pixel * this.inChannels;
        int at = inputOff + pixel;
        for (int ic = 0; ic < this.inChannels; ic++) {
          patches[dst + ic] = activations[at + ic * span];
        }
      }
    }
  }

  private void scatterChannels(
    final Tensor output,
    final float[] rows,
    final int spatial,
    final MatmulRuntime runtime
  ) {
    float[] dest = output.data();
    int destOff = output.offset();
    float[] biasData = this.bias == null ? null : this.bias.data();
    int biasOff = this.bias == null ? 0 : this.bias.offset();
    int outputs = this.outChannels;
    runtime.parallelRanges(outputs, (oc0, oc1) -> {
      for (int oc = oc0; oc < oc1; oc++) {
        float biasValue = biasData == null ? 0f : biasData[biasOff + oc];
        int plane = destOff + oc * spatial;
        for (int s = 0; s < spatial; s++) {
          dest[plane + s] = rows[s * outputs + oc] + biasValue;
        }
      }
    });
  }

  private void forwardDirect(
    final Tensor input,
    final Tensor output,
    final int height,
    final int width,
    final int outHeight,
    final int outWidth,
    final MatmulRuntime runtime
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
    int weightChannel = inPerGroup * kernel * kernel;
    runtime.parallelRanges(this.outChannels, (oc0, oc1) -> {
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
  }

  private static final class ImageMatmulScratch {
    private float[] patches = new float[0];
    private float[] rows = new float[0];

    private float[] patches(final int needed) {
      if (this.patches.length < needed) {
        this.patches = new float[needed];
      }
      return this.patches;
    }

    private float[] rows(final int needed) {
      if (this.rows.length < needed) {
        this.rows = new float[needed];
      }
      return this.rows;
    }
  }
}
