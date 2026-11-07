package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * TornadoVM convolutions over an NCHW picture. The picture is read in place, so a 3×3 layer does
 * not upload a nine-times-larger packed window. Bias and SiLU stay in the same task.
 *
 * @since 1.5.0
 */
final class TornadoConvKernels {

  private TornadoConvKernels() {
  }

  static void pointwise(
    final float[] input, final float[] weight, final float[] bias, final float[] output,
    final int weightOffset, final int biasOffset,
    final int inChannels, final int height, final int width, final int outChannels,
    final int silu
  ) {
    int spatial = height * width;
    for (@Parallel int index = 0; index < output.length; index++) {
      int channel = index / spatial;
      int pixel = index - channel * spatial;
      float sum = bias[biasOffset + channel];
      int taps = weightOffset + channel * inChannels;
      for (int ic = 0; ic < inChannels; ic++) {
        sum += input[ic * spatial + pixel] * weight[taps + ic];
      }
      float value = sum;
      if (silu != 0) {
        value = sum / (1.0f + (float) Math.exp(-sum));
      }
      output[index] = value;
    }
  }

  static void dense3x3(
    final float[] input, final float[] weight, final float[] bias, final float[] output,
    final int weightOffset, final int biasOffset,
    final int inChannels, final int height, final int width, final int outChannels,
    final int stride, final int silu
  ) {
    int outHeight = (height - 1) / stride + 1;
    int outWidth = (width - 1) / stride + 1;
    int spatial = outHeight * outWidth;
    for (@Parallel int index = 0; index < output.length; index++) {
      int channel = index / spatial;
      int pixel = index - channel * spatial;
      int y = pixel / outWidth;
      int x = pixel - y * outWidth;
      int originY = y * stride - 1;
      int originX = x * stride - 1;
      float sum = bias[biasOffset + channel];
      int channelTaps = weightOffset + channel * inChannels * 9;
      for (int ic = 0; ic < inChannels; ic++) {
        int plane = ic * height * width;
        int taps = channelTaps + ic * 9;
        for (int ky = 0; ky < 3; ky++) {
          int srcY = originY + ky;
          if (srcY >= 0 && srcY < height) {
            int row = plane + srcY * width;
            int rowTaps = taps + ky * 3;
            for (int kx = 0; kx < 3; kx++) {
              int srcX = originX + kx;
              if (srcX >= 0 && srcX < width) {
                sum += input[row + srcX] * weight[rowTaps + kx];
              }
            }
          }
        }
      }
      float value = sum;
      if (silu != 0) {
        value = sum / (1.0f + (float) Math.exp(-sum));
      }
      output[index] = value;
    }
  }

  static void depthwise3x3(
    final float[] input, final float[] weight, final float[] bias, final float[] output,
    final int weightOffset, final int biasOffset,
    final int channels, final int height, final int width,
    final int stride, final int silu
  ) {
    int outHeight = (height - 1) / stride + 1;
    int outWidth = (width - 1) / stride + 1;
    int spatial = outHeight * outWidth;
    for (@Parallel int index = 0; index < output.length; index++) {
      int channel = index / spatial;
      int pixel = index - channel * spatial;
      int y = pixel / outWidth;
      int x = pixel - y * outWidth;
      float sum = bias[biasOffset + channel];
      int plane = channel * height * width;
      int taps = weightOffset + channel * 9;
      int originY = y * stride - 1;
      int originX = x * stride - 1;
      for (int ky = 0; ky < 3; ky++) {
        int srcY = originY + ky;
        if (srcY >= 0 && srcY < height) {
          int row = plane + srcY * width;
          int rowTaps = taps + ky * 3;
          for (int kx = 0; kx < 3; kx++) {
            int srcX = originX + kx;
            if (srcX >= 0 && srcX < width) {
              sum += input[row + srcX] * weight[rowTaps + kx];
            }
          }
        }
      }
      float value = sum;
      if (silu != 0) {
        value = sum / (1.0f + (float) Math.exp(-sum));
      }
      output[index] = value;
    }
  }

  static void general(
    final float[] input, final float[] weight, final float[] bias, final float[] output,
    final int[] spec
  ) {
    int weightOffset = spec[0];
    int biasOffset = spec[1];
    int inChannels = spec[2];
    int height = spec[3];
    int width = spec[4];
    int outChannels = spec[5];
    int kernel = spec[6];
    int stride = spec[7];
    int padding = spec[8];
    int groups = spec[9];
    int silu = spec[10];
    int outHeight = spec[11];
    int outWidth = spec[12];
    int inPerGroup = inChannels / groups;
    int outPerGroup = outChannels / groups;
    int spatial = outHeight * outWidth;
    int patch = kernel * kernel;
    for (@Parallel int index = 0; index < output.length; index++) {
      int channel = index / spatial;
      int pixel = index - channel * spatial;
      int y = pixel / outWidth;
      int x = pixel - y * outWidth;
      int group = channel / outPerGroup;
      int originY = y * stride - padding;
      int originX = x * stride - padding;
      float sum = bias[biasOffset + channel];
      int groupIn = group * inPerGroup;
      int taps = weightOffset + (channel * inPerGroup) * patch;
      for (int ic = 0; ic < inPerGroup; ic++) {
        int plane = (groupIn + ic) * height * width;
        int kernelTaps = taps + ic * patch;
        for (int ky = 0; ky < kernel; ky++) {
          int srcY = originY + ky;
          if (srcY >= 0 && srcY < height) {
            int row = plane + srcY * width;
            int rowTaps = kernelTaps + ky * kernel;
            for (int kx = 0; kx < kernel; kx++) {
              int srcX = originX + kx;
              if (srcX >= 0 && srcX < width) {
                sum += input[row + srcX] * weight[rowTaps + kx];
              }
            }
          }
        }
      }
      float value = sum;
      if (silu != 0) {
        value = sum / (1.0f + (float) Math.exp(-sum));
      }
      output[index] = value;
    }
  }

}
