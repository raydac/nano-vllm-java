package com.igormaznitsa.nanollvm.tensor.tornado;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import org.junit.jupiter.api.Test;

class TornadoConvKernelsTest {

  @Test
  void yoloFilterShapesMatchTheScalarDots() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
    assumeTrue(tornado != null);

    this.assertMatches(tornado, 3, 4, 1, 1, 0, 1, 2, 2, false);
    this.assertMatches(tornado, 4, 8, 3, 2, 1, 1, 5, 5, false);
    this.assertMatches(tornado, 4, 4, 3, 1, 1, 4, 5, 5, false);
    this.assertMatches(tornado, 4, 4, 3, 1, 1, 2, 3, 3, false);
    this.assertMatches(tornado, 1, 1, 1, 2, 0, 1, 2, 2, false);
    this.assertMatches(tornado, 3, 4, 1, 1, 0, 1, 2, 2, true);
    this.assertMatches(tornado, 4, 8, 3, 2, 1, 1, 5, 5, true);
  }

  private void assertMatches(
    final FloatKernels tornado,
    final int inChannels, final int outChannels, final int kernel, final int stride,
    final int padding, final int groups, final int height, final int width, final boolean silu
  ) {
    int inPerGroup = inChannels / groups;
    float[] weight = this.filled(outChannels * inPerGroup * kernel * kernel, 5, 0.1f, -0.2f);
    float[] bias = this.filled(outChannels, 3, 0.05f, 0.1f);
    float[] input = this.filled(inChannels * height * width, 7, 0.25f, 0f);
    int outHeight = (height + 2 * padding - kernel) / stride + 1;
    int outWidth = (width + 2 * padding - kernel) / stride + 1;
    float[] expected = this.scalar(
      weight, bias, input, inChannels, outChannels, kernel, stride, padding, groups,
      height, width, silu);
    float[] actual = new float[expected.length];
    assertTrue(tornado.conv2d(
      input, 0, weight, 0, bias, 0, actual, 0,
      inChannels, height, width, outChannels, kernel, stride, padding, groups, silu));
    assertArrayEquals(expected, actual, 1e-3f);
    assertTrue(tornado.conv2d(
      input, 0, weight, 0, bias, 0, actual, 0,
      inChannels, height, width, outChannels, kernel, stride, padding, groups, silu));
    assertArrayEquals(expected, actual, 1e-3f);
    assertTrue(outHeight * outWidth * outChannels == expected.length);
  }

  private float[] scalar(
    final float[] weight, final float[] bias, final float[] input,
    final int inChannels, final int outChannels, final int kernel, final int stride,
    final int padding, final int groups, final int height, final int width, final boolean silu
  ) {
    int outHeight = (height + 2 * padding - kernel) / stride + 1;
    int outWidth = (width + 2 * padding - kernel) / stride + 1;
    int inPerGroup = inChannels / groups;
    int outPerGroup = outChannels / groups;
    float[] dest = new float[outChannels * outHeight * outWidth];
    for (int oc = 0; oc < outChannels; oc++) {
      int group = oc / outPerGroup;
      for (int y = 0; y < outHeight; y++) {
        int originY = y * stride - padding;
        for (int x = 0; x < outWidth; x++) {
          int originX = x * stride - padding;
          float sum = bias[oc];
          for (int ic = 0; ic < inPerGroup; ic++) {
            int plane = (group * inPerGroup + ic) * height * width;
            int taps = (oc * inPerGroup + ic) * kernel * kernel;
            for (int ky = 0; ky < kernel; ky++) {
              int srcY = originY + ky;
              if (srcY < 0 || srcY >= height) {
                continue;
              }
              for (int kx = 0; kx < kernel; kx++) {
                int srcX = originX + kx;
                if (srcX >= 0 && srcX < width) {
                  sum += input[plane + srcY * width + srcX] * weight[taps + ky * kernel + kx];
                }
              }
            }
          }
          if (silu) {
            sum = sum / (1f + (float) Math.exp(-sum));
          }
          dest[(oc * outHeight + y) * outWidth + x] = sum;
        }
      }
    }
    return dest;
  }

  private float[] filled(final int length, final int period, final float scale, final float shift) {
    float[] values = new float[length];
    for (int i = 0; i < length; i++) {
      values[i] = (i % period) * scale + shift;
    }
    return values;
  }
}
