package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import org.junit.jupiter.api.Test;

final class Conv2dTest {

  @Test
  void pointwiseKernelScalesThePixel() {
    Conv2d conv = new Conv2d(
      Tensor.of(new float[] {2f}, 1, 1, 1, 1),
      Tensor.of(new float[] {0.5f}, 1),
      1,
      0,
      1);
    Tensor out = conv.forward(Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 2));

    assertEquals(1, out.size(0));
    assertEquals(2, out.size(1));
    assertArrayEquals(new float[] {2.5f, 4.5f, 6.5f, 8.5f}, out.data(), 1e-5f);
  }

  @Test
  void strideTwoHalvesTheMap() {
    Conv2d conv = new Conv2d(
      Tensor.of(new float[] {1f}, 1, 1, 1, 1),
      null,
      2,
      0,
      1);
    Tensor out = conv.forward(Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 2));

    assertEquals(1, out.size(1));
    assertEquals(1, out.size(2));
    assertEquals(1f, out.data()[0], 1e-6f);
  }

  private static float[] filled(final int length, final int period, final float scale,
                                final float shift) {
    float[] values = new float[length];
    for (int i = 0; i < length; i++) {
      values[i] = (i % period) * scale + shift;
    }
    return values;
  }

  private static float[] scalarConv(
    final float[] weight,
    final float[] bias,
    final float[] input,
    final int in,
    final int out,
    final int kernel,
    final int stride,
    final int pad,
    final int height,
    final int width
  ) {
    int outHeight = (height + 2 * pad - kernel) / stride + 1;
    int outWidth = (width + 2 * pad - kernel) / stride + 1;
    float[] dest = new float[out * outHeight * outWidth];
    for (int oc = 0; oc < out; oc++) {
      for (int y = 0; y < outHeight; y++) {
        int originY = y * stride - pad;
        for (int x = 0; x < outWidth; x++) {
          int originX = x * stride - pad;
          float sum = bias == null ? 0f : bias[oc];
          for (int ic = 0; ic < in; ic++) {
            for (int ky = 0; ky < kernel; ky++) {
              int srcY = originY + ky;
              if (srcY < 0 || srcY >= height) {
                continue;
              }
              for (int kx = 0; kx < kernel; kx++) {
                int srcX = originX + kx;
                if (srcX >= 0 && srcX < width) {
                  int wIndex = ((oc * in + ic) * kernel + ky) * kernel + kx;
                  sum += weight[wIndex] * input[(ic * height + srcY) * width + srcX];
                }
              }
            }
          }
          dest[(oc * outHeight + y) * outWidth + x] = sum;
        }
      }
    }
    return dest;
  }

  private static float[] scalarGrouped(
    final float[] weight,
    final float[] input,
    final int inPerGroup,
    final int groups,
    final int out,
    final int height,
    final int width
  ) {
    int kernel = 3;
    int outPerGroup = out / groups;
    float[] dest = new float[out * height * width];
    for (int oc = 0; oc < out; oc++) {
      int group = oc / outPerGroup;
      int inStart = group * inPerGroup;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          float sum = 0f;
          for (int ic = 0; ic < inPerGroup; ic++) {
            for (int ky = 0; ky < kernel; ky++) {
              int srcY = y - 1 + ky;
              if (srcY < 0 || srcY >= height) {
                continue;
              }
              for (int kx = 0; kx < kernel; kx++) {
                int srcX = x - 1 + kx;
                if (srcX >= 0 && srcX < width) {
                  int wIndex = ((oc * inPerGroup + ic) * kernel + ky) * kernel + kx;
                  int xIndex = ((inStart + ic) * height + srcY) * width + srcX;
                  sum += weight[wIndex] * input[xIndex];
                }
              }
            }
          }
          dest[(oc * height + y) * width + x] = sum;
        }
      }
    }
    return dest;
  }

  @Test
  void parallelRuntimeMatchesSequential() {
    Conv2d conv = new Conv2d(
      Tensor.of(new float[] {1f, 2f}, 2, 1, 1, 1),
      Tensor.of(new float[] {0.1f, -0.2f}, 2),
      1,
      0,
      1);
    Tensor input = Tensor.of(new float[] {3f, 4f, 5f, 6f}, 1, 2, 2);
    Tensor sequential = conv.forward(input);
    try (MatmulRuntime runtime = MatmulRuntime.builder().cpuThreads(2).dedicatedPool().build()) {
      assertArrayEquals(sequential.data(), conv.forward(input, runtime).data(), 1e-5f);
    }
  }

  @Test
  void maxPoolKeepsTheWindowPeak() {
    MaxPool2d pool = new MaxPool2d(2, 2, 0);
    Tensor out = pool.forward(Tensor.of(new float[] {1f, 4f, 2f, 3f}, 1, 2, 2));

    assertEquals(1, out.numel());
    assertEquals(4f, out.data()[0], 1e-6f);
  }

  @Test
  void transposedStrideTwoDoublesTheMap() {
    ConvTranspose2d conv = new ConvTranspose2d(
      Tensor.of(new float[] {1f, 0f, 0f, 0f}, 1, 1, 2, 2),
      Tensor.of(new float[] {0f}, 1),
      2,
      0);
    Tensor out = conv.forward(Tensor.of(new float[] {5f}, 1, 1, 1));

    assertEquals(2, out.size(1));
    assertEquals(2, out.size(2));
    assertEquals(5f, out.data()[0], 1e-5f);
    assertEquals(0f, out.data()[1], 1e-5f);
  }

  @Test
  void widePointwiseMatchesScalarDots() {
    int in = 16;
    int out = 8;
    int height = 2;
    int width = 3;
    float[] weight = filled(out * in, 5, 0.1f, -0.2f);
    float[] bias = filled(out, 1, 0.01f, 0f);
    float[] input = filled(in * height * width, 7, 0.25f, 0f);
    Conv2d conv = new Conv2d(Tensor.of(weight, out, in, 1, 1), Tensor.of(bias, out), 1, 0, 1);

    Tensor image = Tensor.of(input, in, height, width);
    float[] expected = scalarConv(weight, bias, input, in, out, 1, 1, 0, height, width);
    assertArrayEquals(expected, conv.forward(image).data(), 1e-4f);
    try (MatmulRuntime runtime = MatmulRuntime.builder().cpuThreads(2).dedicatedPool().build()) {
      assertArrayEquals(expected, conv.forward(image, runtime).data(), 1e-4f);
    }
  }

  @Test
  void tallNarrowMapUsesEveryCore() {
    int in = 16;
    int out = 16;
    int height = 8;
    int width = 8;
    float[] weight = filled(out * in, 5, 0.1f, -0.2f);
    float[] bias = filled(out, 1, 0.01f, 0f);
    float[] input = filled(in * height * width, 7, 0.25f, 0f);
    Conv2d conv = new Conv2d(Tensor.of(weight, out, in, 1, 1), Tensor.of(bias, out), 1, 0, 1);
    Tensor image = Tensor.of(input, in, height, width);
    float[] expected = scalarConv(weight, bias, input, in, out, 1, 1, 0, height, width);

    try (MatmulRuntime runtime = MatmulRuntime.builder().cpuThreads(4).dedicatedPool().build()) {
      assertArrayEquals(expected, conv.forward(image, runtime).data(), 1e-4f);
    }
  }

  @Test
  void paddedStrideMatchesScalarDots() {
    int in = 4;
    int out = 8;
    int height = 5;
    int width = 5;
    float[] weight = filled(out * in * 9, 5, 0.1f, -0.2f);
    float[] input = filled(in * height * width, 7, 0.25f, 0f);
    Conv2d conv = new Conv2d(Tensor.of(weight, out, in, 3, 3), null, 2, 1, 1);

    assertArrayEquals(
      scalarConv(weight, null, input, in, out, 3, 2, 1, height, width),
      conv.forward(Tensor.of(input, in, height, width)).data(),
      1e-4f);
  }

  @Test
  void groupedFilterStaysOnTheDirectLoop() {
    int inPerGroup = 2;
    int groups = 2;
    int out = 4;
    int height = 3;
    int width = 3;
    float[] weight = filled(out * inPerGroup * 9, 3, 0.2f, 0.1f);
    float[] input = filled((inPerGroup * groups) * height * width, 4, 0.5f, -0.3f);
    Conv2d conv = new Conv2d(
      Tensor.of(weight, out, inPerGroup, 3, 3),
      null,
      1,
      1,
      groups);

    assertArrayEquals(
      scalarGrouped(weight, input, inPerGroup, groups, out, height, width),
      conv.forward(Tensor.of(input, inPerGroup * groups, height, width)).data(),
      1e-5f);
  }
}
