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
}
