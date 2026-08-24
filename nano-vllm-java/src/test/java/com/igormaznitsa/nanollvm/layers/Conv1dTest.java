package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import org.junit.jupiter.api.Test;

final class Conv1dTest {

  @Test
  void identityKernelCopiesTheSequence() {
    Conv1d conv =
      new Conv1d(Tensor.of(new float[] {1f}, 1, 1, 1), Tensor.of(new float[] {0f}, 1), 1, 0);
    Tensor out = conv.forward(Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 4));

    assertEquals(1, out.size(0));
    assertEquals(4, out.size(1));
    assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, out.data(), 1e-6f);
  }

  @Test
  void paddedCenteredTapMatchesManualConvolution() {
    Conv1d conv = new Conv1d(
      Tensor.of(new float[] {0.5f, 1f, 0.5f}, 1, 1, 3),
      Tensor.of(new float[] {0.25f}, 1),
      1,
      1);
    Tensor out = conv.forward(Tensor.of(new float[] {2f, 4f, 6f}, 1, 3));

    assertArrayEquals(new float[] {4.25f, 8.25f, 8.25f}, out.data(), 1e-5f);
  }

  @Test
  void dilatedKernelSkipsHoles() {
    Conv1d conv = new Conv1d(
      Tensor.of(new float[] {1f, 1f}, 1, 1, 2),
      null,
      1,
      1,
      2,
      1);
    Tensor out = conv.forward(Tensor.of(new float[] {3f, 5f, 7f}, 1, 3));
    assertArrayEquals(new float[] {5f, 10f, 5f}, out.data(), 1e-5f);
  }

  @Test
  void parallelRuntimeMatchesSequential() {
    Conv1d conv = new Conv1d(
      Tensor.of(new float[] {
        1f, 0f, 0.5f,
        0f, 1f, -0.5f
      }, 2, 1, 3),
      Tensor.of(new float[] {0.1f, -0.2f}, 2),
      1,
      1);
    Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 1, 5);
    Tensor sequential = conv.forward(input);
    try (MatmulRuntime runtime = MatmulRuntime.builder().cpuThreads(2).dedicatedPool().build()) {
      Tensor parallel = conv.forward(input, runtime);
      assertArrayEquals(sequential.data(), parallel.data(), 1e-5f);
    }
  }
}
