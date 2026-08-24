package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import org.junit.jupiter.api.Test;

final class ConvTranspose1dTest {

  @Test
  void unitKernelStrideTwoInsertsHoles() {
    ConvTranspose1d conv = new ConvTranspose1d(
      Tensor.of(new float[] {1f}, 1, 1, 1),
      null,
      2,
      0);
    Tensor out = conv.forward(Tensor.of(new float[] {3f, 5f}, 1, 2));
    assertEquals(1, out.size(0));
    assertEquals(3, out.size(1));
    assertArrayEquals(new float[] {3f, 0f, 5f}, out.data(), 1e-5f);
  }

  @Test
  void parallelRuntimeMatchesSequential() {
    ConvTranspose1d conv = new ConvTranspose1d(
      Tensor.of(new float[] {1f, 0.5f}, 1, 1, 2),
      Tensor.of(new float[] {0.25f}, 1),
      2,
      1);
    Tensor input = Tensor.of(new float[] {2f, 4f, 6f}, 1, 3);
    Tensor sequential = conv.forward(input);
    try (MatmulRuntime runtime = MatmulRuntime.builder().cpuThreads(2).dedicatedPool().build()) {
      Tensor parallel = conv.forward(input, runtime);
      assertArrayEquals(sequential.data(), parallel.data(), 1e-5f);
    }
  }
}
