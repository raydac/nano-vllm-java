package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

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
}
