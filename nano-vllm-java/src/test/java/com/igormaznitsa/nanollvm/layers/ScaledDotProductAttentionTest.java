package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.igormaznitsa.nanollvm.tensor.Tensor;
import org.junit.jupiter.api.Test;

final class ScaledDotProductAttentionTest {

  @Test
  void causalMaskKeepsTheFirstTokenOnItsOwnValue() {
    Tensor q = Tensor.of(new float[] {1f, 0f, 1f, 0f}, 2, 2);
    Tensor k = Tensor.of(new float[] {1f, 0f, 1f, 0f}, 2, 2);
    Tensor v = Tensor.of(new float[] {1f, 0f, 0f, 1f}, 2, 2);

    Tensor causal = new ScaledDotProductAttention(1, 2, 1f, true).forward(q, k, v);
    Tensor full = new ScaledDotProductAttention(1, 2, 1f, false).forward(q, k, v);

    assertArrayEquals(new float[] {1f, 0f}, new float[] {causal.get(0), causal.get(1)}, 1e-5f);
    assertArrayEquals(new float[] {0.5f, 0.5f}, new float[] {causal.get(2), causal.get(3)}, 1e-5f);
    assertArrayEquals(new float[] {0.5f, 0.5f}, new float[] {full.get(0), full.get(1)}, 1e-5f);
    assertArrayEquals(new float[] {0.5f, 0.5f}, new float[] {full.get(2), full.get(3)}, 1e-5f);
  }
}
