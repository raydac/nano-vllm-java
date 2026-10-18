package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import org.junit.jupiter.api.Test;

class Qwen35KernelTest {

  private static GatedDeltaNet tinyNet() {
    return new GatedDeltaNet(
      Tensor.of(new float[] {0f, 1f, 0f}, 3, 1),
      Tensor.zeros(1, 1),
      Tensor.zeros(1, 1),
      Tensor.zeros(1, 1),
      Tensor.of(new float[] {0f, 1f, 1f, 0f, 0f, 1f}, 3, 2),
      Tensor.of(new float[] {0f}, 1),
      Tensor.of(new float[] {0f}, 1),
      Tensor.of(new float[] {1f}, 1),
      Tensor.zeros(1, 1),
      1,
      1,
      1,
      1,
      1e-6f,
      0);
  }

  @Test
  void gatedDeltaRuleWritesTheCorrection() {
    GatedDeltaNet net = tinyNet();
    float[] query = {1f, 1f};
    float[] key = {1f, 1f};
    float[] value = {2f, 0f};
    float[] beta = {1f, 1f};
    float[] decay = {0f, 0f};
    float[] core = new float[2];
    float[] state = new float[1];
    net.ruleRange(query, key, value, beta, decay, core, 0, 2, state);
    assertEquals(2f, core[0], 1e-5);
    assertEquals(0f, core[1], 1e-5);
    assertEquals(0f, state[0], 1e-5);
  }

  @Test
  void causalConvKeepsThePreviousInput() {
    GatedDeltaNet net = tinyNet();
    float[] mixed = {3f, 4f, 9f};
    Context ctx = new Context();
    net.causalConv(mixed, 1, ctx);
    assertEquals((float) (3.0 / (1.0 + Math.exp(-3.0))), mixed[0], 1e-4);
    assertEquals(0f, mixed[1], 1e-4);
    assertEquals((float) (9.0 / (1.0 + Math.exp(-9.0))), mixed[2], 1e-4);
  }

  @Test
  void leadingRopeRotatesOnlyThePrefix() {
    RotaryEmbedding rope = RotaryEmbedding.leading(4, 2, 4, 10_000f);
    Tensor positions = Tensor.of(new float[] {1f}, 1);
    Tensor query = Tensor.of(new float[] {1f, 0f, 9f, 8f}, 1, 1, 4);
    Tensor key = Tensor.of(new float[] {1f, 0f, 0f, 0f}, 1, 1, 4);
    Tensor[] rotated = rope.forward(positions, query, key);
    double angle = 1.0;
    assertEquals((float) Math.cos(angle), rotated[0].get(0), 1e-5);
    assertEquals((float) Math.sin(angle), rotated[0].get(1), 1e-5);
    assertEquals(9f, rotated[0].get(2), 1e-5);
    assertEquals(8f, rotated[0].get(3), 1e-5);
  }
}
