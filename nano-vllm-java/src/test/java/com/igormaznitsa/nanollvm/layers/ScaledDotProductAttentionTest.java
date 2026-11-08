package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
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

  @Test
  void incrementalCausalMatchesPrefillAtLastPosition() {
    ScaledDotProductAttention attn = new ScaledDotProductAttention(1, 2, 1f, true);
    Tensor q = Tensor.of(new float[] {1f, 0f, 0f, 1f, 1f, 1f}, 3, 2);
    Tensor k = Tensor.of(new float[] {1f, 0f, 0f, 1f, 1f, 0f}, 3, 2);
    Tensor v = Tensor.of(new float[] {1f, 0f, 0f, 1f, 0f, 1f}, 3, 2);

    Tensor prefill = attn.forward(q, k, v);
    Tensor step = attn.forward(
      Tensor.wrap(q.data(), 4, 1, 2),
      k,
      v,
      null,
      2);

    assertArrayEquals(
      new float[] {prefill.get(4), prefill.get(5)},
      new float[] {step.get(0), step.get(1)},
      1e-5f);
  }

  private static float[] reference(
    final float[] queries,
    final float[] keys,
    final float[] values,
    final int heads,
    final int headDim,
    final int qLen,
    final int kvLen,
    final float scale,
    final boolean causal,
    final int queryStart
  ) {
    int width = heads * headDim;
    float[] out = new float[qLen * width];
    float[] scores = new float[kvLen];
    for (int head = 0; head < heads; head++) {
      int headBase = head * headDim;
      for (int row = 0; row < qLen; row++) {
        int lastKey = causal ? Math.min(queryStart + row, kvLen - 1) : kvLen - 1;
        float max = Float.NEGATIVE_INFINITY;
        for (int key = 0; key <= lastKey; key++) {
          float score = 0f;
          int qBase = row * width + headBase;
          int kBase = key * width + headBase;
          for (int lane = 0; lane < headDim; lane++) {
            score += queries[qBase + lane] * keys[kBase + lane];
          }
          score *= scale;
          scores[key] = score;
          if (score > max) {
            max = score;
          }
        }
        float sum = 0f;
        for (int key = 0; key <= lastKey; key++) {
          scores[key] = (float) Math.exp(scores[key] - max);
          sum += scores[key];
        }
        float inv = 1f / sum;
        for (int lane = 0; lane < headDim; lane++) {
          float mixed = 0f;
          for (int key = 0; key <= lastKey; key++) {
            mixed += scores[key] * inv * values[key * width + headBase + lane];
          }
          out[row * width + headBase + lane] = mixed;
        }
      }
    }
    return out;
  }

  @Test
  void longSequenceMatchesThePerKeyFormula() {
    int heads = 2;
    int headDim = 4;
    int length = 40;
    int width = heads * headDim;
    float[] queries = new float[length * width];
    float[] keys = new float[length * width];
    float[] values = new float[length * width];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = ((i % 11) - 5) * 0.05f;
      keys[i] = ((i % 7) - 3) * 0.07f;
      values[i] = ((i % 5) - 2) * 0.1f;
    }
    Tensor q = Tensor.of(queries, length, width);
    Tensor k = Tensor.of(keys, length, width);
    Tensor v = Tensor.of(values, length, width);
    float scale = 0.5f;
    Tensor out = new ScaledDotProductAttention(heads, headDim, scale, false).forward(q, k, v);

    assertArrayEquals(
      reference(queries, keys, values, heads, headDim, length, length, scale, false, 0),
      out.data(),
      1e-4f);
  }

  @Test
  void longCausalStepMatchesThePerKeyFormula() {
    int headDim = 4;
    int kvLen = 36;
    int width = headDim;
    float[] queries = new float[width];
    float[] keys = new float[kvLen * width];
    float[] values = new float[kvLen * width];
    for (int i = 0; i < keys.length; i++) {
      keys[i] = ((i % 9) - 4) * 0.04f;
      values[i] = ((i % 6) - 2) * 0.08f;
    }
    for (int i = 0; i < queries.length; i++) {
      queries[i] = ((i % 5) - 2) * 0.1f;
    }
    Tensor step = new ScaledDotProductAttention(1, headDim, 1f, true).forward(
      Tensor.of(queries, 1, width),
      Tensor.of(keys, kvLen, width),
      Tensor.of(values, kvLen, width),
      null,
      kvLen - 1);

    assertArrayEquals(
      reference(queries, keys, values, 1, headDim, 1, kvLen, 1f, true, kvLen - 1),
      step.data(),
      1e-4f);
  }

  @Test
  void longSequenceAgreesAcrossThePool() {
    int length = 48;
    int width = 8;
    float[] data = new float[length * width];
    for (int i = 0; i < data.length; i++) {
      data[i] = ((i % 13) - 6) * 0.03f;
    }
    Tensor q = Tensor.of(data, length, width);
    Tensor k = Tensor.of(data, length, width);
    Tensor v = Tensor.of(data, length, width);
    ScaledDotProductAttention attn = new ScaledDotProductAttention(2, 4, 0.5f, false);
    Tensor sequential = attn.forward(q, k, v);
    try (MatmulRuntime runtime = MatmulRuntime.builder().cpuThreads(4).dedicatedPool().build()) {
      Context context = new Context();
      context.bindMatmul(runtime);
      Tensor parallel = attn.forward(q, k, v, context);
      assertArrayEquals(sequential.data(), parallel.data(), 1e-4f);
    }
  }
}
