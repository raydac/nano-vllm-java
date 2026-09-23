package com.igormaznitsa.nanollvm.tensor.tornado;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory;
import org.junit.jupiter.api.Test;

class TornadoFloatKernelsTest {

  private static void scalarAttend(
    final float[] query, final float[] key, final float[] value, final float[] result,
    final int queryLength, final int keyLength, final int numHeads, final int numKvHeads,
    final int headDim,
    final float scale, final int slidingWindow, final boolean causal, final int[] keySlots
  ) {
    int repeats = numHeads / numKvHeads;
    float[] acc = new float[headDim];
    for (int job = 0; job < queryLength * numHeads; job++) {
      int head = job / queryLength;
      int queryIndex = job - head * queryLength;
      int kvHead = head / repeats;
      int causalEnd = causal ? (keyLength - queryLength + queryIndex + 1) : keyLength;
      int absoluteQuery = causal ? (keyLength - queryLength + queryIndex) : (keyLength - 1);
      int causalStart = slidingWindow > 0 ? Math.max(0, absoluteQuery - slidingWindow + 1) : 0;
      int queryBase = (queryIndex * numHeads + head) * headDim;
      for (int lane = 0; lane < headDim; lane++) {
        acc[lane] = 0f;
      }
      float max = Float.NEGATIVE_INFINITY;
      float sum = 0f;
      for (int keyIndex = causalStart; keyIndex < causalEnd; keyIndex++) {
        int token = keySlots == null ? keyIndex : keySlots[keyIndex];
        float score = 0f;
        if (token >= 0) {
          int keyBase = (token * numKvHeads + kvHead) * headDim;
          for (int lane = 0; lane < headDim; lane++) {
            score += query[queryBase + lane] * key[keyBase + lane];
          }
        }
        score *= scale;
        float nextMax = Math.max(max, score);
        float alpha = max == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(max - nextMax);
        sum *= alpha;
        for (int lane = 0; lane < headDim; lane++) {
          acc[lane] *= alpha;
        }
        float weight = (float) Math.exp(score - nextMax);
        sum += weight;
        if (token >= 0) {
          int valueBase = (token * numKvHeads + kvHead) * headDim;
          for (int lane = 0; lane < headDim; lane++) {
            acc[lane] += weight * value[valueBase + lane];
          }
        }
        max = nextMax;
      }
      float inverse = sum == 0f ? 0f : 1f / sum;
      for (int lane = 0; lane < headDim; lane++) {
        result[queryBase + lane] = acc[lane] * inverse;
      }
    }
  }

  private static float[] fill(final int n, final float scale) {
    float[] values = new float[n];
    fillSequence(values, scale);
    return values;
  }

  private static void assertClose(final float[] expected, final float[] actual) {
    for (int i = 0; i < expected.length; i++) {
      assertEquals(expected[i], actual[i], 1e-3f, "index " + i);
    }
  }

  private static void fillSequence(final float[] values, final float scale) {
    for (int i = 0; i < values.length; i++) {
      values[i] = (i % 19) * scale;
    }
  }

  @Test
  void largeGemvMatchesCpuFallback() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create(cpu);
    assumeTrue(tornado != null);

    int in = 256;
    int out = 256;
    float[] x = new float[in];
    float[] weight = new float[out * in];
    float[] bias = new float[out];
    float[] expected = new float[out];
    float[] actual = new float[out];
    for (int i = 0; i < in; i++) {
      x[i] = (i % 11) * 0.04f;
    }
    for (int i = 0; i < weight.length; i++) {
      weight[i] = (i * 3 % 17) * 0.02f;
    }
    for (int i = 0; i < out; i++) {
      bias[i] = i * 0.01f;
    }

    cpu.gemv(x, 0, weight, 0, bias, expected, 0, in, 0, out);
    tornado.gemv(x, 0, weight, 0, bias, actual, 0, in, 0, out);
    for (int i = 0; i < out; i++) {
      assertEquals(expected[i], actual[i], 1e-3f, "gemv " + i);
    }
  }

  @Test
  void smallGemvMatchesScalar() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create(cpu);
    assumeTrue(tornado != null);

    int in = 32;
    int out = 32;
    float[] x = new float[in];
    float[] weight = new float[out * in];
    float[] yCpu = new float[out];
    float[] yTornado = new float[out];
    for (int i = 0; i < in; i++) {
      x[i] = i * 0.1f;
    }
    for (int i = 0; i < weight.length; i++) {
      weight[i] = 0.01f * i;
    }

    cpu.gemv(x, 0, weight, 0, null, yCpu, 0, in, 0, out);
    tornado.gemv(x, 0, weight, 0, null, yTornado, 0, in, 0, out);
    for (int i = 0; i < out; i++) {
      assertEquals(yCpu[i], yTornado[i], 1e-3f);
    }
  }

  @Test
  void batchedGemvReusesWeightAcrossFreshOutputs() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    int rows = 3;
    int in = 260;
    int out = 256;
    int out0 = 32;
    int rowStride = out + out0;
    float[] x = new float[rows * in];
    float[] weight = new float[rowStride * in];
    float[] bias = new float[rowStride];
    fillSequence(x, 0.03f);
    fillSequence(weight, 0.02f);
    fillSequence(bias, 0.01f);

    float[] expected = new float[rows * rowStride];
    float[] first = new float[rows * rowStride];
    float[] second = new float[rows * rowStride];
    cpu.gemvRows(x, 0, weight, 0, bias, expected, 0, rows, in, rowStride, out0, out0 + out);
    TornadoGemvExecutor.gemvRows(
      x, 0, weight, 0, bias, first, 0, rows, in, rowStride, out0, out0 + out);
    assertClose(expected, first);
    x[0] += 0.5f;
    cpu.gemvRows(x, 0, weight, 0, bias, expected, 0, rows, in, rowStride, out0, out0 + out);
    TornadoGemvExecutor.gemvRows(
      x, 0, weight, 0, bias, second, 0, rows, in, rowStride, out0, out0 + out);
    assertClose(expected, second);
  }

  @Test
  void elementKernelsMatchScalar() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create(cpu);
    assumeTrue(tornado != null);
    assertEquals("TornadoVM", tornado.name());

    int n = 64;
    int off = 3;
    float[] a = fill(n + off, 0.05f);
    float[] b = fill(n + off, 0.07f);
    float[] weight = fill(n + off, 0.02f);
    float[] expected = new float[n + off];
    float[] actual = new float[n + off];

    assertEquals(cpu.dot(a, off, b, off, n), tornado.dot(a, off, b, off, n), 1e-3f);
    assertEquals(cpu.sumSquares(a, off, n), tornado.sumSquares(a, off, n), 1e-3f);

    cpu.add(a, off, b, off, expected, off, n);
    tornado.add(a, off, b, off, actual, off, n);
    assertClose(expected, actual);

    cpu.mul(a, off, b, off, expected, off, n);
    tornado.mul(a, off, b, off, actual, off, n);
    assertClose(expected, actual);

    cpu.scale(a, off, 1.5f, expected, off, n);
    tornado.scale(a, off, 1.5f, actual, off, n);
    assertClose(expected, actual);

    System.arraycopy(a, 0, expected, 0, expected.length);
    System.arraycopy(a, 0, actual, 0, actual.length);
    cpu.axpy(expected, off, 0.25f, b, off, n);
    tornado.axpy(actual, off, 0.25f, b, off, n);
    assertClose(expected, actual);

    cpu.scaleAdd(a, off, weight, off, 0.5f, expected, off, n);
    tornado.scaleAdd(a, off, weight, off, 0.5f, actual, off, n);
    assertClose(expected, actual);

    cpu.scaleAddOnePlus(a, off, weight, off, 0.5f, expected, off, n);
    tornado.scaleAddOnePlus(a, off, weight, off, 0.5f, actual, off, n);
    assertClose(expected, actual);

    assertEquals(
      cpu.addSumSquares(a, off, b, off, expected, off, n),
      tornado.addSumSquares(a, off, b, off, actual, off, n),
      1e-2f
    );
    assertClose(expected, actual);

    cpu.siluMul(a, off, b, off, expected, off, n);
    tornado.siluMul(a, off, b, off, actual, off, n);
    assertClose(expected, actual);

    cpu.geluTanh(a, off, expected, off, n);
    tornado.geluTanh(a, off, actual, off, n);
    assertClose(expected, actual);

    cpu.geluTanhMul(a, off, b, off, expected, off, n);
    tornado.geluTanhMul(a, off, b, off, actual, off, n);
    assertClose(expected, actual);

    cpu.tanhSoftcap(a, off, 2.0f, expected, off, n);
    tornado.tanhSoftcap(a, off, 2.0f, actual, off, n);
    assertClose(expected, actual);
  }

  @Test
  void attentionRangeMatchesScalar() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    int queryLength = 3;
    int keyLength = 4;
    int headDim = 4;
    int numHeads = 2;
    int numKvHeads = 1;
    float[] query = fill(queryLength * numHeads * headDim, 0.05f);
    float[] key = fill(keyLength * numKvHeads * headDim, 0.03f);
    float[] value = fill(key.length, 0.04f);
    float[] expected = new float[query.length];
    float[] actual = new float[query.length];
    scalarAttend(query, key, value, expected, queryLength, keyLength, numHeads, numKvHeads, headDim,
      0.5f, 0, true, null);
    assumeTrue(TornadoAttentionExecutor.attend(
      query, 0, key, 0, value, 0, actual, 0,
      0, queryLength, 0, keyLength, numHeads, numKvHeads, headDim, 0.5f, 0, true, null
    ));
    assertClose(expected, actual);

    key[key.length - 1] = 9f;
    scalarAttend(query, key, value, expected, queryLength, 3, numHeads, numKvHeads, headDim, 0.5f,
      2, true, null);
    assumeTrue(TornadoAttentionExecutor.attend(
      query, 0, key, 0, value, 0, actual, 0,
      0, queryLength, 0, 3, numHeads, numKvHeads, headDim, 0.5f, 2, true, null
    ));
    assertClose(expected, actual);
  }
}
