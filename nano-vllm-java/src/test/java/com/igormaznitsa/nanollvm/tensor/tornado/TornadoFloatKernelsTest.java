package com.igormaznitsa.nanollvm.tensor.tornado;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
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
  void oneTokenMatmulAndOneQueryAttentionRunOnTheDevice() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
    assumeTrue(tornado != null);
    int in = 32;
    int out = 16;
    float[] x = new float[in];
    float[] weight = new float[out * in];
    float[] expected = new float[out];
    float[] actual = new float[out];
    float[] oneRow = new float[out];
    for (int i = 0; i < in; i++) {
      x[i] = i * 0.1f;
    }
    for (int i = 0; i < weight.length; i++) {
      weight[i] = 0.01f * i;
    }

    int launches = TornadoClBlastGemv.launchCount();
    cpu.gemv(x, 0, weight, 0, null, expected, 0, in, 0, out);
    tornado.gemv(x, 0, weight, 0, null, actual, 0, in, 0, out);
    tornado.gemvRows(x, 0, weight, 0, null, oneRow, 0, 1, in, out, 0, out);
    assertClose(expected, actual);
    assertClose(expected, oneRow);
    int held = TornadoGemvExecutor.cachedPlanCount();
    assertTrue(held >= 1);
    tornado.gemv(x, 0, weight, 0, null, actual, 0, in, 0, out);
    assertEquals(held, TornadoGemvExecutor.cachedPlanCount());
    assertEquals(launches, TornadoClBlastGemv.launchCount());

    int queryLength = 1;
    int keyLength = 2;
    int headDim = 4;
    int numHeads = 1;
    int numKvHeads = 1;
    float[] query = fill(queryLength * numHeads * headDim, 0.05f);
    float[] key = fill(keyLength * numKvHeads * headDim, 0.03f);
    float[] value = fill(key.length, 0.04f);
    float[] attended = new float[query.length];
    float[] deviceAttended = new float[query.length];
    scalarAttend(query, key, value, attended, queryLength, keyLength, numHeads, numKvHeads, headDim,
      0.5f, 0, true, null);
    assertTrue(tornado.attend(
      query, 0, key, 0, value, 0, deviceAttended, 0,
      0, queryLength, 0, keyLength, numHeads, numKvHeads, headDim, 0.5f, 0, true, null
    ));
    assertClose(attended, deviceAttended);
  }

  @Test
  void largeGemvMatchesCpuFallback() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
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
  void vocabularySizedGemvStreamsWeightTiles() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
    assumeTrue(tornado != null);

    int in = 4096;
    int out = 2053;
    float[] x = new float[in];
    float[] weight = new float[out * in];
    float[] expected = new float[out];
    float[] actual = new float[out];
    for (int i = 0; i < in; i++) {
      x[i] = ((i % 7) - 3) * 0.01f;
    }
    for (int i = 0; i < weight.length; i++) {
      weight[i] = ((i % 5) - 2) * 0.01f;
    }

    int tiles = TornadoGemvExecutor.streamedTileCount();
    int launches = TornadoClBlastGemv.launchCount();
    cpu.gemv(x, 0, weight, 0, null, expected, 0, in, 0, out);
    tornado.gemv(x, 0, weight, 0, null, actual, 0, in, 0, out);
    assertClose(expected, actual);
    assertTrue(TornadoGemvExecutor.streamedTileCount() > tiles);
    int held = TornadoGemvExecutor.streamedTileCount();

    for (int i = 0; i < weight.length; i++) {
      weight[i] = ((i % 11) - 5) * 0.02f;
    }
    cpu.gemv(x, 0, weight, 0, null, expected, 0, in, 0, out);
    tornado.gemv(x, 0, weight, 0, null, actual, 0, in, 0, out);
    assertClose(expected, actual);
    assertEquals(held, TornadoGemvExecutor.streamedTileCount());
    assertEquals(launches, TornadoClBlastGemv.launchCount());
  }

  @Test
  void smallGemvMatchesScalar() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
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
    int launches = TornadoClBlastGemv.launchCount();
    cpu.gemvRows(x, 0, weight, 0, bias, expected, 0, rows, in, rowStride, out0, out0 + out);
    TornadoGemvExecutor.gemvRows(
      x, 0, weight, 0, bias, first, 0, rows, in, rowStride, out0, out0 + out);
    assertClose(expected, first);
    if (TornadoClBlastGemv.isInstalled()) {
      assertTrue(TornadoClBlastGemv.launchCount() > launches);
    }
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
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
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
  void largeReductionsMatchScalar() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
    assumeTrue(tornado != null);

    int n = 65536;
    float[] left = fill(n, 0.01f);
    float[] right = fill(n, 0.02f);
    assertEquals(cpu.dot(left, 0, right, 0, n), tornado.dot(left, 0, right, 0, n), 0.1f);
    assertEquals(cpu.sumSquares(left, 0, n), tornado.sumSquares(left, 0, n), 0.1f);
  }

  @Test
  void rmsNormRowsMatchScalar() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
    assumeTrue(tornado != null);
    int rows = 4;
    int last = 8;
    float[] src = fill(rows * last + 2, 0.05f);
    float[] weight = fill(last + 1, 0.02f);
    float[] expected = new float[rows * last];
    float[] actual = new float[rows * last];
    float eps = 1.0e-6f;
    for (int row = 0; row < rows; row++) {
      float energy = cpu.sumSquares(src, 2 + row * last, last);
      float inv = (float) (1.0 / Math.sqrt(energy / last + eps));
      cpu.scaleAdd(src, 2 + row * last, weight, 1, inv, expected, row * last, last);
    }
    assertTrue(tornado.rmsNormRows(src, 2, weight, 1, eps, false, actual, 0, rows, last));
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

  @Test
  void releasedWeightDropsDevicePlanAndRecomputes() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create();
    assumeTrue(tornado != null);

    int in = 32;
    int out = 32;
    float[] x = fill(in, 0.1f);
    float[] weight = fill(out * in, 0.01f);
    float[] expected = new float[out];
    float[] actual = new float[out];
    cpu.gemv(x, 0, weight, 0, null, expected, 0, in, 0, out);
    tornado.gemv(x, 0, weight, 0, null, actual, 0, in, 0, out);
    assertClose(expected, actual);
    int held = TornadoGemvExecutor.cachedPlanCount();
    assertTrue(held >= 1);

    tornado.releaseWeights(weight);
    assertEquals(held - 1, TornadoGemvExecutor.cachedPlanCount());

    x[0] += 0.25f;
    cpu.gemv(x, 0, weight, 0, null, expected, 0, in, 0, out);
    tornado.gemv(x, 0, weight, 0, null, actual, 0, in, 0, out);
    assertClose(expected, actual);
    assertEquals(held, TornadoGemvExecutor.cachedPlanCount());
  }

  @Test
  void tokenCountChangeDropsResidentGemvPlans() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels tornado = TornadoFloatKernelsProvider.create();
    assumeTrue(tornado != null);
    int in = 16;
    int out = 16;
    float[] x = fill(in * 2, 0.1f);
    float[] weight = fill(out * in, 0.01f);
    float[] y = new float[out * 2];
    tornado.gemvRows(x, 0, weight, 0, null, y, 0, 2, in, out, 0, out);
    assertTrue(TornadoGemvExecutor.cachedPlanCount() >= 1);

    tornado.gemv(x, 0, weight, 0, null, y, 0, in, 0, out);
    assertEquals(1, TornadoGemvExecutor.cachedPlanCount());
  }

  @Test
  void attentionReusesPackedPrefixWhenKeysGrow() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    int queryLength = 1;
    int headDim = 4;
    int numHeads = 2;
    int numKvHeads = 1;
    float[] query = fill(queryLength * numHeads * headDim, 0.05f);
    float[] key = fill(4 * numKvHeads * headDim, 0.03f);
    float[] value = fill(key.length, 0.04f);
    float[] expected = new float[query.length];
    float[] actual = new float[query.length];
    int[] slots = {0, 2, -1, 1};
    scalarAttend(query, key, value, expected, queryLength, 3, numHeads, numKvHeads, headDim,
      0.5f, 0, false, slots);
    assumeTrue(TornadoAttentionExecutor.attend(
      query, 0, key, 0, value, 0, actual, 0,
      0, queryLength, 0, 3, numHeads, numKvHeads, headDim, 0.5f, 0, false, slots
    ));
    assertClose(expected, actual);

    key[1 * headDim] = 1.5f;
    scalarAttend(query, key, value, expected, queryLength, 4, numHeads, numKvHeads, headDim,
      0.5f, 0, false, slots);
    assumeTrue(TornadoAttentionExecutor.attend(
      query, 0, key, 0, value, 0, actual, 0,
      0, queryLength, 0, 4, numHeads, numKvHeads, headDim, 0.5f, 0, false, slots
    ));
    assertClose(expected, actual);
  }
}
