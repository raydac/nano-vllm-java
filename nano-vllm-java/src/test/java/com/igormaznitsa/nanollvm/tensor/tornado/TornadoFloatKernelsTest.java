package com.igormaznitsa.nanollvm.tensor.tornado;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory;
import org.junit.jupiter.api.Test;

class TornadoFloatKernelsTest {

  @Test
  void largeGemvMatchesCpuFallback() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    FloatKernels tornado = TornadoFloatKernelsProvider.create(cpu);
    assumeTrue(tornado != null);

    int in = TornadoFloatKernels.MIN_IN;
    int out = TornadoFloatKernels.MIN_OUT;
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
  void smallGemvStaysOnCpuDelegate() {
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
      assertEquals(yCpu[i], yTornado[i], 1e-6f);
    }
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
  void batchedGemvReusesWeightAcrossFreshOutputs() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    FloatKernels cpu = FloatKernelsFactory.create("scalar");
    int rows = 3;
    int in = TornadoFloatKernels.MIN_IN + 4;
    int out = TornadoFloatKernels.MIN_OUT;
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
}
