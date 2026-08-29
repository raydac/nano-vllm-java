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
}
