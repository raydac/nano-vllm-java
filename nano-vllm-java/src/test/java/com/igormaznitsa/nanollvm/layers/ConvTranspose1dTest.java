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

  private static float[] direct(
    final float[] weight,
    final float[] bias,
    final float[] samples,
    final int inChannels,
    final int inLength,
    final int outChannels,
    final int kernel,
    final int stride,
    final int padding,
    final int outLength
  ) {
    float[] dest = new float[outChannels * outLength];
    for (int oc = 0; oc < outChannels; oc++) {
      float shift = bias[oc];
      if (shift != 0f) {
        for (int time = 0; time < outLength; time++) {
          dest[oc * outLength + time] = shift;
        }
      }
      for (int channel = 0; channel < inChannels; channel++) {
        int weightRow = channel * outChannels * kernel + oc * kernel;
        for (int time = 0; time < inLength; time++) {
          float sample = samples[channel * inLength + time];
          int origin = time * stride - padding;
          for (int tap = 0; tap < kernel; tap++) {
            int dst = origin + tap;
            if (dst >= 0 && dst < outLength) {
              dest[oc * outLength + dst] += weight[weightRow + tap] * sample;
            }
          }
        }
      }
    }
    return dest;
  }

  @Test
  void packedMatmulMatchesTheDirectFilter() {
    int inChannels = 8;
    int outChannels = 4;
    int kernel = 4;
    int length = 12;
    float[] weight = new float[inChannels * outChannels * kernel];
    float[] bias = new float[outChannels];
    float[] samples = new float[inChannels * length];
    for (int i = 0; i < weight.length; i++) {
      weight[i] = ((i % 9) - 4) * 0.02f;
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = 0.05f * i;
    }
    for (int i = 0; i < samples.length; i++) {
      samples[i] = (i % 6) - 1f;
    }
    ConvTranspose1d conv = new ConvTranspose1d(
      Tensor.of(weight, inChannels, outChannels, kernel),
      Tensor.of(bias, outChannels),
      2,
      1,
      0,
      1,
      1);
    Tensor out = conv.forward(Tensor.of(samples, inChannels, length));
    int outLength = (length - 1) * 2 - 2 + kernel;

    assertEquals(outLength, out.size(1));
    assertArrayEquals(
      direct(weight, bias, samples, inChannels, length, outChannels, kernel, 2, 1, outLength),
      out.data(),
      1e-4f);
  }
}
