package com.igormaznitsa.nanollvm.layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
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

  @Test
  void dilatedKernelSkipsHoles() {
    Conv1d conv = new Conv1d(
      Tensor.of(new float[] {1f, 1f}, 1, 1, 2),
      null,
      1,
      1,
      2,
      1);
    Tensor out = conv.forward(Tensor.of(new float[] {3f, 5f, 7f}, 1, 3));
    assertArrayEquals(new float[] {5f, 10f, 5f}, out.data(), 1e-5f);
  }

  @Test
  void parallelRuntimeMatchesSequential() {
    Conv1d conv = new Conv1d(
      Tensor.of(new float[] {
        1f, 0f, 0.5f,
        0f, 1f, -0.5f
      }, 2, 1, 3),
      Tensor.of(new float[] {0.1f, -0.2f}, 2),
      1,
      1);
    Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 1, 5);
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
    final int dilation,
    final int outLength
  ) {
    int groups = weight.length / (outChannels * kernel) == 1 ? inChannels : 1;
    int inPerGroup = inChannels / groups;
    int outPerGroup = outChannels / groups;
    float[] dest = new float[outChannels * outLength];
    for (int oc = 0; oc < outChannels; oc++) {
      float shift = bias == null ? 0f : bias[oc];
      int group = oc / outPerGroup;
      int weightBase = oc * inPerGroup * kernel;
      for (int time = 0; time < outLength; time++) {
        float sum = shift;
        int origin = time * stride - padding;
        for (int channel = 0; channel < inPerGroup; channel++) {
          int activationBase = (group * inPerGroup + channel) * inLength;
          int weightRow = weightBase + channel * kernel;
          for (int tap = 0; tap < kernel; tap++) {
            int src = origin + tap * dilation;
            if (src >= 0 && src < inLength) {
              sum += weight[weightRow + tap] * samples[activationBase + src];
            }
          }
        }
        dest[oc * outLength + time] = sum;
      }
    }
    return dest;
  }

  @Test
  void packedMatmulMatchesTheDirectFilter() {
    int inChannels = 8;
    int outChannels = 8;
    int kernel = 3;
    int length = 20;
    float[] weight = new float[outChannels * inChannels * kernel];
    float[] bias = new float[outChannels];
    float[] samples = new float[inChannels * length];
    for (int i = 0; i < weight.length; i++) {
      weight[i] = ((i % 7) - 3) * 0.05f;
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = i * 0.01f;
    }
    for (int i = 0; i < samples.length; i++) {
      samples[i] = (i % 5) - 2f;
    }
    Conv1d conv = new Conv1d(
      Tensor.of(weight, outChannels, inChannels, kernel), Tensor.of(bias, outChannels), 2, 1, 2, 1);
    Tensor out = conv.forward(Tensor.of(samples, inChannels, length));
    int outLength = out.size(1);

    assertArrayEquals(
      direct(weight, bias, samples, inChannels, length, outChannels, kernel, 2, 1, 2, outLength),
      out.data(),
      1e-4f);
  }

  @Test
  void depthwiseVectorMatchesTheDirectFilter() {
    int channels = 4;
    int kernel = 3;
    int length = 40;
    float[] weight = new float[channels * kernel];
    float[] samples = new float[channels * length];
    for (int i = 0; i < weight.length; i++) {
      weight[i] = ((i % 5) - 2) * 0.1f;
    }
    for (int i = 0; i < samples.length; i++) {
      samples[i] = (i % 9) * 0.25f;
    }
    Conv1d conv = new Conv1d(
      Tensor.of(weight, channels, 1, kernel), null, 1, 2, 2, channels);
    Tensor out = conv.forward(Tensor.of(samples, channels, length));

    assertArrayEquals(
      direct(weight, null, samples, channels, length, channels, kernel, 1, 2, 2, out.size(1)),
      out.data(),
      1e-4f);
  }
}
