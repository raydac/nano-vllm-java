package com.igormaznitsa.nanollvm.layers;

import com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Child process for {@link OptionalVectorModuleTest}. Started without
 * {@code --add-modules=jdk.incubator.vector}.
 */
public final class AttentionWithoutVectorProbe {

  private AttentionWithoutVectorProbe() {
  }

  public static void main(final String[] args) {
    if (FloatKernelsFactory.isVectorApiAvailable()) {
      System.out.println("vector-visible");
      return;
    }
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
    Tensor attended = new ScaledDotProductAttention(heads, headDim, 0.5f, false).forward(
      Tensor.of(queries, length, width),
      Tensor.of(keys, length, width),
      Tensor.of(values, length, width));
    if (attended.numel() != queries.length) {
      throw new IllegalStateException("attention length");
    }
    int channels = 4;
    int time = 16;
    float[] weight = new float[channels * 3];
    float[] input = new float[channels * time];
    for (int i = 0; i < weight.length; i++) {
      weight[i] = 0.1f * (i + 1);
    }
    for (int i = 0; i < input.length; i++) {
      input[i] = (i % 5) * 0.2f;
    }
    Tensor audio = new Conv1d(
      Tensor.of(weight, channels, 1, 3),
      null,
      1,
      1,
      1,
      channels).forward(Tensor.of(input, channels, time));
    if (audio.size(0) != channels || audio.size(1) != time) {
      throw new IllegalStateException("conv shape");
    }
    System.out.println("ok");
  }
}
