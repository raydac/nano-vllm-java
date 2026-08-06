package com.igormaznitsa.nanollvm.layers;

import com.igormaznitsa.nanollvm.tensor.Tensor;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public final class Sampler {

  public int[] forward(final Tensor logits, final float[] temperatures, final int[] topKs,
                       final float[] topPs) {
    int rows = logits.size(0);
    int vocab = logits.size(1);
    int[] out = new int[rows];
    for (int r = 0; r < rows; r++) {
      float temperature = temperatures[r];
      int topK = topKs != null ? topKs[r] : 0;
      float topP = topPs != null ? topPs[r] : 1f;
      out[r] = this.sampleRow(logits, r, vocab, temperature, topK, topP);
    }
    return out;
  }

  public int[] forward(final Tensor logits, final float[] temperatures) {
    int[] topKs = new int[temperatures.length];
    float[] topPs = new float[temperatures.length];
    Arrays.fill(topPs, 1f);
    return this.forward(logits, temperatures, topKs, topPs);
  }

  private int sampleRow(final Tensor logits, final int row, final int vocab,
                        final float temperature, final int topK,
                        final float topP) {
    float[] scores = new float[vocab];
    int base = logits.offset() + row * vocab;
    float max = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < vocab; i++) {
      float v = logits.data()[base + i] / temperature;
      scores[i] = v;
      if (v > max) {
        max = v;
      }
    }
    float sum = 0f;
    for (int i = 0; i < vocab; i++) {
      float e = (float) Math.exp(scores[i] - max);
      scores[i] = e;
      sum += e;
    }
    float inv = 1f / sum;
    for (int i = 0; i < vocab; i++) {
      scores[i] *= inv;
    }

    int[] order = this.sortedIndicesDesc(scores);
    if (topK > 0 && topK < vocab) {
      for (int i = topK; i < vocab; i++) {
        scores[order[i]] = 0f;
      }
      this.renormalize(scores);
      order = this.sortedIndicesDesc(scores);
    }
    if (topP < 1f) {
      float cum = 0f;
      for (int i = 0; i < vocab; i++) {
        int idx = order[i];
        if (cum >= topP && i > 0) {
          scores[idx] = 0f;
        } else {
          cum += scores[idx];
        }
      }
      this.renormalize(scores);
    }

    int best = 0;
    float bestScore = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < vocab; i++) {
      float p = scores[i];
      if (p <= 0f) {
        continue;
      }
      float u = ThreadLocalRandom.current().nextFloat();
      float g = (float) -Math.log(Math.max(1e-10, u));
      float s = p / Math.max(1e-10f, g);
      if (s > bestScore) {
        bestScore = s;
        best = i;
      }
    }
    return best;
  }

  private void renormalize(final float[] scores) {
    float sum = 0f;
    for (float s : scores) {
      sum += s;
    }
    if (sum <= 0f) {
      return;
    }
    float inv = 1f / sum;
    for (int i = 0; i < scores.length; i++) {
      scores[i] *= inv;
    }
  }

  private int[] sortedIndicesDesc(final float[] scores) {
    int n = scores.length;
    Integer[] idx = new Integer[n];
    for (int i = 0; i < n; i++) {
      idx[i] = i;
    }
    Arrays.sort(idx, (a, b) -> Float.compare(scores[b], scores[a]));
    int[] out = new int[n];
    for (int i = 0; i < n; i++) {
      out[i] = idx[i];
    }
    return out;
  }
}
