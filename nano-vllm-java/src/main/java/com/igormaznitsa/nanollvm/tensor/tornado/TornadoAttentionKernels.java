package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * One TornadoVM task for a full query range of causal grouped-query attention.
 *
 * @since 1.5.0
 */
final class TornadoAttentionKernels {

  private TornadoAttentionKernels() {
  }

  static void scoreHeads(
    final float[] query,
    final float[] key,
    final float[] value,
    final float[] result,
    final int[] headOfJob,
    final int[] queryOfJob,
    final int[] kvHeadOfJob,
    final float[] params
  ) {
    int qLen = (int) params[0];
    int kLen = (int) params[1];
    int numHeads = (int) params[2];
    int numKvHeads = (int) params[3];
    int headDim = (int) params[4];
    float scale = params[5];
    int slidingWindow = (int) params[6];
    int causal = (int) params[7];
    float unset = params[8];
    for (@Parallel int job = 0; job < headOfJob.length; job++) {
      int head = headOfJob[job];
      int queryIndex = queryOfJob[job];
      int kvHead = kvHeadOfJob[job];
      int causalEnd = causal != 0 ? (kLen - qLen + queryIndex + 1) : kLen;
      int absoluteQuery = causal != 0 ? (kLen - qLen + queryIndex) : (kLen - 1);
      int causalStart = 0;
      if (slidingWindow > 0) {
        causalStart = absoluteQuery - slidingWindow + 1;
        if (causalStart < 0) {
          causalStart = 0;
        }
      }
      int vector = (queryIndex * numHeads + head) * headDim;
      for (int lane = 0; lane < headDim; lane++) {
        result[vector + lane] = 0f;
      }
      float max = unset;
      float sum = 0f;
      int seenScore = 0;
      for (int keyIndex = causalStart; keyIndex < causalEnd; keyIndex++) {
        int keyBase = (keyIndex * numKvHeads + kvHead) * headDim;
        float score = 0f;
        for (int lane = 0; lane < headDim; lane++) {
          score += query[vector + lane] * key[keyBase + lane];
        }
        score *= scale;
        float nextMax = score > max ? score : max;
        float alpha = seenScore == 0 ? 0f : (float) Math.exp(max - nextMax);
        sum *= alpha;
        for (int lane = 0; lane < headDim; lane++) {
          result[vector + lane] *= alpha;
        }
        float weight = (float) Math.exp(score - nextMax);
        sum += weight;
        for (int lane = 0; lane < headDim; lane++) {
          result[vector + lane] += weight * value[keyBase + lane];
        }
        seenScore = 1;
        max = nextMax;
      }
      float inverse = sum == 0f ? 0f : 1f / sum;
      for (int lane = 0; lane < headDim; lane++) {
        result[vector + lane] *= inverse;
      }
    }
  }
}
