package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * Position-sensitive attention used inside C2PSA and the attention C3k2.
 */
final class YoloAttention {

  private final YoloConv queryKeyValue;
  private final YoloConv projection;
  private final YoloConv position;
  private final int heads;
  private final int headDim;
  private final int keyDim;
  private final float scale;

  private YoloAttention(
    final YoloConv queryKeyValue,
    final YoloConv projection,
    final YoloConv position,
    final int heads,
    final int headDim,
    final int keyDim
  ) {
    this.queryKeyValue = queryKeyValue;
    this.projection = projection;
    this.position = position;
    this.heads = heads;
    this.headDim = headDim;
    this.keyDim = keyDim;
    this.scale = (float) (1.0 / Math.sqrt(keyDim));
  }

  static YoloAttention load(final YoloTensors weights, final String prefix, final int channels) {
    int heads = Math.max(channels / 64, 1);
    int headDim = channels / heads;
    int keyDim = (int) (headDim * 0.5f);
    if (keyDim < 1 || heads * headDim != channels) {
      throw new IllegalArgumentException("attention channels do not divide into heads");
    }
    int mixed = channels + heads * keyDim * 2;
    return new YoloAttention(
      YoloConv.load(weights, prefix + ".qkv", channels, mixed, 1, 1, 1, false),
      YoloConv.load(weights, prefix + ".proj", channels, channels, 1, 1, 1, false),
      YoloConv.load(weights, prefix + ".pe", channels, channels, 3, 1, channels, false),
      heads,
      headDim,
      keyDim);
  }

  Tensor forward(final Tensor input, final MatmulRuntime runtime) {
    int height = input.size(1);
    int width = input.size(2);
    int tokens = height * width;
    Tensor mixed = this.queryKeyValue.forward(input, runtime);
    float[] src = mixed.data();
    int off = mixed.offset();
    int channels = input.size(0);
    float[] attended = new float[channels * tokens];
    int queryKey = this.keyDim * 2 + this.headDim;
    for (int head = 0; head < this.heads; head++) {
      this.attendHead(src, off, head, queryKey, tokens, height, width, attended, runtime);
    }
    Tensor values = Tensor.of(attended, channels, height, width);
    Tensor positioned = YoloMath.add(values, this.position.forward(values, runtime));
    return this.projection.forward(positioned, runtime);
  }

  private void attendHead(
    final float[] mixed,
    final int offset,
    final int head,
    final int queryKey,
    final int tokens,
    final int height,
    final int width,
    final float[] attended,
    final MatmulRuntime runtime
  ) {
    int headBase = head * queryKey;
    float[] query = this.packTokens(mixed, offset, headBase, this.keyDim, tokens, height, width);
    float[] key = this.packTokens(
      mixed, offset, headBase + this.keyDim, this.keyDim, tokens, height, width);
    float[] value = this.packTokens(
      mixed, offset, headBase + this.keyDim * 2, this.headDim, tokens, height, width);
    float[] scores = new float[tokens * tokens];
    runtime.linear(query, 0, key, 0, null, scores, 0, tokens, this.keyDim, tokens);
    float[] sums = new float[tokens];
    runtime.parallelRanges(tokens, (query0, query1) -> {
      for (int q = query0; q < query1; q++) {
        int row = q * tokens;
        float max = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < tokens; k++) {
          float scaled = scores[row + k] * this.scale;
          scores[row + k] = scaled;
          if (scaled > max) {
            max = scaled;
          }
        }
        float sum = 0f;
        for (int k = 0; k < tokens; k++) {
          float exp = (float) Math.exp(scores[row + k] - max);
          scores[row + k] = exp;
          sum += exp;
        }
        sums[q] = sum;
      }
    });
    float[] valueByChannel = new float[this.headDim * tokens];
    for (int token = 0; token < tokens; token++) {
      int src = token * this.headDim;
      for (int d = 0; d < this.headDim; d++) {
        valueByChannel[d * tokens + token] = value[src + d];
      }
    }
    float[] mixedOut = new float[tokens * this.headDim];
    runtime.linear(
      scores, 0, valueByChannel, 0, null, mixedOut, 0, tokens, tokens, this.headDim);
    int outBase = head * this.headDim;
    int plane = height * width;
    for (int q = 0; q < tokens; q++) {
      float inv = 1f / sums[q];
      int y = q / width;
      int x = q - y * width;
      int pixel = y * width + x;
      int src = q * this.headDim;
      for (int d = 0; d < this.headDim; d++) {
        attended[(outBase + d) * plane + pixel] = mixedOut[src + d] * inv;
      }
    }
  }

  private float[] packTokens(
    final float[] mixed,
    final int offset,
    final int channel0,
    final int count,
    final int tokens,
    final int height,
    final int width
  ) {
    float[] packed = new float[tokens * count];
    int plane = height * width;
    for (int token = 0; token < tokens; token++) {
      int y = token / width;
      int x = token - y * width;
      int pixel = y * width + x;
      int dst = token * count;
      for (int d = 0; d < count; d++) {
        packed[dst + d] = mixed[offset + (channel0 + d) * plane + pixel];
      }
    }
    return packed;
  }
}
