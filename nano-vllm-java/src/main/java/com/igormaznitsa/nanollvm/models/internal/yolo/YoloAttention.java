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
      this.attendHead(src, off, head, queryKey, tokens, height, width, attended);
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
    final float[] attended
  ) {
    int headBase = head * queryKey;
    float[] scores = new float[tokens * tokens];
    float[] sums = new float[tokens];
    for (int query = 0; query < tokens; query++) {
      float max = Float.NEGATIVE_INFINITY;
      for (int key = 0; key < tokens; key++) {
        float dot = 0f;
        for (int d = 0; d < this.keyDim; d++) {
          dot += this.at(mixed, offset, headBase + d, query, height, width)
            * this.at(mixed, offset, headBase + this.keyDim + d, key, height, width);
        }
        float scaled = dot * this.scale;
        scores[query * tokens + key] = scaled;
        max = Math.max(max, scaled);
      }
      float sum = 0f;
      for (int key = 0; key < tokens; key++) {
        float exp = (float) Math.exp(scores[query * tokens + key] - max);
        scores[query * tokens + key] = exp;
        sum += exp;
      }
      sums[query] = sum;
    }
    int valueBase = headBase + this.keyDim * 2;
    int outBase = head * this.headDim;
    for (int d = 0; d < this.headDim; d++) {
      for (int query = 0; query < tokens; query++) {
        float acc = 0f;
        for (int key = 0; key < tokens; key++) {
          acc += this.at(mixed, offset, valueBase + d, key, height, width)
            * scores[query * tokens + key];
        }
        int y = query / width;
        int x = query - y * width;
        attended[((outBase + d) * height + y) * width + x] = acc / sums[query];
      }
    }
  }

  private float at(
    final float[] mixed,
    final int offset,
    final int channel,
    final int token,
    final int height,
    final int width
  ) {
    int y = token / width;
    int x = token - y * width;
    return mixed[offset + (channel * height + y) * width + x];
  }
}
