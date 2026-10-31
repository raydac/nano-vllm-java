package com.igormaznitsa.nanollvm.tensor.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;

/**
 * Device tasks for one decoder layer. They are chained in a single task graph so the activation
 * between matmul, norm, RoPE, and attention stays on the device.
 */
final class TornadoQwen3LayerKernels {

  static final int POSITION = 0;
  static final int SLOT = 1;
  static final int KEY_LENGTH = 2;
  static final int HAS_RESIDUAL = 3;
  static final int WINDOW = 4;
  static final int CACHE_OFFSET = 5;

  private TornadoQwen3LayerKernels() {
  }

  static void residualRms(
    final float[] x, final float[] residual, final float[] stream, final float[] normed,
    final float[] weight, final float[] eps, final int[] params
  ) {
    for (@Parallel int row = 0; row < 1; row++) {
      int useResidual = params[HAS_RESIDUAL];
      float epsilon = eps[0];
      int n = x.length - row;
      float energy = 0f;
      for (int i = 0; i < n; i++) {
        float value = useResidual == 0 ? x[i] : x[i] + residual[i];
        stream[i] = useResidual == 0 ? x[i] : value;
        energy += value * value;
      }
      float inv = 1f / (float) Math.sqrt(energy / (float) n + epsilon);
      for (int i = 0; i < n; i++) {
        float value = useResidual == 0 ? x[i] : x[i] + residual[i];
        normed[i] = value * inv * weight[i];
      }
    }
  }

  static void splitQkv(
    final float[] qkv, final float[] query, final float[] key, final float[] value
  ) {
    int querySize = query.length;
    int keySize = key.length;
    for (@Parallel int i = 0; i < qkv.length; i++) {
      if (i < querySize) {
        query[i] = qkv[i];
      } else if (i < querySize + keySize) {
        key[i - querySize] = qkv[i];
      } else {
        value[i - querySize - keySize] = qkv[i];
      }
    }
  }

  static void ropeHeads(
    final float[] in, final float[] out, final float[] angles, final int heads, final int headDim
  ) {
    int rotary = angles.length;
    int pairs = rotary / 2;
    for (@Parallel int head = 0; head < heads; head++) {
      int base = head * headDim;
      for (int i = 0; i < pairs; i++) {
        float left = in[base + i];
        float right = in[base + pairs + i];
        float cos = angles[i];
        float sin = angles[pairs + i];
        out[base + i] = left * cos - right * sin;
        out[base + pairs + i] = right * cos + left * sin;
      }
      for (int i = rotary; i < headDim; i++) {
        out[base + i] = in[base + i];
      }
    }
  }

  static void storeKv(
    final float[] key, final float[] value, final float[] keyCache, final float[] valueCache,
    final int[] params
  ) {
    int base = params[CACHE_OFFSET] + params[SLOT] * key.length;
    for (@Parallel int lane = 0; lane < key.length; lane++) {
      keyCache[base + lane] = key[lane];
      valueCache[base + lane] = value[lane];
    }
  }

  static void attendOne(
    final float[] query, final float[] keyCache, final float[] valueCache, final int[] slots,
    final float[] result, final float[] coeff, final int[] params,
    final int numHeads, final int numKvHeads, final int headDim
  ) {
    int keyLength = params[KEY_LENGTH];
    int window = params[WINDOW];
    int cacheOffset = params[CACHE_OFFSET];
    float scale = coeff[0];
    float unset = coeff[1];
    int repeats = numHeads / numKvHeads;
    int width = numKvHeads * headDim;
    int start = 0;
    if (window > 0) {
      start = keyLength - window;
      if (start < 0) {
        start = 0;
      }
    }
    for (@Parallel int head = 0; head < numHeads; head++) {
      int kvHead = head / repeats;
      int vector = head * headDim;
      for (int lane = 0; lane < headDim; lane++) {
        result[vector + lane] = 0f;
      }
      float max = unset;
      float sum = 0f;
      int seen = 0;
      for (int keyIndex = start; keyIndex < keyLength; keyIndex++) {
        int keyBase = cacheOffset + slots[keyIndex] * width + kvHead * headDim;
        float score = 0f;
        for (int lane = 0; lane < headDim; lane++) {
          score += query[vector + lane] * keyCache[keyBase + lane];
        }
        score *= scale;
        float nextMax = Math.max(score, max);
        float alpha = seen == 0 ? 0f : (float) Math.exp(max - nextMax);
        sum *= alpha;
        for (int lane = 0; lane < headDim; lane++) {
          result[vector + lane] *= alpha;
        }
        float weight = (float) Math.exp(score - nextMax);
        sum += weight;
        for (int lane = 0; lane < headDim; lane++) {
          result[vector + lane] += weight * valueCache[keyBase + lane];
        }
        seen = 1;
        max = nextMax;
      }
      float inverse = sum == 0f ? 0f : 1f / sum;
      for (int lane = 0; lane < headDim; lane++) {
        result[vector + lane] *= inverse;
      }
    }
  }

  static void siluHalves(final float[] gateUp, final float[] activated) {
    int half = activated.length;
    for (@Parallel int i = 0; i < half; i++) {
      float value = gateUp[i];
      activated[i] = (value / (1.0f + (float) Math.exp(-value))) * gateUp[half + i];
    }
  }

  static void siluMul(final float[] gate, final float[] up, final float[] activated) {
    for (@Parallel int i = 0; i < activated.length; i++) {
      float value = gate[i];
      activated[i] = (value / (1.0f + (float) Math.exp(-value))) * up[i];
    }
  }

  static void geluHalves(final float[] gateUp, final float[] activated) {
    int half = activated.length;
    for (@Parallel int i = 0; i < half; i++) {
      float value = gateUp[i];
      float gelu = 0.5f * value * (1.0f + (float) Math.tanh(
        0.79788456f * (value + 0.044715f * value * value * value)));
      activated[i] = gelu * gateUp[half + i];
    }
  }

  static void splitHalves(final float[] packed, final float[] left, final float[] right) {
    int half = left.length;
    for (@Parallel int i = 0; i < packed.length; i++) {
      if (i < half) {
        left[i] = packed[i];
      } else {
        right[i - half] = packed[i];
      }
    }
  }

  static void sigmoidMul(final float[] values, final float[] gate, final float[] dst) {
    for (@Parallel int i = 0; i < dst.length; i++) {
      float logit = gate[i];
      dst[i] = values[i] * (1.0f / (1.0f + (float) Math.exp(-logit)));
    }
  }

  static void srq(final float[] src, final float[] scale, final float[] dst) {
    float unit = scale[0];
    for (@Parallel int i = 0; i < dst.length; i++) {
      float quantized = (float) Math.floor(src[i] / unit + 0.5f);
      if (quantized > 127f) {
        quantized = 127f;
      }
      if (quantized < -128f) {
        quantized = -128f;
      }
      dst[i] = quantized * unit;
    }
  }
}
