package com.igormaznitsa.nanollvm.tensor.tornado;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.DeviceDecoders;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import org.junit.jupiter.api.Test;

class TornadoQwen3DecodeTest {

  private static Context prefill(final int tokens) {
    Context context = new Context();
    context.set(
      true, new int[] {0, tokens}, new int[] {0, tokens}, tokens, tokens,
      new int[] {0, 1}, null, null, new int[] {1, 1});
    return context;
  }

  private static void oracle(
    final FloatKernels scalar, final RotaryEmbedding rope,
    final float[] hiddenIn, final float[] residualIn,
    final float[] qkv, final float[] output, final float[] gate, final float[] down,
    final float[] inputNorm, final float[] postNorm, final float[] queryNorm, final float[] keyNorm,
    final float[] keyCache, final float[] valueCache,
    final float[] hiddenOut, final float[] residualOut,
    final int tokens, final int hidden, final int heads, final int kvHeads, final int headDim,
    final int intermediate, final float epsilon, final float scale
  ) {
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    float[] angles = new float[headDim];
    for (int token = 0; token < tokens; token++) {
      int row = token * hidden;
      float[] x = copy(hiddenIn, row, hidden);
      float[] residual = copy(residualIn, row, hidden);
      float[] stream = add(x, residual);
      float[] normed = rms(stream, inputNorm, epsilon);
      float[] qkvOut = new float[querySize + keySize * 2];
      scalar.gemv(normed, 0, qkv, 0, null, qkvOut, 0, hidden, 0, qkvOut.length);
      float[] query = copy(qkvOut, 0, querySize);
      float[] key = copy(qkvOut, querySize, keySize);
      float[] value = copy(qkvOut, querySize + keySize, keySize);
      if (queryNorm != null) {
        query = rmsHeads(query, queryNorm, heads, headDim, epsilon, false);
        key = rmsHeads(key, keyNorm, kvHeads, headDim, epsilon, false);
      }
      rope.copyAngles(token, angles);
      float[] rotatedQuery = rotate(query, angles, heads, headDim);
      float[] rotatedKey = rotate(key, angles, kvHeads, headDim);
      System.arraycopy(rotatedKey, 0, keyCache, token * keySize, keySize);
      System.arraycopy(value, 0, valueCache, token * keySize, keySize);
      float[] attended = attend(rotatedQuery, keyCache, valueCache, token + 1, heads, kvHeads,
        headDim, scale);
      float[] projected = new float[hidden];
      scalar.gemv(attended, 0, output, 0, null, projected, 0, querySize, 0, hidden);
      float[] carried = add(projected, stream);
      float[] post = rms(carried, postNorm, epsilon);
      float[] gateOut = new float[intermediate * 2];
      scalar.gemv(post, 0, gate, 0, null, gateOut, 0, hidden, 0, gateOut.length);
      float[] activated = silu(gateOut);
      scalar.gemv(activated, 0, down, 0, null, hiddenOut, row, intermediate, 0, hidden);
      System.arraycopy(carried, 0, residualOut, row, hidden);
    }
  }

  private static float[] attend(
    final float[] query, final float[] key, final float[] value, final int keyLength,
    final int heads, final int kvHeads, final int headDim, final float scale
  ) {
    float[] result = new float[query.length];
    int repeats = heads / kvHeads;
    for (int head = 0; head < heads; head++) {
      int kvHead = head / repeats;
      int vector = head * headDim;
      float max = Float.NEGATIVE_INFINITY;
      float sum = 0f;
      float[] acc = new float[headDim];
      for (int keyIndex = 0; keyIndex < keyLength; keyIndex++) {
        int keyBase = (keyIndex * kvHeads + kvHead) * headDim;
        float score = 0f;
        for (int lane = 0; lane < headDim; lane++) {
          score += query[vector + lane] * key[keyBase + lane];
        }
        score *= scale;
        float nextMax = Math.max(score, max);
        float alpha = max == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(max - nextMax);
        sum *= alpha;
        for (int lane = 0; lane < headDim; lane++) {
          acc[lane] *= alpha;
        }
        float weight = (float) Math.exp(score - nextMax);
        sum += weight;
        for (int lane = 0; lane < headDim; lane++) {
          acc[lane] += weight * value[keyBase + lane];
        }
        max = nextMax;
      }
      float inverse = sum == 0f ? 0f : 1f / sum;
      for (int lane = 0; lane < headDim; lane++) {
        result[vector + lane] = acc[lane] * inverse;
      }
    }
    return result;
  }

  private static float[] rotate(
    final float[] in, final float[] angles, final int heads, final int headDim
  ) {
    float[] out = new float[in.length];
    int pairs = headDim / 2;
    for (int head = 0; head < heads; head++) {
      int base = head * headDim;
      for (int i = 0; i < pairs; i++) {
        float left = in[base + i];
        float right = in[base + pairs + i];
        float cos = angles[i];
        float sin = angles[pairs + i];
        out[base + i] = left * cos - right * sin;
        out[base + pairs + i] = right * cos + left * sin;
      }
    }
    return out;
  }

  private static void queryGateOracle(
    final FloatKernels scalar, final RotaryEmbedding rope,
    final float[] hiddenIn, final float[] residualIn,
    final float[] queryWeight, final float[] keyWeight, final float[] valueWeight,
    final float[] output, final float[] gate, final float[] down,
    final float[] inputNorm, final float[] postNorm, final float[] queryNorm, final float[] keyNorm,
    final float[] keyCache, final float[] valueCache,
    final float[] hiddenOut, final float[] residualOut,
    final int tokens, final int hidden, final int heads, final int kvHeads, final int headDim,
    final int intermediate, final float epsilon, final float scale
  ) {
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    float[] angles = new float[headDim];
    for (int token = 0; token < tokens; token++) {
      int row = token * hidden;
      float[] stream = add(copy(hiddenIn, row, hidden), copy(residualIn, row, hidden));
      float[] normed = rms(stream, inputNorm, epsilon);
      float[] packed = new float[querySize * 2];
      scalar.gemv(normed, 0, queryWeight, 0, null, packed, 0, hidden, 0, packed.length);
      float[] query =
        rmsHeads(copy(packed, 0, querySize), queryNorm, heads, headDim, epsilon, false);
      float[] logit = copy(packed, querySize, querySize);
      float[] key = new float[keySize];
      float[] value = new float[keySize];
      scalar.gemv(normed, 0, keyWeight, 0, null, key, 0, hidden, 0, keySize);
      scalar.gemv(normed, 0, valueWeight, 0, null, value, 0, hidden, 0, keySize);
      key = rmsHeads(key, keyNorm, kvHeads, headDim, epsilon, false);
      rope.copyAngles(token, angles);
      float[] rotatedQuery = rotate(query, angles, heads, headDim);
      float[] rotatedKey = rotate(key, angles, kvHeads, headDim);
      System.arraycopy(rotatedKey, 0, keyCache, token * keySize, keySize);
      System.arraycopy(value, 0, valueCache, token * keySize, keySize);
      float[] attended = attend(rotatedQuery, keyCache, valueCache, token + 1, heads, kvHeads,
        headDim, scale);
      for (int i = 0; i < attended.length; i++) {
        float g = logit[i];
        attended[i] *= (float) (1.0 / (1.0 + Math.exp(-g)));
      }
      float[] projected = new float[hidden];
      scalar.gemv(attended, 0, output, 0, null, projected, 0, querySize, 0, hidden);
      float[] carried = add(projected, stream);
      float[] post = rms(carried, postNorm, epsilon);
      float[] gateOut = new float[intermediate * 2];
      scalar.gemv(post, 0, gate, 0, null, gateOut, 0, hidden, 0, gateOut.length);
      scalar.gemv(silu(gateOut), 0, down, 0, null, hiddenOut, row, intermediate, 0, hidden);
      System.arraycopy(carried, 0, residualOut, row, hidden);
    }
  }

  private static float[] plainOracle(
    final FloatKernels scalar, final RotaryEmbedding rope,
    final float[] hiddenIn, final float[] qkv, final float[] output, final float[] gate,
    final float[] down, final float[] inputNorm, final float[] preMlp,
    final float[] queryNorm, final float[] keyNorm, final float[] keyCache,
    final float[] valueCache,
    final int tokens, final int hidden, final int heads, final int kvHeads, final int headDim,
    final int intermediate, final float epsilon, final float scale
  ) {
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    float[] angles = new float[headDim];
    float[] hiddenOut = new float[tokens * hidden];
    for (int token = 0; token < tokens; token++) {
      int row = token * hidden;
      float[] x = copy(hiddenIn, row, hidden);
      float[] normed = rms(x, inputNorm, epsilon);
      float[] qkvOut = new float[querySize + keySize * 2];
      scalar.gemv(normed, 0, qkv, 0, null, qkvOut, 0, hidden, 0, qkvOut.length);
      float[] query =
        rmsHeads(copy(qkvOut, 0, querySize), queryNorm, heads, headDim, epsilon, false);
      float[] key = rmsHeads(
        copy(qkvOut, querySize, keySize), keyNorm, kvHeads, headDim, epsilon, false);
      float[] value = copy(qkvOut, querySize + keySize, keySize);
      rope.copyAngles(token, angles);
      float[] rotatedQuery = rotate(query, angles, heads, headDim);
      float[] rotatedKey = rotate(key, angles, kvHeads, headDim);
      System.arraycopy(rotatedKey, 0, keyCache, token * keySize, keySize);
      System.arraycopy(value, 0, valueCache, token * keySize, keySize);
      float[] attended = attend(rotatedQuery, keyCache, valueCache, token + 1, heads, kvHeads,
        headDim, scale);
      float[] projected = new float[hidden];
      scalar.gemv(attended, 0, output, 0, null, projected, 0, querySize, 0, hidden);
      float[] carried = add(x, projected);
      float[] post = rms(carried, preMlp, epsilon);
      float[] gateOut = new float[intermediate * 2];
      scalar.gemv(post, 0, gate, 0, null, gateOut, 0, hidden, 0, gateOut.length);
      float[] mlpOut = new float[hidden];
      scalar.gemv(silu(gateOut), 0, down, 0, null, mlpOut, 0, intermediate, 0, hidden);
      System.arraycopy(add(carried, mlpOut), 0, hiddenOut, row, hidden);
    }
    return hiddenOut;
  }

  private static float[] gemmaOracle(
    final FloatKernels scalar, final RotaryEmbedding rope,
    final float[] hiddenIn, final float[] qkv, final float[] output, final float[] gate,
    final float[] down, final float[] inputNorm, final float[] postAttn, final float[] preMlp,
    final float[] postMlp, final float[] queryNorm, final float[] keyNorm,
    final float[] keyCache, final float[] valueCache,
    final int tokens, final int hidden, final int heads, final int kvHeads, final int headDim,
    final int intermediate, final float epsilon, final float scale
  ) {
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    float[] angles = new float[headDim];
    float[] hiddenOut = new float[tokens * hidden];
    for (int token = 0; token < tokens; token++) {
      int row = token * hidden;
      float[] x = copy(hiddenIn, row, hidden);
      float[] normed = rms(x, inputNorm, epsilon, true);
      float[] qkvOut = new float[querySize + keySize * 2];
      scalar.gemv(normed, 0, qkv, 0, null, qkvOut, 0, hidden, 0, qkvOut.length);
      float[] query =
        rmsHeads(copy(qkvOut, 0, querySize), queryNorm, heads, headDim, epsilon, true);
      float[] key = rmsHeads(
        copy(qkvOut, querySize, keySize), keyNorm, kvHeads, headDim, epsilon, true);
      float[] value = copy(qkvOut, querySize + keySize, keySize);
      rope.copyAngles(token, angles);
      float[] rotatedQuery = rotate(query, angles, heads, headDim);
      float[] rotatedKey = rotate(key, angles, kvHeads, headDim);
      System.arraycopy(rotatedKey, 0, keyCache, token * keySize, keySize);
      System.arraycopy(value, 0, valueCache, token * keySize, keySize);
      float[] attended = attend(rotatedQuery, keyCache, valueCache, token + 1, heads, kvHeads,
        headDim, scale);
      float[] projected = new float[hidden];
      scalar.gemv(attended, 0, output, 0, null, projected, 0, querySize, 0, hidden);
      float[] carried = add(x, rms(projected, postAttn, epsilon, true));
      float[] mlpIn = rms(carried, preMlp, epsilon, true);
      float[] gateOut = new float[intermediate * 2];
      scalar.gemv(mlpIn, 0, gate, 0, null, gateOut, 0, hidden, 0, gateOut.length);
      float[] activated = Ops.geluPytorchTanhAndMul(Tensor.of(gateOut, gateOut.length)).data();
      float[] mlpOut = new float[hidden];
      scalar.gemv(activated, 0, down, 0, null, mlpOut, 0, intermediate, 0, hidden);
      System.arraycopy(add(carried, rms(mlpOut, postMlp, epsilon, true)), 0, hiddenOut, row,
        hidden);
    }
    return hiddenOut;
  }

  private static float[] rmsHeads(
    final float[] src, final float[] weight, final int heads, final int headDim,
    final float epsilon, final boolean onePlus
  ) {
    return Ops.rmsNorm(Tensor.of(src, heads, headDim), Tensor.of(weight, headDim), epsilon, onePlus)
      .data();
  }

  private static float[] rms(
    final float[] src, final float[] weight, final float epsilon, final boolean onePlus
  ) {
    return Ops.rmsNorm(
      Tensor.of(src, src.length), Tensor.of(weight, weight.length), epsilon, onePlus).data();
  }

  private static float[] rms(final float[] src, final float[] weight, final float epsilon) {
    return rms(src, weight, epsilon, false);
  }

  private static float[] silu(final float[] gateUp) {
    return Ops.siluAndMul(Tensor.of(gateUp, gateUp.length)).data();
  }

  private static float[] add(final float[] left, final float[] right) {
    float[] sum = new float[left.length];
    for (int i = 0; i < left.length; i++) {
      sum[i] = left[i] + right[i];
    }
    return sum;
  }

  private static float[] copy(final float[] source, final int offset, final int length) {
    float[] values = new float[length];
    System.arraycopy(source, offset, values, 0, length);
    return values;
  }

  private static Tensor matrix(final int rows, final int columns, final float scale) {
    return Tensor.of(pattern(rows * columns, scale), rows, columns);
  }

  private static Tensor vector(final int length, final float scale) {
    return Tensor.of(pattern(length, scale), length);
  }

  private static float[] pattern(final int length, final float scale) {
    float[] values = new float[length];
    for (int i = 0; i < length; i++) {
      values[i] = ((i % 5) - 2) * scale;
    }
    return values;
  }

  private static void assertClose(final float[] expected, final float[] actual) {
    assertEquals(expected.length, actual.length);
    for (int i = 0; i < expected.length; i++) {
      assertEquals(expected[i], actual[i], 2.0e-3f, "index " + i);
    }
  }

  @Test
  void oneLayerStaysOnTheDevice() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");

    int hidden = 8;
    int heads = 4;
    int kvHeads = 2;
    int headDim = 4;
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    int intermediate = 8;
    int tokens = 2;
    float epsilon = 1.0e-5f;
    float scale = (float) Math.pow(headDim, -0.5f);
    RotaryEmbedding rope = RotaryEmbedding.of(headDim, headDim, 16, 10_000f);
    FloatKernels scalar = FloatKernelsFactory.create("scalar");

    Tensor qkv = matrix(querySize + keySize * 2, hidden, 0.02f);
    Tensor output = matrix(hidden, querySize, 0.03f);
    Tensor gate = matrix(intermediate * 2, hidden, 0.04f);
    Tensor down = matrix(hidden, intermediate, 0.05f);
    Tensor inputNorm = vector(hidden, 0.1f);
    Tensor postNorm = vector(hidden, 0.2f);
    Tensor queryNorm = vector(headDim, 0.3f);
    Tensor keyNorm = vector(headDim, 0.4f);
    Tensor hiddenIn = matrix(tokens, hidden, 0.15f);
    Tensor residualIn = matrix(tokens, hidden, 0.05f);
    Tensor positions = Tensor.of(new float[] {0f, 1f}, tokens);
    Tensor keyCache = Tensor.zeros(1, 4, kvHeads, headDim);
    Tensor valueCache = Tensor.zeros(1, 4, kvHeads, headDim);
    float[] expectedHidden = new float[tokens * hidden];
    float[] expectedResidual = new float[tokens * hidden];
    float[] cpuKey = new float[keyCache.numel()];
    float[] cpuValue = new float[valueCache.numel()];
    oracle(
      scalar, rope, hiddenIn.data(), residualIn.data(), qkv.data(), output.data(), gate.data(),
      down.data(), inputNorm.data(), postNorm.data(), queryNorm.data(), keyNorm.data(),
      cpuKey, cpuValue, expectedHidden, expectedResidual,
      tokens, hidden, heads, kvHeads, headDim, intermediate, epsilon, scale
    );

    Context context = new Context();
    context.set(
      true, new int[] {0, tokens}, new int[] {0, tokens}, tokens, tokens,
      new int[] {0, 1}, null, null, new int[] {1, 1});
    Tensor[] actual = TornadoDecoder.forward(DeviceDecoders.layer()
      .highway()
      .silu()
      .positions(positions)
      .hidden(hiddenIn)
      .residual(residualIn)
      .context(context)
      .qkv(qkv)
      .output(output)
      .gateUp(gate)
      .down(down)
      .inputNorm(inputNorm, false)
      .postAttentionNorm(postNorm, false)
      .queryNorm(queryNorm, false)
      .keyNorm(keyNorm, false)
      .epsilon(epsilon)
      .rope(rope)
      .heads(heads, kvHeads, headDim)
      .scoreScale(scale)
      .window(0)
      .caches(keyCache, valueCache)
      .build());

    assertClose(expectedHidden, actual[0].data());
    assertClose(expectedResidual, actual[1].data());
  }

  @Test
  void llamaLayerOmitsHeadNorms() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");
    int hidden = 8;
    int heads = 4;
    int kvHeads = 2;
    int headDim = 4;
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    int intermediate = 8;
    int tokens = 2;
    float epsilon = 1.0e-5f;
    float scale = (float) Math.pow(headDim, -0.5f);
    RotaryEmbedding rope = RotaryEmbedding.of(headDim, headDim, 16, 10_000f);
    FloatKernels scalar = FloatKernelsFactory.create("scalar");
    Tensor qkv = matrix(querySize + keySize * 2, hidden, 0.02f);
    Tensor output = matrix(hidden, querySize, 0.03f);
    Tensor gate = matrix(intermediate * 2, hidden, 0.04f);
    Tensor down = matrix(hidden, intermediate, 0.05f);
    Tensor inputNorm = vector(hidden, 0.1f);
    Tensor postNorm = vector(hidden, 0.2f);
    Tensor hiddenIn = matrix(tokens, hidden, 0.15f);
    Tensor residualIn = matrix(tokens, hidden, 0.05f);
    Tensor positions = Tensor.of(new float[] {0f, 1f}, tokens);
    Tensor keyCache = Tensor.zeros(1, 4, kvHeads, headDim);
    Tensor valueCache = Tensor.zeros(1, 4, kvHeads, headDim);
    float[] expectedHidden = new float[tokens * hidden];
    float[] expectedResidual = new float[tokens * hidden];
    oracle(
      scalar, rope, hiddenIn.data(), residualIn.data(), qkv.data(), output.data(), gate.data(),
      down.data(), inputNorm.data(), postNorm.data(), null, null,
      new float[keyCache.numel()], new float[valueCache.numel()], expectedHidden, expectedResidual,
      tokens, hidden, heads, kvHeads, headDim, intermediate, epsilon, scale
    );
    Context context = prefill(tokens);
    Tensor[] actual = TornadoDecoder.forward(DeviceDecoders.layer()
      .highway()
      .silu()
      .positions(positions)
      .hidden(hiddenIn)
      .residual(residualIn)
      .context(context)
      .qkv(qkv)
      .output(output)
      .gateUp(gate)
      .down(down)
      .inputNorm(inputNorm, false)
      .postAttentionNorm(postNorm, false)
      .epsilon(epsilon)
      .rope(rope)
      .heads(heads, kvHeads, headDim)
      .scoreScale(scale)
      .caches(keyCache, valueCache)
      .build());
    assertClose(expectedHidden, actual[0].data());
    assertClose(expectedResidual, actual[1].data());
  }

  @Test
  void gemmaBranchNormsTheAttentionAndMlp() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");
    int hidden = 8;
    int heads = 4;
    int kvHeads = 2;
    int headDim = 4;
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    int intermediate = 8;
    int tokens = 2;
    float epsilon = 1.0e-5f;
    float scale = (float) Math.pow(headDim, -0.5f);
    RotaryEmbedding rope = RotaryEmbedding.of(headDim, headDim, 16, 10_000f);
    FloatKernels scalar = FloatKernelsFactory.create("scalar");
    Tensor qkv = matrix(querySize + keySize * 2, hidden, 0.02f);
    Tensor output = matrix(hidden, querySize, 0.03f);
    Tensor gate = matrix(intermediate * 2, hidden, 0.04f);
    Tensor down = matrix(hidden, intermediate, 0.05f);
    Tensor inputNorm = vector(hidden, 0.1f);
    Tensor postAttn = vector(hidden, 0.2f);
    Tensor preMlp = vector(hidden, 0.15f);
    Tensor postMlp = vector(hidden, 0.25f);
    Tensor queryNorm = vector(headDim, 0.3f);
    Tensor keyNorm = vector(headDim, 0.4f);
    Tensor hiddenIn = matrix(tokens, hidden, 0.15f);
    Tensor positions = Tensor.of(new float[] {0f, 1f}, tokens);
    Tensor keyCache = Tensor.zeros(1, 4, kvHeads, headDim);
    Tensor valueCache = Tensor.zeros(1, 4, kvHeads, headDim);
    float[] expected = gemmaOracle(
      scalar, rope, hiddenIn.data(), qkv.data(), output.data(), gate.data(), down.data(),
      inputNorm.data(), postAttn.data(), preMlp.data(), postMlp.data(),
      queryNorm.data(), keyNorm.data(), new float[keyCache.numel()], new float[valueCache.numel()],
      tokens, hidden, heads, kvHeads, headDim, intermediate, epsilon, scale
    );
    Tensor[] actual = TornadoDecoder.forward(DeviceDecoders.layer()
      .normBranch()
      .gelu()
      .positions(positions)
      .hidden(hiddenIn)
      .context(prefill(tokens))
      .qkv(qkv)
      .output(output)
      .gateUp(gate)
      .down(down)
      .inputNorm(inputNorm, true)
      .postAttentionNorm(postAttn, true)
      .preMlpNorm(preMlp, true)
      .postMlpNorm(postMlp, true)
      .queryNorm(queryNorm, true)
      .keyNorm(keyNorm, true)
      .epsilon(epsilon)
      .rope(rope)
      .heads(heads, kvHeads, headDim)
      .scoreScale(scale)
      .caches(keyCache, valueCache)
      .build());
    assertClose(expected, actual[0].data());
  }

  @Test
  void queryGateAndPlainBranchStayOnTheDevice() {
    assumeTrue(TornadoFloatKernelsProvider.isAvailable(), "TornadoVM device required");
    int hidden = 8;
    int heads = 4;
    int kvHeads = 2;
    int headDim = 4;
    int querySize = heads * headDim;
    int keySize = kvHeads * headDim;
    int intermediate = 8;
    int tokens = 2;
    float epsilon = 1.0e-5f;
    float scale = (float) Math.pow(headDim, -0.5f);
    RotaryEmbedding rope = RotaryEmbedding.of(headDim, headDim, 16, 10_000f);
    FloatKernels scalar = FloatKernelsFactory.create("scalar");
    Tensor query = matrix(querySize * 2, hidden, 0.02f);
    Tensor key = matrix(keySize, hidden, 0.03f);
    Tensor value = matrix(keySize, hidden, 0.04f);
    Tensor output = matrix(hidden, querySize, 0.05f);
    Tensor gate = matrix(intermediate * 2, hidden, 0.06f);
    Tensor down = matrix(hidden, intermediate, 0.07f);
    Tensor inputNorm = vector(hidden, 0.1f);
    Tensor postNorm = vector(hidden, 0.2f);
    Tensor preMlp = vector(hidden, 0.18f);
    Tensor queryNorm = vector(headDim, 0.3f);
    Tensor keyNorm = vector(headDim, 0.4f);
    Tensor hiddenIn = matrix(tokens, hidden, 0.15f);
    Tensor residualIn = matrix(tokens, hidden, 0.05f);
    Tensor positions = Tensor.of(new float[] {0f, 1f}, tokens);
    Tensor keyCache = Tensor.zeros(1, 4, kvHeads, headDim);
    Tensor valueCache = Tensor.zeros(1, 4, kvHeads, headDim);
    float[] gatedHidden = new float[tokens * hidden];
    float[] gatedResidual = new float[tokens * hidden];
    queryGateOracle(
      scalar, rope, hiddenIn.data(), residualIn.data(), query.data(), key.data(), value.data(),
      output.data(), gate.data(), down.data(), inputNorm.data(), postNorm.data(),
      queryNorm.data(), keyNorm.data(), new float[keyCache.numel()], new float[valueCache.numel()],
      gatedHidden, gatedResidual, tokens, hidden, heads, kvHeads, headDim, intermediate, epsilon,
      scale
    );
    Tensor[] gated = TornadoDecoder.forward(DeviceDecoders.layer()
      .highway()
      .silu()
      .queryGate()
      .positions(positions)
      .hidden(hiddenIn)
      .residual(residualIn)
      .context(prefill(tokens))
      .query(query)
      .key(key)
      .value(value)
      .output(output)
      .queryNorm(queryNorm, false)
      .keyNorm(keyNorm, false)
      .gateUp(gate)
      .down(down)
      .inputNorm(inputNorm, false)
      .postAttentionNorm(postNorm, false)
      .epsilon(epsilon)
      .rope(rope)
      .heads(heads, kvHeads, headDim)
      .scoreScale(scale)
      .caches(keyCache, valueCache)
      .build());
    assertClose(gatedHidden, gated[0].data());
    assertClose(gatedResidual, gated[1].data());

    Tensor plainCacheK = Tensor.zeros(1, 4, kvHeads, headDim);
    Tensor plainCacheV = Tensor.zeros(1, 4, kvHeads, headDim);
    Tensor qkv = matrix(querySize + keySize * 2, hidden, 0.02f);
    float[] plain = plainOracle(
      scalar, rope, hiddenIn.data(), qkv.data(), output.data(), gate.data(), down.data(),
      inputNorm.data(), preMlp.data(), queryNorm.data(), keyNorm.data(),
      new float[plainCacheK.numel()], new float[plainCacheV.numel()],
      tokens, hidden, heads, kvHeads, headDim, intermediate, epsilon, scale
    );
    Tensor[] actual = TornadoDecoder.forward(DeviceDecoders.layer()
      .plainBranch()
      .silu()
      .positions(positions)
      .hidden(hiddenIn)
      .context(prefill(tokens))
      .qkv(qkv)
      .output(output)
      .queryNorm(queryNorm, false)
      .keyNorm(keyNorm, false)
      .gateUp(gate)
      .down(down)
      .inputNorm(inputNorm, false)
      .preMlpNorm(preMlp, false)
      .epsilon(epsilon)
      .rope(rope)
      .heads(heads, kvHeads, headDim)
      .scoreScale(scale)
      .caches(plainCacheK, plainCacheV)
      .build());
    assertClose(plain, actual[0].data());
  }
}
