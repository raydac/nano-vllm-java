package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.internal.GgufDequant;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.RotaryEmbedding;
import com.igormaznitsa.nanollvm.layers.ShortConv;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.models.internal.PackedWeight;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufReader;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.tensor.EmbeddingKernel;
import com.igormaznitsa.nanollvm.tensor.LinearKernel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class GgufUnitTest {

  @Test
  void dequantQ4_0KnownBlock() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q4_0).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.5f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x88);
    }
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, 2, GgufDequant.QK4_0);
    assertEquals(GgufDequant.QK4_0, out.length);
    assertEquals(0f, out[0], 1e-5f);
    assertEquals(0f, out[16], 1e-5f);
  }

  @Test
  void dequantF16RoundTrip() {
    ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(1.0f));
    buf.putShort(Float.floatToFloat16(-2.0f));
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, 1, 2);
    assertEquals(1.0f, out[0], 1e-3f);
    assertEquals(-2.0f, out[1], 1e-3f);
  }

  @Test
  void dequantQ4_1KnownBlock() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q4_1).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.5f));
    buf.putShort(Float.floatToFloat16(-1f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x21);
    }
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, GgufDequant.TYPE_Q4_1, GgufDequant.QK4_1);
    assertEquals(1 * 0.5f - 1f, out[0], 1e-4f);
    assertEquals(2 * 0.5f - 1f, out[16], 1e-4f);
  }

  @Test
  void dequantQ8_1MatchesQ8_0Scale() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q8_1).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.25f));
    buf.putShort(Float.floatToFloat16(0f));
    for (int i = 0; i < 32; i++) {
      buf.put((byte) 4);
    }
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, GgufDequant.TYPE_Q8_1, GgufDequant.QK8_1);
    assertEquals(1f, out[0], 1e-4f);
    assertEquals(1f, out[31], 1e-4f);
  }

  @Test
  void dequantQ8_KUsesFloatScale() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q8_K).order(ByteOrder.LITTLE_ENDIAN);
    buf.putFloat(0.5f);
    for (int i = 0; i < GgufDequant.QK_K; i++) {
      buf.put((byte) 2);
    }
    for (int i = 0; i < GgufDequant.QK_K / 16; i++) {
      buf.putShort((short) 0);
    }
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, GgufDequant.TYPE_Q8_K, GgufDequant.QK_K);
    assertEquals(1f, out[0], 1e-5f);
    assertEquals(1f, out[255], 1e-5f);
  }

  @Test
  void dequantMxfp4ZeroMantissa() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_MXFP4).order(ByteOrder.LITTLE_ENDIAN);
    buf.put((byte) 127);
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0);
    }
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, GgufDequant.TYPE_MXFP4, GgufDequant.QK_MXFP4);
    assertEquals(0f, out[0], 0f);
    assertEquals(0f, out[16], 0f);
  }

  @Test
  void dequantQ2_0KnownBlock() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q2_0).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(2f));
    for (int i = 0; i < GgufDequant.QK2_0 / 4; i++) {
      buf.put((byte) 0b11_10_01_00);
    }
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, GgufDequant.TYPE_Q2_0, GgufDequant.QK2_0);
    assertEquals(-2f, out[0], 1e-4f);
    assertEquals(0f, out[1], 1e-4f);
    assertEquals(2f, out[2], 1e-4f);
    assertEquals(4f, out[3], 1e-4f);
  }

  @Test
  void ggmlWeightTypesHaveBlockMetadata() {
    int[] types = {
      GgufDequant.TYPE_Q4_1, GgufDequant.TYPE_Q5_0, GgufDequant.TYPE_Q5_1, GgufDequant.TYPE_Q8_1,
      GgufDequant.TYPE_Q2_K, GgufDequant.TYPE_Q5_K, GgufDequant.TYPE_Q8_K,
      GgufDequant.TYPE_IQ2_XXS, GgufDequant.TYPE_IQ2_XS, GgufDequant.TYPE_IQ2_S,
      GgufDequant.TYPE_IQ3_XXS, GgufDequant.TYPE_IQ3_S, GgufDequant.TYPE_IQ1_S,
      GgufDequant.TYPE_IQ1_M,
      GgufDequant.TYPE_IQ4_XS, GgufDequant.TYPE_TQ1_0, GgufDequant.TYPE_TQ2_0,
      GgufDequant.TYPE_MXFP4, GgufDequant.TYPE_NVFP4, GgufDequant.TYPE_Q1_0, GgufDequant.TYPE_Q2_0,
      GgufDequant.TYPE_I8, GgufDequant.TYPE_I16, GgufDequant.TYPE_I32, GgufDequant.TYPE_I64,
      GgufDequant.TYPE_F64
    };
    for (int type : types) {
      assertTrue(GgufDequant.typeBlockSize(type) > 0, "block size " + type);
      assertTrue(GgufDequant.typeBlockElems(type) > 0, "block elems " + type);
      new PackedWeight(
        new byte[(int) GgufDequant.packedByteLength(type, GgufDequant.typeBlockElems(type))],
        type,
        new int[] {1, GgufDequant.typeBlockElems(type)},
        GgufDequant.typeBlockElems(type));
    }
  }

  @Test
  void packedKernelBindsNewGgmlTypes() {
    int k = GgufDequant.QK_K;
    PackedWeight q5k = new PackedWeight(
      new byte[(int) GgufDequant.packedByteLength(GgufDequant.TYPE_Q5_K, k)],
      GgufDequant.TYPE_Q5_K, new int[] {1, k}, k);
    assertTrue(LinearKernel.of(q5k).name().contains("q5")
      || LinearKernel.of(q5k).name().contains("ggml"));
  }

  /**
   * Reference Q4_K kernel (pre in-place rewrite) for golden comparison.
   */
  private static float[] legacyDequantQ4_K(final ByteBuffer src, final int n) {
    float[] out = new float[n];
    byte[] scales = new byte[12];
    byte[] qs = new byte[GgufDequant.QK_K / 2];
    int y = 0;
    int nb = n / GgufDequant.QK_K;
    for (int i = 0; i < nb; i++) {
      float d = Float.float16ToFloat(src.getShort());
      float min = Float.float16ToFloat(src.getShort());
      src.get(scales);
      src.get(qs);
      int qOff = 0;
      int is = 0;
      for (int j = 0; j < GgufDequant.QK_K; j += 64) {
        int sc0 = legacyScaleMinK4(is, scales);
        int m0 = legacyScaleMinK4Min(is, scales);
        float d1 = d * sc0;
        float m1 = min * m0;
        int sc1 = legacyScaleMinK4(is + 1, scales);
        int m1b = legacyScaleMinK4Min(is + 1, scales);
        float d2 = d * sc1;
        float m2 = min * m1b;
        for (int l = 0; l < 32; l++) {
          out[y++] = d1 * (qs[qOff + l] & 0x0F) - m1;
        }
        for (int l = 0; l < 32; l++) {
          out[y++] = d2 * ((qs[qOff + l] & 0xFF) >> 4) - m2;
        }
        qOff += 32;
        is += 2;
      }
    }
    return out;
  }

  private static float[] legacyDequantQ6_K(final ByteBuffer src, final int n) {
    float[] out = new float[n];
    byte[] ql = new byte[GgufDequant.QK_K / 2];
    byte[] qh = new byte[GgufDequant.QK_K / 4];
    byte[] sc = new byte[GgufDequant.QK_K / 16];
    int y = 0;
    int nb = n / GgufDequant.QK_K;
    for (int i = 0; i < nb; i++) {
      src.get(ql);
      src.get(qh);
      src.get(sc);
      float d = Float.float16ToFloat(src.getShort());
      int qlOff = 0;
      int qhOff = 0;
      int scOff = 0;
      for (int block = 0; block < GgufDequant.QK_K; block += 128) {
        for (int l = 0; l < 32; l++) {
          int is = l / 16;
          int q1 = (ql[qlOff + l] & 0x0F) | ((qh[qhOff + l] & 3) << 4);
          int q2 = (ql[qlOff + l + 32] & 0x0F) | (((qh[qhOff + l] >> 2) & 3) << 4);
          int q3 = ((ql[qlOff + l] & 0xFF) >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4);
          int q4 = ((ql[qlOff + l + 32] & 0xFF) >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4);
          out[y + l] = d * sc[scOff + is] * (q1 - 32);
          out[y + l + 32] = d * sc[scOff + is + 2] * (q2 - 32);
          out[y + l + 64] = d * sc[scOff + is + 4] * (q3 - 32);
          out[y + l + 96] = d * sc[scOff + is + 6] * (q4 - 32);
        }
        y += 128;
        qlOff += 64;
        qhOff += 32;
        scOff += 8;
      }
    }
    return out;
  }

  private static int legacyScaleMinK4(final int j, final byte[] q) {
    if (j < 4) {
      return q[j] & 63;
    }
    return (q[j + 4] & 0x0F) | (((q[j - 4] & 0xFF) >> 6) << 4);
  }

  private static int legacyScaleMinK4Min(final int j, final byte[] q) {
    if (j < 4) {
      return q[j + 4] & 63;
    }
    return ((q[j + 4] & 0xFF) >> 4) | (((q[j] & 0xFF) >> 6) << 4);
  }

  private static void putF16(final byte[] packed, final int off, final float value) {
    ByteBuffer.wrap(packed, off, 2).order(ByteOrder.LITTLE_ENDIAN)
      .putShort(Float.floatToFloat16(value));
  }

  @Test
  void dequantRangeMatchesFullQ4_0() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q4_0).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.5f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x12);
    }
    buf.flip();
    byte[] packed = new byte[buf.remaining()];
    buf.get(packed);

    float[] full = GgufDequant.dequantize(
      ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN), 2, GgufDequant.QK4_0);
    float[] slice = new float[16];
    GgufDequant.dequantizeRange(packed, 2, GgufDequant.QK4_0, 8, 16, slice, 0);
    for (int i = 0; i < 16; i++) {
      assertEquals(full[8 + i], slice[i], 1e-6f);
    }
  }

  @Test
  void dequantRangeQ4_KFullAndMidBlockMatch() {
    this.assertBlockedRangeConsistent(GgufDequant.TYPE_Q4_K, GgufDequant.BLOCK_Q4_K, 3);
  }

  @Test
  void dequantRangeQ6_KFullAndMidBlockMatch() {
    this.assertBlockedRangeConsistent(GgufDequant.TYPE_Q6_K, GgufDequant.BLOCK_Q6_K, 3);
  }

  @Test
  void dequantRangeQ4_0DirectMatchesKnownBlock() {
    byte[] packed = new byte[GgufDequant.BLOCK_Q4_0];
    ByteBuffer buf = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.5f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x88);
    }
    float[] out = new float[GgufDequant.QK4_0];
    GgufDequant.dequantizeRange(
      packed, GgufDequant.TYPE_Q4_0, GgufDequant.QK4_0, 0, GgufDequant.QK4_0, out, 0);
    assertEquals(0f, out[0], 1e-5f);
    assertEquals(0f, out[16], 1e-5f);
  }

  private void assertBlockedRangeConsistent(
    final int ggmlType,
    final int blockBytes,
    final int blocks
  ) {
    int n = blocks * GgufDequant.QK_K;
    byte[] packed = new byte[blocks * blockBytes];
    for (int i = 0; i < packed.length; i++) {
      packed[i] = (byte) (i * 13 + 7);
    }
    // Valid f16 scales at block headers so values stay finite
    for (int b = 0; b < blocks; b++) {
      int off = b * blockBytes;
      if (ggmlType == GgufDequant.TYPE_Q4_K) {
        putF16(packed, off, 1f);
        putF16(packed, off + 2, 0.25f);
      } else if (ggmlType == GgufDequant.TYPE_Q6_K) {
        putF16(packed, off + blockBytes - 2, 1f);
      }
    }

    float[] full = new float[n];
    GgufDequant.dequantizeRange(packed, ggmlType, n, 0, n, full, 0);

    float[] left = new float[n / 2];
    float[] right = new float[n - n / 2];
    GgufDequant.dequantizeRange(packed, ggmlType, n, 0, left.length, left, 0);
    GgufDequant.dequantizeRange(packed, ggmlType, n, left.length, right.length, right, 0);
    for (int i = 0; i < left.length; i++) {
      assertEquals(full[i], left[i], 0f);
    }
    for (int i = 0; i < right.length; i++) {
      assertEquals(full[left.length + i], right[i], 0f);
    }

    float[] mid = new float[97];
    int midStart = 200;
    GgufDequant.dequantizeRange(packed, ggmlType, n, midStart, mid.length, mid, 0);
    for (int i = 0; i < mid.length; i++) {
      assertEquals(full[midStart + i], mid[i], 0f);
      assertTrue(Float.isFinite(mid[i]));
    }
  }

  @Test
  void dequantRangeQ4_KMatchesLegacyByteBufferKernel() {
    byte[] packed = new byte[GgufDequant.BLOCK_Q4_K];
    for (int i = 0; i < packed.length; i++) {
      packed[i] = (byte) (i * 17 + 3);
    }
    putF16(packed, 0, 0.75f);
    putF16(packed, 2, 0.125f);

    float[] legacy = legacyDequantQ4_K(ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN),
      GgufDequant.QK_K);
    float[] direct = new float[GgufDequant.QK_K];
    GgufDequant.dequantizeRange(
      packed, GgufDequant.TYPE_Q4_K, GgufDequant.QK_K, 0, GgufDequant.QK_K, direct, 0);
    assertArrayEquals(legacy, direct, 0f);
  }

  @Test
  void dequantRangeQ6_KMatchesLegacyByteBufferKernel() {
    byte[] packed = new byte[GgufDequant.BLOCK_Q6_K];
    for (int i = 0; i < packed.length; i++) {
      packed[i] = (byte) (i * 19 + 5);
    }
    putF16(packed, GgufDequant.BLOCK_Q6_K - 2, 0.5f);

    float[] legacy = legacyDequantQ6_K(ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN),
      GgufDequant.QK_K);
    float[] direct = new float[GgufDequant.QK_K];
    GgufDequant.dequantizeRange(
      packed, GgufDequant.TYPE_Q6_K, GgufDequant.QK_K, 0, GgufDequant.QK_K, direct, 0);
    assertArrayEquals(legacy, direct, 0f);
  }

  @Test
  void weightBagAsDenseMaterializesPackedEntries() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q4_0).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.25f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x00);
    }
    buf.flip();
    byte[] bytes = new byte[buf.remaining()];
    buf.get(bytes);

    PackedWeight packed = new PackedWeight(bytes, 2, new int[] {32, 1}, 32);
    WeightBag bag = new WeightBag(Map.of(
      "w.packed", packed,
      "w.dense", Tensor.of(new float[] {1f, 2f}, 2)));
    assertTrue(bag.hasPacked());
    assertTrue(bag.isPacked("w.packed"));
    assertFalse(bag.isPacked("w.dense"));

    WeightBag dense = bag.asDense();
    assertFalse(dense.hasPacked());
    assertFalse(dense.isPacked("w.packed"));
    assertArrayEquals(packed.materialize().toFloatArray(), dense.require("w.packed").toFloatArray(),
      1e-6f);
    assertArrayEquals(new float[] {1f, 2f}, dense.require("w.dense").toFloatArray(), 1e-6f);
    assertTrue(bag.hasPacked());
    assertFalse(packed.isReleased());
  }

  @Test
  void weightBagAsDenseReleasingPackedDropsPayload() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q4_0).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.25f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x00);
    }
    buf.flip();
    byte[] bytes = new byte[buf.remaining()];
    buf.get(bytes);

    PackedWeight packed = new PackedWeight(bytes, 2, new int[] {32, 1}, 32);
    WeightBag bag = new WeightBag(Map.of("w.packed", packed));
    WeightBag dense = bag.asDenseReleasingPacked();

    assertFalse(dense.hasPacked());
    assertTrue(packed.isReleased());
    assertEquals(0, packed.packedBytes());
    assertEquals(32, dense.require("w.packed").numel());
    org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, packed::materialize);
  }

  @Test
  void linearAndEmbeddingKernelsSelectSpecializedBackends() {
    Tensor dense = Tensor.of(new float[] {1f, 0f, 0f, 1f}, 2, 2);
    assertEquals("dense-f32", LinearKernel.of(dense).name());
    assertEquals("dense-f32-embed", EmbeddingKernel.of(dense).name());

    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q4_0).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(1f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x88);
    }
    buf.flip();
    byte[] bytes = new byte[buf.remaining()];
    buf.get(bytes);
    PackedWeight q4 = new PackedWeight(bytes, GgufDequant.TYPE_Q4_0, new int[] {1, 32}, 32);
    assertEquals("packed-q4_0", LinearKernel.of(q4).name());
    assertEquals("packed-q4_0-embed", EmbeddingKernel.of(q4).name());

    int k = GgufDequant.QK_K;
    PackedWeight q4k = new PackedWeight(
      new byte[(int) GgufDequant.packedByteLength(GgufDequant.TYPE_Q4_K, k)],
      GgufDequant.TYPE_Q4_K, new int[] {1, k}, k);
    assertEquals("packed-q4_k", LinearKernel.of(q4k).name());
    assertEquals("packed-q4_k-embed", EmbeddingKernel.of(q4k).name());

    PackedWeight q6k = new PackedWeight(
      new byte[(int) GgufDequant.packedByteLength(GgufDequant.TYPE_Q6_K, k)],
      GgufDequant.TYPE_Q6_K, new int[] {1, k}, k);
    assertEquals("packed-q6_k", LinearKernel.of(q6k).name());
    assertEquals("packed-q6_k-embed", EmbeddingKernel.of(q6k).name());

    LinearKernel denseKernel = LinearKernel.of(dense);
    float[] x = {3f, 4f};
    float[] y = new float[2];
    denseKernel.apply(x, 0, null, y, 0, 1, MatmulRuntime.sequential());
    assertEquals(3f, y[0], 1e-5f);
    assertEquals(4f, y[1], 1e-5f);

    float[] ids = {0f};
    float[] embedOut = new float[2];
    EmbeddingKernel.of(dense).gather(ids, 0, 1, embedOut, 0);
    assertEquals(1f, embedOut[0], 1e-5f);
    assertEquals(0f, embedOut[1], 1e-5f);
  }

  @Test
  void shortConvKernelIdentityish() {
    int hidden = 2;
    int kernel = 3;
    float[] inProj = new float[3 * hidden * hidden];
    for (int o = 0; o < 3 * hidden; o++) {
      int in = o % hidden;
      inProj[o * hidden + in] = 1f;
    }
    float[] conv = new float[hidden * kernel];
    for (int h = 0; h < hidden; h++) {
      conv[h * kernel + (kernel - 1)] = 1f;
    }
    float[] outProj = new float[hidden * hidden];
    for (int i = 0; i < hidden; i++) {
      outProj[i * hidden + i] = 1f;
    }

    ShortConv layer = new ShortConv(
      Tensor.of(inProj, 3 * hidden, hidden),
      Tensor.of(conv, hidden, kernel),
      Tensor.of(outProj, hidden, hidden),
      0);

    Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, hidden);
    Tensor out = layer.forward(input, new Context());
    assertEquals(2, out.size(0));
    assertEquals(hidden, out.size(1));
    assertTrue(Float.isFinite(out.data()[0]));
    assertArrayEquals(new int[] {2, hidden}, out.shape());
  }

  @Test
  void lfm2GgufTokenizerUsesChatMlFormat() throws Exception {
    Path path = OptionalModelAssumptions.requireLfm2Gguf();

    try (GgufReader reader = GgufReader.open(path)) {
      Tokenizer tok = Tokenizer.fromGguf(reader);
      assertFalse(tok.isTurnBasedChat());
      assertEquals(Tokenizer.ChatFormat.CHATML, tok.chatFormat());
      assertEquals("", ChatPrompts.systemFor(tok));

      boolean enableThinking = tok.invitesThinking();
      String chat = tok.applyChatTemplate(
        List.of(
          Map.of("role", "system", "content", "be brief"),
          Map.of("role", "user", "content", "hello")),
        true,
        enableThinking);
      assertTrue(chat.contains("<|im_start|>user"));
      assertTrue(chat.contains("<|im_start|>assistant"));
      assertFalse(chat.startsWith("user:"), chat);
      if (!enableThinking) {
        assertFalse(chat.contains("<think>"), chat);
      }
    }
  }

  private static PackedWeight float32Matrix() {
    ByteBuffer buf = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    buf.putFloat(1f);
    buf.putFloat(0f);
    buf.putFloat(0f);
    buf.putFloat(1f);
    return new PackedWeight(buf.array(), GgufDequant.TYPE_F32, new int[] {2, 2}, 4);
  }

  @Test
  void ggufReaderConstructorFailureReleasesFile() throws Exception {
    Path file = Files.createTempFile("nanollvm-bad-gguf", ".gguf");
    try {
      Files.write(file, new byte[] {0, 1, 2, 3, 4, 5, 6, 7});
      IOException thrown = assertThrows(IOException.class, () -> GgufReader.open(file));
      assertTrue(thrown.getMessage().contains("bad magic"));
      Files.delete(file);
      assertFalse(Files.exists(file));
    } finally {
      Files.deleteIfExists(file);
    }
  }

  @Test
  void rotaryTablesArePerInstance() {
    RotaryEmbedding.Tables first = new RotaryEmbedding.Tables();
    RotaryEmbedding.Tables second = new RotaryEmbedding.Tables();
    RotaryEmbedding shared = first.get(4, 4, 8, 10_000f);
    assertSame(shared, first.get(4, 4, 8, 10_000f));
    assertNotSame(shared, second.get(4, 4, 8, 10_000f));
  }

  @Test
  void float32PackedLayersDropPackedCopy() {
    PackedWeight linearWeight = float32Matrix();
    Linear linear = new Linear(linearWeight);
    assertFalse(linear.isPacked());
    assertTrue(linearWeight.isReleased());

    PackedWeight embedWeight = float32Matrix();
    VocabParallelEmbedding embed = new VocabParallelEmbedding(embedWeight);
    assertFalse(embed.isPacked());
    assertTrue(embedWeight.isReleased());
  }
}
