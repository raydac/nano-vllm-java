package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class GemmaQatTest {

  @Test
  void unpacksInt4LowNibbleFirst() {
    byte packed = (byte) (13 << 4 | 3);
    float[] dst = new float[2];
    GemmaQat.unpackInt4(new byte[] {packed}, 0, 1, dst, 2);
    assertEquals(3 - 8, dst[0], 0f);
    assertEquals(13 - 8, dst[1], 0f);
  }

  @Test
  void unpacksInt2CrumbsInBitOrder() {
    byte packed = (byte) 0b11_10_01_00;
    float[] dst = new float[4];
    GemmaQat.unpackInt2(new byte[] {packed}, 0, 1, dst, 4);
    assertArrayEquals(new float[] {0 - 2, 1 - 2, 2 - 2, 3 - 2}, dst);
  }

  @Test
  void dequantizesRowWithPerBlockScale() {
    byte[] packed = new byte[] {(byte) 0x80};
    float[] scales = new float[] {0.5f, 2.0f};
    GemmaQatWeight weight = new GemmaQatWeight(
      "t", packed, scales, 1, 2, 4, 2, 0f, 0f);
    float[] dst = new float[2];
    weight.dequantizeRow(0, dst);
    assertEquals((0 - 8) * 0.5f, dst[0], 1e-6f);
    assertEquals((8 - 8) * 2.0f, dst[1], 1e-6f);
  }
}
