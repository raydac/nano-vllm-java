package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

final class VitsSplineTest {

  @Test
  void identitySplineLeavesInteriorSamplesUnchanged() {
    int bins = 10;
    float[] widths = new float[bins];
    float[] heights = new float[bins];
    float[] derivatives = new float[bins - 1];
    float unitDerivative = (float) Math.log(Math.expm1(1.0 - 1e-3));
    for (int i = 0; i < derivatives.length; i++) {
      derivatives[i] = unitDerivative;
    }
    float[] samples = {-4.5f, -1f, 0f, 0.7f, 4.9f};
    for (float sample : samples) {
      assertEquals(
        sample,
        VitsSynthesizer.inverseLinearTailRationalQuadratic(
          sample, widths, heights, derivatives, 5f),
        1e-4f);
    }
    assertEquals(
      6f,
      VitsSynthesizer.inverseLinearTailRationalQuadratic(6f, widths, heights, derivatives, 5f));
  }
}
