package com.igormaznitsa.nanollvm.models.internal.audio;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

final class RealDftTest {

  private static float[] naivePower(final float[] pcm, final float[] window) {
    int n = window.length;
    int half = n / 2;
    float[] power = new float[half];
    for (int k = 0; k < half; k++) {
      double sumRe = 0.0;
      double sumIm = 0.0;
      for (int t = 0; t < n; t++) {
        float sample = pcm[t] * window[t];
        double angle = 2.0 * Math.PI * k * t / n;
        sumRe += sample * Math.cos(angle);
        sumIm -= sample * Math.sin(angle);
      }
      power[k] = (float) (sumRe * sumRe + sumIm * sumIm);
    }
    return power;
  }

  @Test
  void powerSpectrumMatchesNaiveDftForHannWindow() {
    int n = WhisperLogMel.N_FFT;
    float[] window = new float[n];
    for (int i = 0; i < n; i++) {
      window[i] = 0.5f - 0.5f * (float) Math.cos(2.0 * Math.PI * i / (n - 1));
    }
    float[] pcm = new float[n];
    for (int i = 0; i < n; i++) {
      pcm[i] = (float) Math.sin(2.0 * Math.PI * 7 * i / n);
    }

    RealDft dft = new RealDft(n);
    float[] fast = new float[n / 2];
    float[] scratchRe = new float[dft.fftSize()];
    float[] scratchIm = new float[dft.fftSize()];
    float[] aRe = new float[dft.fftSize()];
    float[] aIm = new float[dft.fftSize()];
    dft.powerSpectrum(pcm, 0, window, fast, 0, scratchRe, scratchIm, aRe, aIm);

    float[] naive = naivePower(pcm, window);
    assertEquals(naive.length, fast.length);
    assertArrayEquals(naive, fast, 1e-2f);
  }
}
