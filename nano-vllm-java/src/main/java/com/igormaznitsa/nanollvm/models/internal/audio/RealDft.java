package com.igormaznitsa.nanollvm.models.internal.audio;

/**
 * Real-input DFT via Bluestein's chirp-z transform (arbitrary length, O(N log N)).
 *
 * <p>Used by Whisper STFT where {@code n_fft = 400} is not a power of two.
 *
 * @since 1.3.0
 */
final class RealDft {

  private final int n;
  private final int half;
  private final int fftSize;
  private final float[] chirpRe;
  private final float[] chirpIm;
  private final float[] kernelRe;
  private final float[] kernelIm;
  private final float[] twiddleRe;
  private final float[] twiddleIm;

  RealDft(final int n) {
    if (n < 2) {
      throw new IllegalArgumentException("n must be >= 2");
    }
    this.n = n;
    this.half = n / 2;
    this.fftSize = nextPow2(2 * n - 1);
    this.chirpRe = new float[n];
    this.chirpIm = new float[n];
    for (int i = 0; i < n; i++) {
      double angle = Math.PI * i * i / n;
      this.chirpRe[i] = (float) Math.cos(angle);
      this.chirpIm[i] = (float) -Math.sin(angle);
    }
    float[] kRe = new float[this.fftSize];
    float[] kIm = new float[this.fftSize];
    for (int i = 0; i < n; i++) {
      double angle = Math.PI * i * i / n;
      float re = (float) Math.cos(angle);
      float im = (float) Math.sin(angle);
      kRe[i] = re;
      kIm[i] = im;
      if (i > 0) {
        kRe[this.fftSize - i] = re;
        kIm[this.fftSize - i] = im;
      }
    }
    this.twiddleRe = new float[this.fftSize / 2];
    this.twiddleIm = new float[this.fftSize / 2];
    for (int i = 0; i < this.twiddleRe.length; i++) {
      double angle = -2.0 * Math.PI * i / this.fftSize;
      this.twiddleRe[i] = (float) Math.cos(angle);
      this.twiddleIm[i] = (float) Math.sin(angle);
    }
    this.fftInPlace(kRe, kIm);
    this.kernelRe = kRe;
    this.kernelIm = kIm;
  }

  private static int nextPow2(final int value) {
    int n = 1;
    while (n < value) {
      n <<= 1;
    }
    return n;
  }

  int fftSize() {
    return this.fftSize;
  }

  /**
   * Writes power spectrum bins {@code 0 .. n/2 - 1} (Nyquist dropped, Whisper-style).
   *
   * @param pcm       windowed real samples; length at least {@code origin + n}
   * @param origin    start of the window in {@code pcm}
   * @param window    length-{@code n} Hann (or other) weights
   * @param power     destination length {@code >= n/2}
   * @param powerOff  start index in {@code power}
   * @param scratchRe length {@code >= fftSize} reusable buffer
   * @param scratchIm length {@code >= fftSize} reusable buffer
   * @param aRe       length {@code >= fftSize} reusable buffer
   * @param aIm       length {@code >= fftSize} reusable buffer
   */
  void powerSpectrum(
    final float[] pcm,
    final int origin,
    final float[] window,
    final float[] power,
    final int powerOff,
    final float[] scratchRe,
    final float[] scratchIm,
    final float[] aRe,
    final float[] aIm
  ) {
    int m = this.fftSize;
    for (int i = 0; i < m; i++) {
      aRe[i] = 0f;
      aIm[i] = 0f;
    }
    for (int i = 0; i < this.n; i++) {
      float sample = pcm[origin + i] * window[i];
      aRe[i] = sample * this.chirpRe[i];
      aIm[i] = sample * this.chirpIm[i];
    }
    this.fftInPlace(aRe, aIm);
    for (int i = 0; i < m; i++) {
      float ar = aRe[i];
      float ai = aIm[i];
      float kr = this.kernelRe[i];
      float ki = this.kernelIm[i];
      scratchRe[i] = ar * kr - ai * ki;
      scratchIm[i] = ar * ki + ai * kr;
    }
    this.ifftInPlace(scratchRe, scratchIm);
    for (int k = 0; k < this.half; k++) {
      float re = scratchRe[k] * this.chirpRe[k] - scratchIm[k] * this.chirpIm[k];
      float im = scratchRe[k] * this.chirpIm[k] + scratchIm[k] * this.chirpRe[k];
      power[powerOff + k] = re * re + im * im;
    }
  }

  private void fftInPlace(final float[] re, final float[] im) {
    int m = this.fftSize;
    for (int i = 1, j = 0; i < m; i++) {
      int bit = m >>> 1;
      for (; (j & bit) != 0; bit >>>= 1) {
        j ^= bit;
      }
      j ^= bit;
      if (i < j) {
        float tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        float ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    for (int len = 2; len <= m; len <<= 1) {
      int halfLen = len >> 1;
      int step = m / len;
      for (int i = 0; i < m; i += len) {
        for (int j = 0; j < halfLen; j++) {
          int tw = j * step;
          float wr = this.twiddleRe[tw];
          float wi = this.twiddleIm[tw];
          int u = i + j;
          int v = u + halfLen;
          float tr = wr * re[v] - wi * im[v];
          float ti = wr * im[v] + wi * re[v];
          re[v] = re[u] - tr;
          im[v] = im[u] - ti;
          re[u] += tr;
          im[u] += ti;
        }
      }
    }
  }

  private void ifftInPlace(final float[] re, final float[] im) {
    for (int i = 0; i < this.fftSize; i++) {
      im[i] = -im[i];
    }
    this.fftInPlace(re, im);
    float inv = 1f / this.fftSize;
    for (int i = 0; i < this.fftSize; i++) {
      re[i] *= inv;
      im[i] = -im[i] * inv;
    }
  }
}
