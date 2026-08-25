package com.igormaznitsa.nanollvm.models.internal.audio;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * OpenAI Whisper log-mel features: 16 kHz, n_fft=400, hop=160, 80 Slaney mels, 30 s pad/trim.
 *
 * @since 1.3.0
 */
public final class WhisperLogMel {

  /**
   * Encoder sample rate (Hz).
   */
  public static final int SAMPLE_RATE = 16_000;
  /** STFT window length in samples. */
  public static final int N_FFT = 400;
  /** STFT hop in samples. */
  public static final int HOP_LENGTH = 160;
  /** Slaney mel bands (Whisper-base). */
  public static final int N_MELS = 80;
  /** 30 s of 16 kHz mono. */
  public static final int CHUNK_SAMPLES = 480_000;
  /** Mel frames for one 30 s chunk. */
  public static final int N_FRAMES = 3_000;

  private static final int FREQ_BINS = N_FFT / 2;
  private static final float[] HANN = hann();
  private static final float[][] MEL_FILTERS = slaneyMelFilters();

  private WhisperLogMel() {
  }

  /**
   * Resamples mono PCM to 16 kHz when needed, then computes Whisper log-mel {@code [80, 3000]}.
   * Audio longer than 30 s is trimmed; shorter audio is zero-padded.
   *
   * @param pcm        mono samples
   * @param sampleRate Hertz of {@code pcm}
   * @return log-mel spectrogram {@code [numMelBins, 3000]}
   */
  public static Tensor features(final float[] pcm, final int sampleRate) {
    requireNonNull(pcm, "pcm");
    if (sampleRate < 1) {
      throw new IllegalArgumentException("sampleRate must be >= 1");
    }
    float[] at16k = sampleRate == SAMPLE_RATE ? pcm : resampleLinear(pcm, sampleRate, SAMPLE_RATE);
    float[] chunk = padOrTrim(at16k, CHUNK_SAMPLES);
    float[] spec = stftPower(chunk);
    float[] mel = new float[N_MELS * N_FRAMES];
    for (int frame = 0; frame < N_FRAMES; frame++) {
      int specOff = frame * FREQ_BINS;
      for (int m = 0; m < N_MELS; m++) {
        float sum = 0f;
        float[] filter = MEL_FILTERS[m];
        for (int f = 0; f < FREQ_BINS; f++) {
          sum += filter[f] * spec[specOff + f];
        }
        mel[m * N_FRAMES + frame] = sum;
      }
    }
    float max = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < mel.length; i++) {
      float log = (float) Math.log10(Math.max(mel[i], 1e-10f));
      mel[i] = log;
      if (log > max) {
        max = log;
      }
    }
    float floor = max - 8f;
    for (int i = 0; i < mel.length; i++) {
      mel[i] = (Math.max(mel[i], floor) + 4f) / 4f;
    }
    return Tensor.of(mel, N_MELS, N_FRAMES);
  }

  /**
   * Linear-interpolates mono PCM from {@code fromRate} to {@code toRate}.
   *
   * @param pcm      source samples
   * @param fromRate source Hertz
   * @param toRate   target Hertz
   * @return resampled samples
   */
  public static float[] resampleLinear(final float[] pcm, final int fromRate, final int toRate) {
    if (pcm.length == 0) {
      return pcm;
    }
    double ratio = (double) toRate / (double) fromRate;
    int outLen = Math.max(1, (int) Math.round(pcm.length * ratio));
    float[] out = new float[outLen];
    double step = (double) fromRate / (double) toRate;
    for (int i = 0; i < outLen; i++) {
      double src = i * step;
      int i0 = (int) Math.floor(src);
      int i1 = Math.min(i0 + 1, pcm.length - 1);
      float t = (float) (src - i0);
      out[i] = pcm[Math.min(i0, pcm.length - 1)] * (1f - t) + pcm[i1] * t;
    }
    return out;
  }

  static float[] padOrTrim(final float[] pcm, final int length) {
    if (pcm.length == length) {
      return pcm;
    }
    float[] out = new float[length];
    System.arraycopy(pcm, 0, out, 0, Math.min(pcm.length, length));
    return out;
  }

  private static float[] stftPower(final float[] pcm) {
    int pad = N_FFT / 2;
    float[] padded = new float[pcm.length + 2 * pad];
    System.arraycopy(pcm, 0, padded, pad, pcm.length);
    for (int i = 0; i < pad; i++) {
      padded[pad - 1 - i] = pcm[Math.min(i, pcm.length - 1)];
      padded[pad + pcm.length + i] = pcm[Math.max(0, pcm.length - 1 - i)];
    }
    float[] spec = new float[N_FRAMES * FREQ_BINS];
    float[] re = new float[FREQ_BINS];
    float[] im = new float[FREQ_BINS];
    for (int frame = 0; frame < N_FRAMES; frame++) {
      int origin = frame * HOP_LENGTH;
      dftReal(padded, origin, re, im);
      int off = frame * FREQ_BINS;
      for (int f = 0; f < FREQ_BINS; f++) {
        final float r = re[f];
        final float i = im[f];
        spec[off + f] = r * r + i * i;
      }
    }
    return spec;
  }

  private static void dftReal(
    final float[] pcm,
    final int origin,
    final float[] re,
    final float[] im
  ) {
    int n = N_FFT;
    for (int k = 0; k < FREQ_BINS; k++) {
      double sumRe = 0.0;
      double sumIm = 0.0;
      for (int t = 0; t < n; t++) {
        float sample = pcm[origin + t] * HANN[t];
        double angle = 2.0 * Math.PI * k * t / n;
        sumRe += sample * Math.cos(angle);
        sumIm -= sample * Math.sin(angle);
      }
      re[k] = (float) sumRe;
      im[k] = (float) sumIm;
    }
  }

  private static float[] hann() {
    float[] w = new float[N_FFT];
    for (int i = 0; i < N_FFT; i++) {
      w[i] = 0.5f - 0.5f * (float) Math.cos(2.0 * Math.PI * i / (N_FFT - 1));
    }
    return w;
  }

  private static float[][] slaneyMelFilters() {
    float fMin = 0f;
    float fMax = SAMPLE_RATE / 2f;
    final int nMels = N_MELS;
    double minMel = hzToMel(fMin);
    double maxMel = hzToMel(fMax);
    double[] melPoints = new double[nMels + 2];
    for (int i = 0; i < melPoints.length; i++) {
      melPoints[i] = minMel + (maxMel - minMel) * i / (nMels + 1);
    }
    double[] hzPoints = new double[melPoints.length];
    for (int i = 0; i < melPoints.length; i++) {
      hzPoints[i] = melToHz(melPoints[i]);
    }
    double[] bins = new double[hzPoints.length];
    for (int i = 0; i < hzPoints.length; i++) {
      bins[i] = hzPoints[i] * (N_FFT / 2.0) / (SAMPLE_RATE / 2.0);
    }
    float[][] filters = new float[nMels][FREQ_BINS];
    for (int m = 0; m < nMels; m++) {
      double left = bins[m];
      double center = bins[m + 1];
      double right = bins[m + 2];
      for (int f = 0; f < FREQ_BINS; f++) {
        double weight = 0.0;
        if (f >= left && f <= center && center > left) {
          weight = (f - left) / (center - left);
        } else if (f >= center && f <= right && right > center) {
          weight = (right - f) / (right - center);
        }
        filters[m][f] = (float) weight;
      }
      double enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);
      for (int f = 0; f < FREQ_BINS; f++) {
        filters[m][f] *= (float) enorm;
      }
    }
    return filters;
  }

  private static double hzToMel(final double hz) {
    double fSp = 200.0 / 3.0;
    double minLogHz = 1000.0;
    double minLogMel = minLogHz / fSp;
    double logStep = Math.log(6.4) / 27.0;
    if (hz < minLogHz) {
      return hz / fSp;
    }
    return minLogMel + Math.log(hz / minLogHz) / logStep;
  }

  private static double melToHz(final double mel) {
    double fSp = 200.0 / 3.0;
    double minLogHz = 1000.0;
    double minLogMel = minLogHz / fSp;
    double logStep = Math.log(6.4) / 27.0;
    if (mel < minLogMel) {
      return fSp * mel;
    }
    return minLogHz * Math.exp(logStep * (mel - minLogMel));
  }
}
