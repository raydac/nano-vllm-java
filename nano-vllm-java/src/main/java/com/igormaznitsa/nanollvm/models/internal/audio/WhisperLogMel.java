package com.igormaznitsa.nanollvm.models.internal.audio;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;

/**
 * OpenAI Whisper log-mel features: 16 kHz, n_fft=400, hop=160, 80 Slaney mels, 30 s pad/trim.
 *
 * <p>STFT uses Bluestein FFT (n_fft is not a power of two). Mel filters are stored sparsely.
 * Frame loops may run on a {@link MatmulRuntime} pool when one is supplied.
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
  private static final SparseMel[] MEL_FILTERS = slaneyMelFilters();
  private static final RealDft DFT = new RealDft(N_FFT);

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
    return features(pcm, sampleRate, MatmulRuntime.sequential());
  }

  /**
   * Same as {@link #features(float[], int)} but may parallelize STFT frames on {@code runtime}.
   *
   * @param pcm        mono samples
   * @param sampleRate Hertz of {@code pcm}
   * @param runtime    matmul / range pool; must not be {@code null}
   * @return log-mel spectrogram {@code [numMelBins, 3000]}
   * @since 1.3.0
   */
  public static Tensor features(
    final float[] pcm,
    final int sampleRate,
    final MatmulRuntime runtime
  ) {
    requireNonNull(pcm, "pcm");
    requireNonNull(runtime, "runtime");
    if (sampleRate < 1) {
      throw new IllegalArgumentException("sampleRate must be >= 1");
    }
    float[] at16k = sampleRate == SAMPLE_RATE ? pcm : resampleLinear(pcm, sampleRate, SAMPLE_RATE);
    return featuresAt16k(at16k, 0, at16k.length, runtime);
  }

  /**
   * Log-mel for a slice of already-16 kHz PCM without copying the slice first.
   *
   * @param pcm16k  mono 16 kHz samples
   * @param origin  start index
   * @param length  slice length ({@code >= 0}); trimmed/padded to 30 s
   * @param runtime matmul / range pool; must not be {@code null}
   * @return log-mel spectrogram {@code [numMelBins, 3000]}
   * @since 1.3.0
   */
  public static Tensor featuresAt16k(
    final float[] pcm16k,
    final int origin,
    final int length,
    final MatmulRuntime runtime
  ) {
    requireNonNull(pcm16k, "pcm16k");
    requireNonNull(runtime, "runtime");
    if (origin < 0 || length < 0 || (long) origin + length > pcm16k.length) {
      throw new IllegalArgumentException("pcm slice out of range");
    }
    float[] chunk = padOrTrim(pcm16k, origin, length, CHUNK_SAMPLES);
    float[] spec = stftPower(chunk, runtime);
    float[] mel = applyMel(spec);
    normalizeLogMel(mel);
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
    return padOrTrim(pcm, 0, pcm.length, length);
  }

  static float[] padOrTrim(
    final float[] pcm,
    final int origin,
    final int length,
    final int target
  ) {
    if (origin == 0 && length == target && pcm.length == target) {
      return pcm;
    }
    float[] out = new float[target];
    System.arraycopy(pcm, origin, out, 0, Math.min(length, target));
    return out;
  }

  private static float[] stftPower(final float[] pcm, final MatmulRuntime runtime) {
    int pad = N_FFT / 2;
    float[] padded = new float[pcm.length + 2 * pad];
    System.arraycopy(pcm, 0, padded, pad, pcm.length);
    for (int i = 0; i < pad; i++) {
      padded[pad - 1 - i] = pcm[Math.min(i, pcm.length - 1)];
      padded[pad + pcm.length + i] = pcm[Math.max(0, pcm.length - 1 - i)];
    }
    float[] spec = new float[N_FRAMES * FREQ_BINS];
    int fftSize = DFT.fftSize();
    runtime.parallelRanges(N_FRAMES, (start, end) -> {
      float[] scratchRe = new float[fftSize];
      float[] scratchIm = new float[fftSize];
      float[] aRe = new float[fftSize];
      float[] aIm = new float[fftSize];
      for (int frame = start; frame < end; frame++) {
        DFT.powerSpectrum(
          padded,
          frame * HOP_LENGTH,
          HANN,
          spec,
          frame * FREQ_BINS,
          scratchRe,
          scratchIm,
          aRe,
          aIm);
      }
    });
    return spec;
  }

  private static float[] applyMel(final float[] spec) {
    float[] mel = new float[N_MELS * N_FRAMES];
    for (int frame = 0; frame < N_FRAMES; frame++) {
      int specOff = frame * FREQ_BINS;
      for (int m = 0; m < N_MELS; m++) {
        SparseMel filter = MEL_FILTERS[m];
        float sum = 0f;
        float[] weights = filter.weights;
        int start = filter.start;
        for (int i = 0; i < weights.length; i++) {
          sum += weights[i] * spec[specOff + start + i];
        }
        mel[m * N_FRAMES + frame] = sum;
      }
    }
    return mel;
  }

  private static void normalizeLogMel(final float[] mel) {
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
  }

  private static float[] hann() {
    float[] w = new float[N_FFT];
    for (int i = 0; i < N_FFT; i++) {
      w[i] = 0.5f - 0.5f * (float) Math.cos(2.0 * Math.PI * i / (N_FFT - 1));
    }
    return w;
  }

  private static SparseMel[] slaneyMelFilters() {
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
    SparseMel[] filters = new SparseMel[nMels];
    for (int m = 0; m < nMels; m++) {
      double left = bins[m];
      double center = bins[m + 1];
      double right = bins[m + 2];
      double enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);
      int start = Math.max(0, (int) Math.floor(left));
      int end = Math.min(FREQ_BINS - 1, (int) Math.ceil(right));
      float[] dense = new float[Math.max(0, end - start + 1)];
      for (int f = start; f <= end; f++) {
        double weight = 0.0;
        if (f >= left && f <= center && center > left) {
          weight = (f - left) / (center - left);
        } else if (f >= center && f <= right && right > center) {
          weight = (right - f) / (right - center);
        }
        dense[f - start] = (float) (weight * enorm);
      }
      int first = 0;
      while (first < dense.length && dense[first] == 0f) {
        first++;
      }
      int last = dense.length - 1;
      while (last >= first && dense[last] == 0f) {
        last--;
      }
      if (last < first) {
        filters[m] = new SparseMel(0, new float[0]);
      } else {
        float[] trimmed = new float[last - first + 1];
        System.arraycopy(dense, first, trimmed, 0, trimmed.length);
        filters[m] = new SparseMel(start + first, trimmed);
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

  @SuppressWarnings("ArrayRecordComponent")
  private record SparseMel(int start, float[] weights) {
  }
}
