package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.util.Locale;

/**
 * Audio payload for speech-to-text via {@link LlmModel#generate(LlmInput, LlmModality)}.
 *
 * <p>Exactly one of {@code wav} or {@code pcm} is set. {@code sampleRate} applies only to PCM.
 * {@code language} is an optional Whisper hint ({@code null} / {@link Locale#ROOT} = auto).
 *
 * @param wav        RIFF/WAVE bytes, or {@code null} when using PCM
 * @param pcm        mono samples in {@code [-1, 1]}, or {@code null} when using WAV
 * @param sampleRate Hertz of {@code pcm}; {@code 0} when {@code wav} is set
 * @param language   Whisper language hint, or {@code null} for auto
 * @since 1.3.0
 */
@SuppressWarnings("ArrayRecordComponent")
public record LlmInSound(byte[] wav, float[] pcm, int sampleRate, Locale language)
  implements LlmInput {

  /**
   * @throws IllegalArgumentException if both {@code wav} and {@code pcm} are null, both are set,
   *                                  or PCM is used with {@code sampleRate < 1}
   */
  public LlmInSound {
    boolean hasWav = wav != null;
    boolean hasPcm = pcm != null;
    if (hasWav == hasPcm) {
      throw new IllegalArgumentException("exactly one of wav or pcm must be set");
    }
    if (hasWav) {
      wav = wav.clone();
      sampleRate = 0;
    } else {
      requireNonNull(pcm, "pcm");
      if (sampleRate < 1) {
        throw new IllegalArgumentException("sampleRate must be >= 1");
      }
      pcm = pcm.clone();
    }
  }

  /**
   * Uncompressed WAV bytes (PCM or IEEE float, mixed to mono by Whisper).
   *
   * @param wav RIFF/WAVE payload; must not be {@code null}
   * @return sound input with auto language
   */
  public static LlmInSound ofWav(final byte[] wav) {
    return ofWav(wav, null);
  }

  /**
   * Uncompressed WAV bytes with an optional Whisper language hint.
   *
   * @param wav      RIFF/WAVE payload; must not be {@code null}
   * @param language hint, or {@code null}/{@link Locale#ROOT} for auto
   * @return sound input
   */
  public static LlmInSound ofWav(final byte[] wav, final Locale language) {
    return new LlmInSound(requireNonNull(wav, "wav"), null, 0, language);
  }

  /**
   * Mono PCM in {@code [-1, 1]}.
   *
   * @param pcm        samples; must not be {@code null}
   * @param sampleRate Hertz; must be {@code >= 1}
   * @return sound input with auto language
   */
  public static LlmInSound ofPcm(final float[] pcm, final int sampleRate) {
    return ofPcm(pcm, sampleRate, null);
  }

  /**
   * Mono PCM with an optional Whisper language hint.
   *
   * @param pcm        samples; must not be {@code null}
   * @param sampleRate Hertz; must be {@code >= 1}
   * @param language   hint, or {@code null}/{@link Locale#ROOT} for auto
   * @return sound input
   */
  public static LlmInSound ofPcm(final float[] pcm, final int sampleRate, final Locale language) {
    return new LlmInSound(null, requireNonNull(pcm, "pcm"), sampleRate, language);
  }

  /**
   * {@code true} when this input carries RIFF/WAVE bytes.
   */
  public boolean isWav() {
    return this.wav != null;
  }

  /**
   * {@code true} when this input carries mono PCM.
   */
  public boolean isPcm() {
    return this.pcm != null;
  }

  @Override
  public byte[] wav() {
    return this.wav == null ? null : this.wav.clone();
  }

  @Override
  public float[] pcm() {
    return this.pcm == null ? null : this.pcm.clone();
  }
}
