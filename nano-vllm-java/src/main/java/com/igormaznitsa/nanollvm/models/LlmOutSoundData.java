package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

/**
 * Synthesized audio: uncompressed RIFF/WAVE (PCM16 little-endian, mono) plus sample rate.
 *
 * @param wav        RIFF/WAVE file bytes; never {@code null}
 * @param sampleRate Hertz of the PCM frames inside {@code wav}; {@code >= 1}
 * @since 1.3.0
 */
@SuppressWarnings("ArrayRecordComponent")
public record LlmOutSoundData(byte[] wav, int sampleRate) implements LlmOutput {

  /**
   * @throws NullPointerException     if {@code wav} is {@code null}
   * @throws IllegalArgumentException if {@code sampleRate < 1}
   */
  public LlmOutSoundData {
    requireNonNull(wav, "wav");
    if (sampleRate < 1) {
      throw new IllegalArgumentException("sampleRate must be >= 1");
    }
    wav = wav.clone();
  }

  @Override
  public byte[] wav() {
    return this.wav.clone();
  }

  @Override
  public LlmModality modality() {
    return LlmModality.AUDIO;
  }
}
