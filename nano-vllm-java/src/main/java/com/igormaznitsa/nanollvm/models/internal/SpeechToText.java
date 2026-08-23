package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

/**
 * Speech-to-text graph: 16 kHz (or resampled) mono PCM in, transcript out.
 *
 * @since 1.3.0
 */
public interface SpeechToText {

  String architectureName();

  String transcribe(float[] pcm, int sampleRate, String language, Tokenizer tokenizer);
}
