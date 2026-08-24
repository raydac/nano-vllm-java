package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.Locale;

/**
 * Speech-to-text graph: 16 kHz (or resampled) mono PCM in, transcript out.
 *
 * @since 1.3.0
 */
public interface SpeechToText {

  /**
   * Architecture id stored on the loaded {@link com.igormaznitsa.nanollvm.models.LlmModel}
   * ({@code whisper}).
   *
   * @return non-blank family key
   */
  String architectureName();

  /**
   * Greedy decode of mono PCM. Rates other than 16 kHz are resampled; clips longer than 30 s
   * are chunked. {@code language} is a hint ({@link Locale#getLanguage()} → {@code <|xx|>});
   * {@code null} or {@link Locale#ROOT} selects automatically. Region is ignored.
   *
   * @param pcm        samples in {@code [-1, 1]}; must not be {@code null}
   * @param sampleRate Hertz of {@code pcm}; must be {@code >= 1}
   * @param language   hint, or {@code null}/{@link Locale#ROOT} for auto
   * @param tokenizer  Whisper vocabulary; must not be {@code null}
   * @param runtime    dense kernel runtime; must not be {@code null}
   * @return transcript text; empty when the clip is silent; never {@code null}
   */
  String transcribe(
    float[] pcm,
    int sampleRate,
    Locale language,
    Tokenizer tokenizer,
    MatmulRuntime runtime);
}
