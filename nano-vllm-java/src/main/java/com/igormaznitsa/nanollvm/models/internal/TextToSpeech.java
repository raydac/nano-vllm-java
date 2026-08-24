package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import java.nio.file.Path;

/**
 * Text-to-speech graph: text in, mono PCM out at the voice sample rate.
 *
 * @since 1.3.0
 */
public interface TextToSpeech {

  /**
   * Architecture id stored on the loaded {@link com.igormaznitsa.nanollvm.models.LlmModel}
   * ({@code piper}).
   *
   * @return non-blank family key
   */
  String architectureName();

  /**
   * Waveform Hertz of {@link #synthesize} output (from the Piper sidecar).
   *
   * @return sample rate {@code >= 1}
   */
  int sampleRate();

  /**
   * G2P then VITS infer. Returns float samples in {@code [-1, 1]}; the public API wraps them as
   * WAV bytes.
   *
   * @param text       text to speak; must not be {@code null} or blank
   * @param espeakData espeak-ng-data directory; must not be {@code null}. A missing folder is ignored.
   * @param runtime    dense kernel runtime; must not be {@code null}
   * @return mono PCM at {@link #sampleRate()}; never {@code null}
   */
  float[] synthesize(CharSequence text, Path espeakData, MatmulRuntime runtime);
}
