package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_WHISPER;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.SpeechToText;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WhisperForAsr;

/**
 * OpenAI Whisper speech-to-text from Hugging Face safetensors.
 *
 * @since 1.3.0
 */
final class WhisperProcessor extends SpeechArchitecture {

  static final WhisperProcessor INSTANCE = new WhisperProcessor();

  private WhisperProcessor() {
  }

  @Override
  public String architectureId() {
    return ARCH_WHISPER;
  }

  @Override
  public SpeechToText createSpeech(final Config.HfConfig config, final WeightBag weights) {
    return new WhisperForAsr(config, weights);
  }
}
