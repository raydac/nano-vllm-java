package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModalities;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.models.internal.audio.WavPcm;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

final class TinyVocaLoadTest {

  private static void speak(final Path modelPath, final String text, final int sampleRate)
    throws Exception {
    try (LlmModel model = LlmModelFactory.open(modelPath).make()) {
      assertTrue(model.isSynthesisModel());
      assertTrue(model.hfConfig().isTinyVoca());
      assertEquals(WeightNames.ARCH_TINY_VOCA, model.architectureName());
      assertEquals(LlmModalities.TEXT_TO_AUDIO, model.usableModalities());
      try (LLM llm = LLM.builder(model).build()) {
        IllegalStateException chat = assertThrows(IllegalStateException.class, llm::chat);
        assertEquals(
          ModelSupport.synthesisEngineMisuseMessage(model.architectureName()), chat.getMessage());
        LlmOutSoundData sound = llm.generate(LlmInText.of(text), LlmModality.AUDIO);
        assertEquals(sampleRate, sound.sampleRate());
        byte[] wav = sound.wav();
        assertEquals("RIFF", new String(wav, 0, 4, StandardCharsets.US_ASCII));
        WavPcm.MonoPcm pcm = WavPcm.read(wav);
        float peak = 0f;
        for (float sample : pcm.samples()) {
          assertTrue(Float.isFinite(sample));
          peak = Math.max(peak, Math.abs(sample));
        }
        assertTrue(peak > 0.05f, "synthesized peak was too quiet: " + peak);
        assertTrue(pcm.samples().length > 4_000, "synthesized clip was too short");
      }
      assertFalse(model.isSpeechModel());
    }
  }

  @Test
  void englishCheckpointSpeaksHello() throws Exception {
    speak(OptionalModelAssumptions.requireTinyVoca(), "voice=F2 Hello", 24_000);
  }

  @Test
  void russianCheckpointSpeaksGreeting() throws Exception {
    speak(OptionalModelAssumptions.requireTinyVocaRussian(), "voice=F Привет", 24_000);
  }
}
