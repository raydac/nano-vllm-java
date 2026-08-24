package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModalities;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import com.igormaznitsa.nanollvm.testsupport.TestWavs;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import org.junit.jupiter.api.Test;

final class WhisperLoadTest {

  private static final String CALL_PROMPT =
    "Thank you for contacting us. All lines are currently busy. Your call is very important to us.";

  private static String lettersAndSpaces(final String text) {
    return text.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", " ").strip();
  }

  @Test
  void loadsWhisperAndTranscribesBundledCallWav() throws Exception {
    Path modelPath = OptionalModelAssumptions.requireWhisper();
    Path wav = TestWavs.classpathFile("wav/call1.wav");

    try (LlmModel model = LlmModelFactory.make(modelPath)) {
      assertTrue(model.isSpeechModel());
      assertFalse(model.isEmbeddingModel());
      assertEquals(WeightNames.ARCH_WHISPER, model.architectureName());
      assertEquals(LlmModalities.AUDIO_TO_TEXT, model.usableModalities());

      IllegalStateException embed = assertThrows(
        IllegalStateException.class, () -> model.embed("hello"));
      assertTrue(embed.getMessage().contains("speech"));

      try (LLM llm = LLM.builder(model).build()) {
        IllegalStateException chat = assertThrows(IllegalStateException.class, llm::chat);
        assertEquals(
          ModelSupport.speechEngineMisuseMessage(model.architectureName()), chat.getMessage());
        assertEquals(0, llm.config().numKvcacheBlocks());

        String transcript = llm.transcribe(Files.readAllBytes(wav), Locale.ENGLISH);
        String normalized = lettersAndSpaces(transcript);
        assertTrue(
          normalized.contains(lettersAndSpaces(CALL_PROMPT)),
          () -> "transcript was: " + transcript);
      }
    }
  }
}
