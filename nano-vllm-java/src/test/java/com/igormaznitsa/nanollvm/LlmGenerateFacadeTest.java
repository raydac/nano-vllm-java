package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInSound;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.LlmOutEmbedding;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import com.igormaznitsa.nanollvm.models.LlmOutText;
import com.igormaznitsa.nanollvm.models.internal.audio.WavPcm;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

final class LlmGenerateFacadeTest {

  @Test
  void piperGenerateReturnsSoundDataWithSampleRate() {
    Path modelPath = OptionalModelAssumptions.requirePiper();
    Path espeak = modelPath.resolve("espeak-ng-data");

    try (LlmModel model = LlmModelFactory.open(modelPath)
      .optionalData(LlmOptionalData.ESPEAK_DATA, espeak)
      .make();
         LLM llm = LLM.builder(model).build()) {
      String phrase = modelPath.getFileName().toString().contains("-en-") ? "Hello" : "Привет";

      LlmOutSoundData sound = model.generate(LlmInText.of(phrase), LlmModality.AUDIO);
      assertTrue(sound.sampleRate() >= 8_000);
      assertTrue(sound.wav().length > 44);
      assertEquals("RIFF", new String(sound.wav(), 0, 4, StandardCharsets.US_ASCII));

      LlmOutSoundData engineSound = llm.generate(LlmInText.of(phrase), LlmModality.AUDIO);
      assertEquals(sound.sampleRate(), engineSound.sampleRate());

      assertThrows(
        IllegalStateException.class,
        () -> model.generate(LlmInText.of(phrase), LlmModality.TEXT));
      assertThrows(
        IllegalArgumentException.class,
        () -> model.generate(LlmInText.of(phrase), LlmModality.IMAGE));
    }
  }

  @Test
  void whisperGenerateTranscribesWavBytes() throws Exception {
    Path modelPath = OptionalModelAssumptions.requireWhisper();
    Path clip = Path.of("src/test/resources/wav/call1.wav");

    try (LlmModel model = LlmModelFactory.make(modelPath);
         LLM llm = LLM.builder(model).build()) {
      byte[] wav = java.nio.file.Files.readAllBytes(clip);
      LlmOutText text = llm.generate(LlmInSound.ofWav(wav), LlmModality.TEXT);
      assertTrue(text.text().length() > 5, "transcript was too short: " + text.text());

      WavPcm.MonoPcm pcm = WavPcm.read(wav);
      LlmOutText fromPcm = model.generate(
        LlmInSound.ofPcm(pcm.samples(), pcm.sampleRate()), LlmModality.TEXT);
      assertTrue(fromPcm.text().length() > 0);
    }
  }

  @Test
  void embeddingGenerateReturnsVector() {
    Path modelPath = OptionalModelAssumptions.requireMultilingualE5Small();

    try (LlmModel model = LlmModelFactory.make(modelPath);
         LLM llm = LLM.builder(model).build()) {
      LlmOutEmbedding embedding = llm.generate(LlmInText.of("query: hello"), LlmModality.EMBEDDING);
      assertTrue(embedding.vector().length > 8);

      LlmOutEmbedding fromModel =
        model.generate(LlmInText.of("query: hello"), LlmModality.EMBEDDING);
      assertEquals(embedding.vector().length, fromModel.vector().length);
    }
  }

  @Test
  void causalGenerateCompletesText() {
    Path modelPath = OptionalModelAssumptions.requireSmolLm2InstructOnnx();

    try (LlmModel model = LlmModelFactory.make(modelPath);
         LLM llm = LLM.builder(model).maxModelLen(256).build()) {
      assertThrows(
        IllegalStateException.class,
        () -> model.generate(LlmInText.of("Hello"), LlmModality.TEXT));

      LlmOutText out = llm.generate(LlmInText.of("Hello"), LlmModality.TEXT);
      assertTrue(out.text() != null);
    }
  }
}
