package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModalities;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutLabels;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Path;
import java.util.Locale;
import org.junit.jupiter.api.Test;

final class FastTextLoadTest {

  @Test
  void loadsLidAndClassifiesLanguages() {
    Path modelPath = OptionalModelAssumptions.requireFastTextLid();

    try (LlmModel model = LlmModelFactory.make(modelPath)) {
      assertTrue(model.isClassificationModel());
      assertFalse(model.isEmbeddingModel());
      assertFalse(model.isSpeechModel());
      assertFalse(model.isSynthesisModel());
      assertEquals(WeightNames.ARCH_FASTTEXT, model.architectureName());
      assertEquals(LlmModalities.TEXT_TO_LABELS, model.usableModalities());
      assertTrue(ModelSupport.isClassificationCheckpoint(modelPath));

      IllegalStateException embed = assertThrows(
        IllegalStateException.class,
        () -> model.generate(LlmInText.of("hello"), LlmModality.EMBEDDING));
      assertTrue(embed.getMessage().toLowerCase(Locale.ROOT).contains("classif"));

      try (LLM llm = LLM.builder(model).build()) {
        IllegalStateException chat = assertThrows(IllegalStateException.class, llm::chat);
        assertEquals(
          ModelSupport.classificationEngineMisuseMessage(model.architectureName()),
          chat.getMessage());
        assertEquals(0, llm.config().numKvcacheBlocks());

        LlmOutLabels english = assertInstanceOf(
          LlmOutLabels.class,
          llm.generate(LlmInText.of("The capital of France is Paris."), LlmModality.LABELS));
        assertTrue(english.topLabel().endsWith("en"), english.topLabel());
        assertTrue(english.top().score() > 0.5f, Float.toString(english.top().score()));

        LlmOutLabels french = assertInstanceOf(
          LlmOutLabels.class,
          llm.generate(LlmInText.of("Bonjour, comment allez-vous ?"), LlmModality.LABELS));
        assertTrue(french.topLabel().endsWith("fr"), french.topLabel());
      }
    }
  }
}
