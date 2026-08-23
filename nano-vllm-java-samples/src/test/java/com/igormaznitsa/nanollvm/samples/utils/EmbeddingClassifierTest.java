package com.igormaznitsa.nanollvm.samples.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier.LabeledText;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier.Prediction;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class EmbeddingClassifierTest {

  private static float[] unit(final float x, final float y, final float z) {
    double norm = Math.sqrt((double) x * x + (double) y * y + (double) z * z);
    return new float[] {(float) (x / norm), (float) (y / norm), (float) (z / norm)};
  }

  @Test
  void parseLabeledLineAcceptsPipeAndTab() {
    assertEquals(
      Optional.of(new LabeledText("poem", "frosty winter")),
      EmbeddingClassifier.parseLabeledLine("  Poem | frosty winter  "));
    assertEquals(
      Optional.of(new LabeledText("news", "rates rose")),
      EmbeddingClassifier.parseLabeledLine("NEWS\trates rose"));
    assertEquals(
      Optional.of(new LabeledText("chat", "hi")),
      EmbeddingClassifier.parseLabeledLine("chat|hi"));
  }

  @Test
  void parseLabeledLineSkipsBlankCommentAndUnlabeled() {
    assertTrue(EmbeddingClassifier.parseLabeledLine("").isEmpty());
    assertTrue(EmbeddingClassifier.parseLabeledLine("   ").isEmpty());
    assertTrue(EmbeddingClassifier.parseLabeledLine("# comment").isEmpty());
    assertTrue(EmbeddingClassifier.parseLabeledLine("no separator here").isEmpty());
    assertTrue(EmbeddingClassifier.parseLabeledLine("| only text").isEmpty());
    assertTrue(EmbeddingClassifier.parseLabeledLine("label |").isEmpty());
  }

  @Test
  void fitRejectsASingleLabel() {
    EmbeddingClassifier.Trainer trainer = EmbeddingClassifier.trainer()
      .add("poem", unit(1f, 0f, 0f));
    assertFalse(trainer.canFit());
    assertThrows(IllegalStateException.class, trainer::fit);
  }

  @Test
  void centeredPrototypesSeparateSharedBiasDirections() {
    EmbeddingClassifier classifier = EmbeddingClassifier.trainer()
      .add("poem", unit(1f, 0.04f, 0f))
      .add("poem", unit(1f, 0.05f, 0.01f))
      .add("chat", unit(1f, 0f, 0.04f))
      .add("chat", unit(1f, 0.01f, 0.05f))
      .fit();

    Prediction poem = classifier.classify(unit(1f, 0.06f, 0f));
    Prediction chat = classifier.classify(unit(1f, 0f, 0.06f));

    assertEquals("poem", poem.label());
    assertEquals("chat", chat.label());
    assertFalse(poem.isClose(0.02));
    assertEquals(2, classifier.labels().size());
    assertEquals(4, classifier.exampleCount());
  }

  @Test
  void classifyRejectsDimensionMismatch() {
    EmbeddingClassifier classifier = EmbeddingClassifier.trainer()
      .add("poem", unit(1f, 0.04f, 0f))
      .add("chat", unit(1f, 0f, 0.04f))
      .fit();
    assertThrows(IllegalArgumentException.class,
      () -> classifier.classify(new float[] {1f, 0f}));
  }

  @Test
  void addRejectsDimensionMismatch() {
    EmbeddingClassifier.Trainer trainer = EmbeddingClassifier.trainer()
      .add("poem", new float[] {1f, 0f});
    assertThrows(IllegalArgumentException.class,
      () -> trainer.add("chat", new float[] {1f, 0f, 0f}));
  }
}
