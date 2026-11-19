package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutEmbedding;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

final class CheckpointLoadSmokeTest {

  private static final int EMBEDDING_WIDTH = 768;

  private static void assertOneToken(final Path path, final String architecture) {
    try (LlmModel model = LlmModelFactory.make(path);
         LLM llm = LLM.builder(model).maxModelLen(128).build()) {
      assertEquals(architecture, model.architectureName());
      assertTrue(model.isCausalModel());
      assertFalse(model.isEmbeddingModel());

      List<LLM.GenerationOutput> out = llm.generate(
        List.of("Hi"),
        SamplingParams.deterministic(1));
      assertEquals(1, out.size());
      assertEquals(1, out.getFirst().tokenIds().size());
      assertTrue(out.getFirst().tokenIds().getFirst() >= 0);
      assertNotNull(out.getFirst().text());
    }
  }

  private static void assertDistinctUnitVectors(
    final Path path,
    final String architecture,
    final String text,
    final String otherText
  ) {
    try (LlmModel model = LlmModelFactory.make(path)) {
      assertEquals(architecture, model.architectureName());
      assertTrue(model.isEmbeddingModel());
      assertFalse(model.isCausalModel());

      float[] first = embed(model, text);
      float[] repeat = embed(model, text);
      float[] other = embed(model, otherText);

      assertUnit(first);
      assertTrue(Arrays.equals(first, repeat));
      assertFalse(Arrays.equals(first, other));
    }
  }

  private static float[] embed(final LlmModel model, final String text) {
    LlmOutEmbedding embedding = model.generate(LlmInText.of(text), LlmModality.EMBEDDING);
    return embedding.vector();
  }

  private static void assertUnit(final float[] vector) {
    assertEquals(EMBEDDING_WIDTH, vector.length);
    double sum = 0.0;
    for (float value : vector) {
      assertTrue(Float.isFinite(value));
      sum += (double) value * value;
    }
    assertEquals(1.0, Math.sqrt(sum), 1e-3);
  }

  @Test
  void qwen3CompletesOneToken() {
    assertOneToken(OptionalModelAssumptions.requireQwen3(), WeightNames.ARCH_QWEN3);
  }

  @Test
  void qwen3GgufCompletesOneTokenWhenAFileIsPresent() {
    assertOneToken(OptionalModelAssumptions.requireQwen3Gguf(), WeightNames.ARCH_QWEN3);
  }

  @Test
  void gemma3CompletesOneToken() {
    assertOneToken(OptionalModelAssumptions.requireGemma3(), WeightNames.ARCH_GEMMA3);
  }

  @Test
  void lfm2CompletesOneToken() {
    assertOneToken(OptionalModelAssumptions.requireLfm2Gguf(), WeightNames.ARCH_LFM2);
  }

  @Test
  void tinyLlmCompletesOneToken() {
    assertOneToken(OptionalModelAssumptions.requireTinyLlmOnnx(), WeightNames.ARCH_LLAMA);
  }

  @Test
  void embeddingGemmaReturnsAUnitVector() {
    assertDistinctUnitVectors(
      OptionalModelAssumptions.requireEmbeddingGemma(),
      WeightNames.ARCH_GEMMA3_EMBEDDING,
      "task: search result | query: hello",
      "title: none | text: astronomy");
  }

  @Test
  void xlmRobertaReturnsAUnitVector() {
    assertDistinctUnitVectors(
      OptionalModelAssumptions.requireXlmRobertaBase(),
      WeightNames.ARCH_BERT,
      "hello world",
      "totally unrelated astronomy text");
  }
}
