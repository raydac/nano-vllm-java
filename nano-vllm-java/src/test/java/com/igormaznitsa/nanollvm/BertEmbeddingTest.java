package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModalities;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Path;
import java.util.Set;
import org.junit.jupiter.api.Test;

final class BertEmbeddingTest {

  private static double l2(final float[] v) {
    double sum = 0.0;
    for (float x : v) {
      sum += (double) x * x;
    }
    return Math.sqrt(sum);
  }

  private static float cosine(final float[] a, final float[] b) {
    float dot = 0f;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  @Test
  void gteSmallEmbedsThroughLlmBuilder() {
    Path path = OptionalModelAssumptions.requireGteSmallGguf();

    try (LlmModel model = LlmModelFactory.make(path)) {
      assertTrue(model.isEmbeddingModel());
      assertFalse(model.isCausalModel());
      assertEquals("bert", model.architectureName());
      assertEquals(LlmModalities.TEXT_TO_EMBEDDING, model.modalities());
      assertEquals(LlmModalities.TEXT_TO_EMBEDDING, model.usableModalities());
      assertEquals(Set.of(LlmModality.TEXT), model.inputModalities());
      assertEquals(Set.of(LlmModality.EMBEDDING), model.outputModalities());
      String text = model.toString();
      assertTrue(text.contains("kind=embedding"), text);
      assertTrue(text.contains("modalities=text->embedding"), text);
      assertTrue(text.contains("architecture=bert"), text);
      assertTrue(text.contains("container=gguf"), text);

      float[] a = model.embed("hello world");
      float[] b = model.embed("hello world");
      float[] c = model.embed("totally unrelated astronomy text");

      assertEquals(384, a.length);
      assertEquals(1.0, l2(a), 1e-4);
      assertEquals(1.0f, cosine(a, b), 1e-5f);
      assertTrue(cosine(a, c) < 0.95f);

      try (LLM llm = LLM.builder(model).build()) {
        assertThrows(IllegalStateException.class, llm::chat);
        assertEquals(0, llm.config().numKvcacheBlocks());
        assertEquals(1.0f, cosine(a, llm.embed("hello world")), 1e-5f);
      }

      int cls = model.tokenizer().tokenId("[CLS]").orElseThrow();
      int sep = model.tokenizer().tokenId("[SEP]").orElseThrow();
      var pieces = model.tokenizer().encode("hello world");
      int[] ids = new int[pieces.size() + 2];
      ids[0] = cls;
      for (int i = 0; i < pieces.size(); i++) {
        ids[i + 1] = pieces.get(i);
      }
      ids[ids.length - 1] = sep;
      assertEquals(1.0f, cosine(a, model.embed(ids)), 1e-5f);
      assertThrows(IllegalArgumentException.class, () -> model.embed(new int[0]));
    }
  }

  @Test
  void multilingualE5SmallEmbedsThroughLlmBuilder() {
    Path path = OptionalModelAssumptions.requireMultilingualE5Small();

    try (LlmModel model = LlmModelFactory.make(path)) {
      assertTrue(model.isEmbeddingModel());
      assertFalse(model.isCausalModel());
      assertEquals("bert", model.architectureName());
      assertEquals(LlmModalities.TEXT_TO_EMBEDDING, model.modalities());
      assertEquals(LlmModalities.TEXT_TO_EMBEDDING, model.usableModalities());
      String text = model.toString();
      assertTrue(text.contains("kind=embedding"), text);
      assertTrue(text.contains("modalities=text->embedding"), text);
      assertTrue(text.contains("architecture=bert"), text);
      assertTrue(text.contains("container=folder"), text);

      float[] a = model.embed("query: hello world");
      float[] b = model.embed("query: hello world");
      float[] c = model.embed("query: totally unrelated astronomy text");

      assertEquals(384, a.length);
      assertEquals(1.0, l2(a), 1e-4);
      assertEquals(1.0f, cosine(a, b), 1e-5f);
      assertTrue(cosine(a, c) < 0.99f);

      try (LLM llm = LLM.builder(model).build()) {
        assertThrows(IllegalStateException.class, llm::chat);
        assertEquals(0, llm.config().numKvcacheBlocks());
        assertEquals(1.0f, cosine(a, llm.embed("query: hello world")), 1e-5f);
      }

      int cls = model.tokenizer().tokenId("<s>").orElseThrow();
      int sep = model.tokenizer().tokenId("</s>").orElseThrow();
      var pieces = model.tokenizer().encode("query: hello world");
      int[] ids = new int[pieces.size() + 2];
      ids[0] = cls;
      for (int i = 0; i < pieces.size(); i++) {
        ids[i + 1] = pieces.get(i);
      }
      ids[ids.length - 1] = sep;
      assertEquals(1.0f, cosine(a, model.embed(ids)), 1e-5f);
    }
  }
}
