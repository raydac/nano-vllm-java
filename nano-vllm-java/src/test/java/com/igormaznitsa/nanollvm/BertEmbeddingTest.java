package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Path;
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
  void gteSmallEmbedsAndRejectsLlmBuilder() {
    Path path = OptionalModelAssumptions.requireGteSmallGguf();

    try (LlmModel model = LlmModelFactory.make(path)) {
      assertTrue(model.isEmbeddingModel());
      assertFalse(model.isCausalModel());
      assertEquals("bert", model.architectureName());
      String text = model.toString();
      assertTrue(text.contains("kind=embedding"), text);
      assertTrue(text.contains("architecture=bert"), text);
      assertTrue(text.contains("container=gguf"), text);

      float[] a = model.embed("hello world");
      float[] b = model.embed("hello world");
      float[] c = model.embed("totally unrelated astronomy text");

      assertEquals(384, a.length);
      assertEquals(1.0, l2(a), 1e-4);
      assertEquals(1.0f, cosine(a, b), 1e-5f);
      assertTrue(cosine(a, c) < 0.95f);

      assertThrows(IllegalStateException.class, () -> LLM.builder(model).build());

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
}
