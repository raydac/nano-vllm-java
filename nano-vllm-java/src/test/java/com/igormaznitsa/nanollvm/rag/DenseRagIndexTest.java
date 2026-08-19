package com.igormaznitsa.nanollvm.rag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class DenseRagIndexTest {

  private static void assertFalseEmpty(final List<RagHit> hits) {
    assertTrue(!hits.isEmpty(), "expected non-empty hits");
  }

  @Test
  void fusePrefersAgreementAcrossRanks() {
    TextChunk paris = TextChunk.of("paris", "Paris is the capital of France.");
    TextChunk wolf = TextChunk.of("wolf", "Little Red Riding Hood met a wolf.");
    TextChunk nile = TextChunk.of("nile", "The Nile is a long river in Africa.");

    List<RagHit> lexical = List.of(new RagHit(paris, 2.0), new RagHit(wolf, 1.0));
    List<RagHit> dense = List.of(new RagHit(nile, 0.9), new RagHit(paris, 0.8));

    List<RagHit> fused = HybridRagIndex.fuse(lexical, dense, 2);
    assertEquals(2, fused.size());
    assertEquals("paris", fused.getFirst().chunk().id());
  }

  @Test
  void withEmbeddingsRejectsCausalModel() {
    Path qwen = OptionalModelAssumptions.requireQwen3();
    PreparedRag lexical = RagFactory.of("Paris is the capital of France.");
    try (LlmModel causal = LlmModelFactory.make(qwen)) {
      IllegalArgumentException ex = assertThrows(
        IllegalArgumentException.class,
        () -> RagFactory.withEmbeddings(lexical, causal));
      assertTrue(ex.getMessage().contains("embedding encoder"));
    }
  }

  @Test
  void ofReportsPassageProgressWhenGtePresent() {
    Path gte = OptionalModelAssumptions.requireGteSmallGguf();
    PreparedRag lexical = RagFactory.builder()
      .add("paris", "Paris is the capital of France.")
      .add("nile", "The Nile is a long river in Africa.")
      .add("tokyo", "Tokyo is the capital of Japan.")
      .build();
    AtomicInteger completed = new AtomicInteger();

    try (LlmModel embed = LlmModelFactory.make(gte)) {
      DenseRagIndex index = DenseRagIndex.of(lexical, embed, completed::set);
      assertEquals(lexical.size(), index.size());
      assertEquals(lexical.size(), completed.get());
    }
  }

  @Test
  void ofSubmitsPassagesToCallerExecutorWhenGtePresent() {
    Path gte = OptionalModelAssumptions.requireGteSmallGguf();
    PreparedRag lexical = RagFactory.builder()
      .add("paris", "Paris is the capital of France.")
      .add("nile", "The Nile is a long river in Africa.")
      .build();
    AtomicInteger submitted = new AtomicInteger();
    Executor executor = command -> {
      submitted.incrementAndGet();
      command.run();
    };

    try (LlmModel embed = LlmModelFactory.make(gte)) {
      DenseRagIndex index = DenseRagIndex.of(lexical, embed, executor);
      assertEquals(lexical.size(), index.size());
      assertEquals(lexical.size(), submitted.get());
    }
  }

  @Test
  void hybridDenseRetrievesParaphraseWhenGtePresent() {
    Path gte = OptionalModelAssumptions.requireGteSmallGguf();
    PreparedRag lexical = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .add("capital.txt", "Paris is the capital of France.")
      .add("river.txt", "The Nile is a long river in Africa.")
      .build();

    try (LlmModel embed = LlmModelFactory.make(gte)) {
      HybridRagIndex hybrid = RagFactory.withEmbeddings(lexical, embed);
      assertEquals(lexical.size(), hybrid.size());

      List<RagHit> hits = hybrid.retrieve("What city is France's capital?", 2);
      assertTrue(
        hits.stream().anyMatch(hit -> hit.chunk().text().contains("Paris")),
        () -> "expected Paris among hits: " + hits);

      List<RagHit> denseOnly = hybrid.dense().retrieve("What city is France's capital?", 1);
      assertFalseEmpty(denseOnly);
      assertTrue(denseOnly.getFirst().chunk().text().contains("Paris"));
    }
  }
}
