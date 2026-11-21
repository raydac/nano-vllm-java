package com.igormaznitsa.nanollvm.rag;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toCollection;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * A caller-supplied retrieval algorithm is a {@link RagIndex}. This file is that algorithm:
 * shared words, not BM25. {@code LLM.rag} and {@link HybridRagIndex#of} take the same object
 * as a {@link PreparedRag}.
 *
 * <pre>{@code
 * PreparedRag loaded = RagFactory.builder()
 *     .add("france.txt", "Paris is the capital of France.")
 *     .build();
 * RagIndex mine = new TokenOverlapIndex(loaded.chunks());
 * List<RagHit> hits = mine.retrieve("capital of France", 1);
 * String prompt = RagSession.formatUserMessage(hits, "What is the capital of France?");
 * RagIndex fused = HybridRagIndex.of(mine, loaded);
 * String answer = llm.rag(mine).topK(1).ask("What is the capital of France?");
 * }</pre>
 */
class CustomRagIndexTest {

  private static PreparedRag corpus() {
    return RagFactory.builder()
      .add("france.txt", "Paris is the capital of France.")
      .add("cities.txt", "France has many cities.")
      .add("germany.txt", "Berlin is a city in Germany.")
      .build();
  }

  @Test
  void customIndexFillsThePromptFromItsOwnHits() {
    PreparedRag loaded = corpus();
    TokenOverlapIndex overlap = new TokenOverlapIndex(loaded.chunks());

    assertEquals(3, overlap.size());
    List<RagHit> hits = overlap.retrieve("capital of France", 3);
    assertEquals(2, hits.size());
    assertTrue(hits.getFirst().chunk().text().contains("Paris"));
    assertEquals("france.txt", hits.getFirst().chunk().source());
    assertTrue(hits.get(1).chunk().text().contains("many cities"));
    assertTrue(hits.getFirst().score() > hits.get(1).score());
    assertTrue(hits.stream().noneMatch(hit -> hit.chunk().text().contains("Berlin")));

    String prompt = RagSession.formatUserMessage(
      overlap.retrieve("capital of France", 1),
      "What is the capital of France?");
    assertTrue(prompt.contains("Paris is the capital of France."));
    assertTrue(prompt.contains(RagPrompts.GROUNDING));
    assertTrue(prompt.endsWith("What is the capital of France?"));

    assertTrue(overlap.isOutsideCorpus("BMW motorcycles"));
    List<RagHit> none = overlap.retrieve("BMW motorcycles", 3);
    assertTrue(none.isEmpty());
    assertEquals(
      "What is a motorcycle?",
      RagSession.formatUserMessage(none, "What is a motorcycle?"));
  }

  @Test
  void customIndexBuiltFromChunksYouSupply() {
    TokenOverlapIndex overlap = new TokenOverlapIndex(List.of(
      new TextChunk("france", "france.txt", "Paris is the capital of France."),
      new TextChunk("germany", "germany.txt", "Berlin is a city in Germany.")));

    List<RagHit> hits = overlap.retrieve("capital of France", 2);
    assertEquals(1, hits.size());
    assertEquals("france", hits.getFirst().chunk().id());
    assertThrows(IllegalArgumentException.class, () -> new TokenOverlapIndex(List.of()));
    assertThrows(IllegalArgumentException.class, () -> overlap.retrieve("capital of France", 0));
  }

  @Test
  void customIndexFusesWithBm25() {
    PreparedRag loaded = corpus();
    TokenOverlapIndex overlap = new TokenOverlapIndex(loaded.chunks());

    HybridRagIndex fused = HybridRagIndex.of(overlap, loaded);
    assertEquals(List.of(overlap, loaded), fused.indexes());
    assertSame(loaded, fused.lexical());
    assertTrue(fused.retrieve("capital of France", 1).getFirst().chunk().text().contains("Paris"));
  }

  /**
   * One point for each query word of three or more letters that also appears in the passage.
   * A query that shares nothing is outside the corpus.
   */
  private static final class TokenOverlapIndex implements RagIndex {

    private static final Pattern WORD = Pattern.compile("\\p{L}{3,}");

    private final List<TextChunk> passages;

    private TokenOverlapIndex(final List<TextChunk> passages) {
      requireNonNull(passages, "passages");
      if (passages.isEmpty()) {
        throw new IllegalArgumentException("passages must not be empty");
      }
      this.passages = List.copyOf(passages);
    }

    private static double overlap(final Set<String> queryTerms, final String text) {
      Set<String> words = terms(text);
      return queryTerms.stream().filter(words::contains).count();
    }

    private static Set<String> terms(final String text) {
      return WORD.matcher(text.toLowerCase(ROOT)).results()
        .map(MatchResult::group)
        .collect(toCollection(LinkedHashSet::new));
    }

    @Override
    public List<RagHit> retrieve(final String query, final int topK) {
      requireNonNull(query, "query");
      if (topK <= 0) {
        throw new IllegalArgumentException("topK must be > 0");
      }
      Set<String> terms = terms(query);
      if (terms.isEmpty()) {
        return List.of();
      }
      return this.passages.stream()
        .map(chunk -> new RagHit(chunk, overlap(terms, chunk.text())))
        .filter(hit -> hit.score() > 0.0)
        .sorted(Comparator.comparingDouble(RagHit::score).reversed()
          .thenComparingInt(hit -> hit.chunk().text().length()))
        .limit(topK)
        .toList();
    }

    @Override
    public boolean isOutsideCorpus(final String query) {
      requireNonNull(query, "query");
      return this.retrieve(query, 1).isEmpty();
    }

    @Override
    public int size() {
      return this.passages.size();
    }
  }
}
