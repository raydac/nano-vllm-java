package com.igormaznitsa.nanollvm.samples;

import static java.util.Locale.ROOT;
import static java.util.stream.Collectors.toCollection;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import com.igormaznitsa.nanollvm.rag.HybridRagIndex;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.rag.TextChunk;
import com.igormaznitsa.nanollvm.samples.utils.SampleModelAssumptions;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class JVectorRagIndexTest {

  private static PreparedRag corpus() {
    return RagFactory.builder()
      .add("france.txt", "Paris is the capital of France.")
      .add("cities.txt", "France has many cities.")
      .add("germany.txt", "Berlin is a city in Germany.")
      .add("river.txt", "The Nile is a long river in Africa.")
      .build();
  }

  @Test
  void jvectorIndexFillsThePromptFromNearestPassage() {
    PreparedRag loaded = corpus();
    TermVectors terms = TermVectors.from(loaded.chunks());
    JVectorRagIndex index =
      JVectorRagIndex.of(loaded.chunks(), terms.embedAll(loaded.chunks()), terms::embed);

    assertEquals(4, index.size());
    List<RagHit> hits = index.retrieve("capital of France", 3);
    assertEquals(2, hits.size());
    assertTrue(hits.getFirst().chunk().text().contains("Paris"));
    assertEquals("france.txt", hits.getFirst().chunk().source());
    assertTrue(hits.get(1).chunk().text().contains("many cities"));
    assertTrue(hits.getFirst().score() > hits.get(1).score());
    assertTrue(hits.stream().noneMatch(hit -> hit.chunk().text().contains("Berlin")));
    assertTrue(hits.stream().noneMatch(hit -> hit.chunk().text().contains("Nile")));

    List<RagHit> river = index.retrieve("long river in Africa", 1);
    assertEquals(1, river.size());
    assertTrue(river.getFirst().chunk().text().contains("Nile"));

    String prompt = RagSession.formatUserMessage(
      index.retrieve("capital of France", 1),
      "What is the capital of France?");
    assertTrue(prompt.contains("Paris is the capital of France."));
    assertTrue(prompt.contains(RagPrompts.GROUNDING));
    assertTrue(prompt.endsWith("What is the capital of France?"));

    assertTrue(index.isOutsideCorpus("BMW motorcycles"));
    assertTrue(index.retrieve("BMW motorcycles", 3).isEmpty());
    assertEquals(
      "What is a motorcycle?",
      RagSession.formatUserMessage(List.of(), "What is a motorcycle?"));
  }

  @Test
  void jvectorIndexRejectsUnusableInput() {
    TextChunk paris = TextChunk.of("france", "Paris is the capital of France.");
    TermVectors terms = TermVectors.from(List.of(paris));
    JVectorRagIndex index = JVectorRagIndex.of(
      List.of(paris),
      terms.embedAll(List.of(paris)),
      terms::embed);

    assertThrows(IllegalArgumentException.class,
      () -> JVectorRagIndex.of(List.of(), new float[][] {}, terms::embed));
    assertThrows(IllegalArgumentException.class, () -> index.retrieve("capital of France", 0));
    assertThrows(
      IllegalArgumentException.class,
      () -> JVectorRagIndex.of(List.of(paris), new float[][] {new float[] {0f, 0f}}, terms::embed));
    assertThrows(
      IllegalArgumentException.class,
      () -> JVectorRagIndex.of(
        List.of(paris),
        new float[][] {new float[] {1f}, new float[] {1f}},
        terms::embed));
  }

  @Test
  void jvectorIndexFusesWithBm25() {
    PreparedRag loaded = corpus();
    TermVectors terms = TermVectors.from(loaded.chunks());
    JVectorRagIndex index =
      JVectorRagIndex.of(loaded.chunks(), terms.embedAll(loaded.chunks()), terms::embed);

    HybridRagIndex fused = HybridRagIndex.of(index, loaded);
    assertEquals(List.of(index, loaded), fused.indexes());
    assertTrue(fused.retrieve("capital of France", 1).getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void jvectorIndexRetrievesParaphraseWhenGtePresent() {
    Path gte = SampleModelAssumptions.requireGteSmallGguf();
    PreparedRag passages = RagFactory.builder()
      .add("paris", "Paris is the capital of France.")
      .add("nile", "The Nile is a long river in Africa.")
      .add("tokyo", "Tokyo is the capital of Japan.")
      .build();

    try (LlmModel encoder = LlmModelFactory.make(gte)) {
      JVectorRagIndex index = JVectorRagIndex.of(passages, encoder);

      assertEquals(passages.size(), index.size());
      List<RagHit> hits = index.retrieve("What city is France's capital?", 1);
      assertTrue(hits.getFirst().chunk().text().contains("Paris"));
      assertTrue(hits.getFirst().score() > 0.5);

      String prompt = RagSession.formatUserMessage(hits, "What city is France's capital?");
      assertTrue(prompt.contains("Paris is the capital of France."));
    }
  }

  private static final class TermVectors {

    private static final Pattern WORD = Pattern.compile("\\p{L}{3,}");

    private final Map<String, Integer> axes;

    private TermVectors(final Map<String, Integer> axes) {
      this.axes = axes;
    }

    private static TermVectors from(final List<TextChunk> passages) {
      Map<String, Integer> axes = new LinkedHashMap<>();
      for (TextChunk passage : passages) {
        for (String term : terms(passage.text())) {
          axes.putIfAbsent(term, axes.size());
        }
      }
      if (axes.isEmpty()) {
        throw new IllegalArgumentException("passages must contain indexable words");
      }
      return new TermVectors(axes);
    }

    private static Set<String> terms(final String text) {
      return WORD.matcher(text.toLowerCase(ROOT)).results()
        .map(MatchResult::group)
        .collect(toCollection(LinkedHashSet::new));
    }

    private float[] embed(final String text) {
      float[] vector = new float[this.axes.size()];
      for (String term : terms(text)) {
        Integer axis = this.axes.get(term);
        if (axis != null) {
          vector[axis] = 1f;
        }
      }
      return vector;
    }

    private float[][] embedAll(final List<TextChunk> passages) {
      return passages.stream().map(passage -> this.embed(passage.text())).toArray(float[][]::new);
    }
  }
}
