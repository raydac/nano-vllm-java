package com.igormaznitsa.nanollvm.rag;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toCollection;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutEmbedding;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * JVector is the vector store behind a {@link RagIndex}: passage vectors are inserted into an
 * in-memory graph, and a query vector is the nearest-neighbor search. The hit score is cosine.
 *
 * <pre>{@code
 * List<TextChunk> passages = loaded.chunks();
 * TermVectors terms = TermVectors.from(passages);
 * RagIndex mine = JVectorRagIndex.of(passages, terms.embedAll(passages), terms::embed);
 * List<RagHit> hits = mine.retrieve("capital of France", 1);
 * String prompt = RagSession.formatUserMessage(hits, "What is the capital of France?");
 * RagIndex fused = HybridRagIndex.of(mine, loaded);
 * }</pre>
 */
class JVectorRagIndexTest {

  static {
    // JVector 4 loads a bundled native library unless this is set. Panama is enough here.
    System.setProperty("jvector.enable_native_vectorization", "false");
  }

  private static PreparedRag corpus() {
    return RagFactory.builder()
      .add("france.txt", "Paris is the capital of France.")
      .add("cities.txt", "France has many cities.")
      .add("germany.txt", "Berlin is a city in Germany.")
      .add("river.txt", "The Nile is a long river in Africa.")
      .build();
  }

  private static float[] embed(final LlmModel encoder, final String text) {
    LlmOutEmbedding embedding = encoder.generate(LlmInText.of(text), LlmModality.EMBEDDING);
    return embedding.vector();
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
    Path gte = OptionalModelAssumptions.requireGteSmallGguf();
    List<TextChunk> passages = List.of(
      TextChunk.of("paris", "Paris is the capital of France."),
      TextChunk.of("nile", "The Nile is a long river in Africa."),
      TextChunk.of("tokyo", "Tokyo is the capital of Japan."));

    try (LlmModel encoder = LlmModelFactory.make(gte)) {
      float[][] vectors =
        passages.stream().map(chunk -> embed(encoder, chunk.text())).toArray(float[][]::new);
      JVectorRagIndex index = JVectorRagIndex.of(passages, vectors, text -> embed(encoder, text));

      assertEquals(passages.size(), index.size());
      List<RagHit> hits = index.retrieve("What city is France's capital?", 1);
      assertEquals("paris", hits.getFirst().chunk().id());
      assertTrue(hits.getFirst().score() > 0.5);

      String prompt = RagSession.formatUserMessage(hits, "What city is France's capital?");
      assertTrue(prompt.contains("Paris is the capital of France."));
    }
  }

  /**
   * One axis per indexed word of three or more letters. A query that shares none of those words
   * is the zero vector, so retrieval returns no hits.
   */
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

  /**
   * In-memory JVector graph over one vector per passage. {@code embedQuery} is called on each
   * search; passage vectors are copied in at construction and are not read again from the caller.
   */
  private static final class JVectorRagIndex implements RagIndex {

    private static final VectorTypeSupport VECTORS =
      VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final int MAX_DEGREE = 16;
    private static final int BEAM_WIDTH = 100;
    private static final float NEIGHBOR_OVERFLOW = 1.2f;
    private static final float ALPHA = 1.2f;
    private static final double MIN_COSINE = 1e-4;

    private final List<TextChunk> passages;
    private final RandomAccessVectorValues vectors;
    private final ImmutableGraphIndex graph;
    private final Function<String, float[]> embedQuery;
    private final int dimension;

    private JVectorRagIndex(
      final List<TextChunk> passages,
      final RandomAccessVectorValues vectors,
      final ImmutableGraphIndex graph,
      final Function<String, float[]> embedQuery,
      final int dimension
    ) {
      this.passages = passages;
      this.vectors = vectors;
      this.graph = graph;
      this.embedQuery = embedQuery;
      this.dimension = dimension;
    }

    private static JVectorRagIndex of(
      final List<TextChunk> passages,
      final float[][] vectors,
      final Function<String, float[]> embedQuery
    ) {
      List<TextChunk> copied = List.copyOf(requireNonNull(passages, "passages"));
      requireNonNull(vectors, "vectors");
      requireNonNull(embedQuery, "embedQuery");
      if (copied.isEmpty()) {
        throw new IllegalArgumentException("passages must not be empty");
      }
      if (vectors.length != copied.size()) {
        throw new IllegalArgumentException("vectors length must match passages");
      }

      int dimension = requireNonNull(vectors[0], "vector[0]").length;
      List<VectorFloat<?>> embedded = new ArrayList<>(copied.size());
      for (int i = 0; i < copied.size(); i++) {
        embedded.add(VECTORS.createFloatVector(unitPassage(vectors[i], dimension, i)));
      }
      RandomAccessVectorValues values = new ListRandomAccessVectorValues(embedded, dimension);
      return new JVectorRagIndex(copied, values, buildGraph(values, dimension), embedQuery,
        dimension);
    }

    private static ImmutableGraphIndex buildGraph(final RandomAccessVectorValues vectors,
                                                  final int dimension) {
      try (GraphIndexBuilder builder = GraphIndexBuilder.builder(
          BuildScoreProvider.randomAccessScoreProvider(vectors, VectorSimilarityFunction.COSINE),
          dimension,
          MAX_DEGREE)
        .withBeamWidth(BEAM_WIDTH)
        .withNeighborOverflow(NEIGHBOR_OVERFLOW)
        .withAlpha(ALPHA)
        .build()) {
        return builder.build(vectors);
      } catch (IOException ex) {
        throw new IllegalStateException("JVector graph was not built", ex);
      }
    }

    private static float[] unitPassage(final float[] raw, final int dimension, final int index) {
      if (raw == null || raw.length != dimension || dimension == 0) {
        throw new IllegalArgumentException("vector[" + index + "] must have length " + dimension);
      }
      float[] copy = raw.clone();
      for (float value : copy) {
        if (!Float.isFinite(value)) {
          throw new IllegalArgumentException("vector[" + index + "] must be finite");
        }
      }
      if (!scaleToUnit(copy)) {
        throw new IllegalArgumentException("vector[" + index + "] must not be zero");
      }
      return copy;
    }

    private static boolean scaleToUnit(final float[] vector) {
      double energy = 0.0;
      for (float value : vector) {
        energy += (double) value * value;
      }
      if (energy == 0.0) {
        return false;
      }
      float scale = (float) (1.0 / Math.sqrt(energy));
      for (int i = 0; i < vector.length; i++) {
        vector[i] *= scale;
      }
      return true;
    }

    // JVector's COSINE compare returns (1 + cosine) / 2.
    private static double cosine(final float similarity) {
      return similarity * 2.0 - 1.0;
    }

    @Override
    public List<RagHit> retrieve(final String query, final int topK) {
      requireNonNull(query, "query");
      if (topK <= 0) {
        throw new IllegalArgumentException("topK must be > 0");
      }

      float[] queryVector = requireNonNull(this.embedQuery.apply(query), "query vector").clone();
      if (queryVector.length != this.dimension) {
        throw new IllegalArgumentException("query vector length must be " + this.dimension);
      }
      if (!scaleToUnit(queryVector)) {
        return List.of();
      }

      return this.search(VECTORS.createFloatVector(queryVector), topK);
    }

    private List<RagHit> search(final VectorFloat<?> query, final int topK) {
      try (GraphSearcher searcher = new GraphSearcher(this.graph)) {
        return Arrays.stream(searcher.search(
            DefaultSearchScoreProvider.exact(query, VectorSimilarityFunction.COSINE, this.vectors),
            topK,
            Bits.ALL).getNodes())
          .map(node -> new RagHit(this.passages.get(node.node), cosine(node.score)))
          .filter(hit -> hit.score() > MIN_COSINE)
          .toList();
      } catch (IOException ex) {
        throw new IllegalStateException("JVector searcher was not closed", ex);
      }
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
