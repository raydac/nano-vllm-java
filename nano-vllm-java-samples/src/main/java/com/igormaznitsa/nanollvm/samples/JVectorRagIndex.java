package com.igormaznitsa.nanollvm.samples;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmOutEmbedding;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagIndex;
import com.igormaznitsa.nanollvm.rag.TextChunk;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * In-memory JVector graph over one embedding per passage. A query is the nearest-neighbor
 * search. The hit score is cosine. The embedding model stays open for later queries; this index
 * does not close it.
 *
 * <p>JVector is a sample dependency. It is not on the published library jar.
 *
 * <pre>{@code
 * try (LlmModel encoder = LlmModelFactory.make(encoderPath)) {
 *   RagIndex index = JVectorRagIndex.of(notes, encoder);
 *   String answer = llm.rag(index).topK(4).ask("When is live chat open?");
 * }
 * }</pre>
 */
public final class JVectorRagIndex implements RagIndex {

  private static final VectorTypeSupport VECTORS =
    VectorizationProvider.getInstance().getVectorTypeSupport();
  private static final int MAX_DEGREE = 16;
  private static final int BEAM_WIDTH = 100;
  private static final float NEIGHBOR_OVERFLOW = 1.2f;
  private static final float ALPHA = 1.2f;
  private static final double MIN_COSINE = 1e-4;
  private static final double HIT_FLOOR_RATIO = 0.45;
  private static final double OUTSIDE_MAX_SIMILARITY = 0.28;

  // JVector 4 loads a bundled native library unless this is set before the provider starts.
  static {
    if (System.getProperty("jvector.enable_native_vectorization") == null) {
      System.setProperty("jvector.enable_native_vectorization", "false");
    }
  }

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

  /**
   * Embeds every chunk with {@code encoder} and inserts those vectors into a graph.
   *
   * @param corpus  passages to index; must contain at least one chunk
   * @param encoder embedding model kept open for each later query
   * @return a graph over {@code corpus}
   */
  public static JVectorRagIndex of(final PreparedRag corpus, final LlmModel encoder) {
    requireNonNull(corpus, "corpus");
    return of(corpus.chunks(), encoder);
  }

  /**
   * Embeds every passage with {@code encoder} and inserts those vectors into a graph.
   *
   * @param passages passages to index; must not be empty
   * @param encoder  embedding model kept open for each later query
   * @return a graph over {@code passages}
   */
  public static JVectorRagIndex of(final List<TextChunk> passages, final LlmModel encoder) {
    requireEncoder(encoder);
    List<TextChunk> copied = List.copyOf(requireNonNull(passages, "passages"));
    float[][] vectors = copied.stream()
      .map(chunk -> embed(encoder, passageText(chunk)))
      .toArray(float[][]::new);
    return of(copied, vectors, text -> embed(encoder, text));
  }

  /**
   * Inserts caller-supplied passage vectors. {@code embedQuery} runs on each search.
   *
   * @param passages   passages, one per row of {@code vectors}
   * @param vectors    one finite non-zero vector per passage, all the same length
   * @param embedQuery maps a query string to a vector of that length
   * @return a graph over {@code passages}
   */
  public static JVectorRagIndex of(
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

  private static void requireEncoder(final LlmModel encoder) {
    requireNonNull(encoder, "encoder");
    if (!encoder.isEmbeddingModel()) {
      throw new IllegalArgumentException(
        "encoder must be an embedding model (got " + encoder.architectureName() + ")");
    }
    if (encoder.isClosed()) {
      throw new IllegalStateException("encoder is closed");
    }
  }

  private static String passageText(final TextChunk chunk) {
    return chunk.text().isBlank() ? chunk.id() : chunk.text();
  }

  private static float[] embed(final LlmModel encoder, final String text) {
    String body = text == null || text.isBlank() ? "_" : text;
    LlmOutEmbedding embedding = encoder.generate(LlmInText.of(body), LlmModality.EMBEDDING);
    return embedding.vector();
  }

  private static ImmutableGraphIndex buildGraph(
    final RandomAccessVectorValues vectors,
    final int dimension
  ) {
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
  private static double cosine(final float packedSimilarity) {
    return packedSimilarity * 2.0 - 1.0;
  }

  @Override
  public List<RagHit> retrieve(final String query, final int topK) {
    requireNonNull(query, "query");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    if (query.isBlank()) {
      return List.of();
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

  @Override
  public boolean isOutsideCorpus(final String query) {
    requireNonNull(query, "query");
    if (query.isBlank()) {
      return true;
    }
    List<RagHit> best = this.retrieve(query, 1);
    return best.isEmpty() || best.getFirst().score() < OUTSIDE_MAX_SIMILARITY;
  }

  @Override
  public int size() {
    return this.passages.size();
  }

  @Override
  public String toString() {
    return "JVectorRagIndex{passages=%d, dim=%d}".formatted(this.size(), this.dimension);
  }

  private List<RagHit> search(final VectorFloat<?> query, final int topK) {
    try (GraphSearcher searcher = new GraphSearcher(this.graph)) {
      List<RagHit> ranked = Arrays.stream(searcher.search(
          DefaultSearchScoreProvider.exact(query, VectorSimilarityFunction.COSINE, this.vectors),
          topK,
          Bits.ALL).getNodes())
        .map(node -> new RagHit(this.passages.get(node.node), cosine(node.score)))
        .filter(hit -> hit.score() > MIN_COSINE)
        .toList();
      return this.keepNearBest(ranked);
    } catch (IOException ex) {
      throw new IllegalStateException("JVector searcher was not closed", ex);
    }
  }

  private List<RagHit> keepNearBest(final List<RagHit> ranked) {
    if (ranked.isEmpty()) {
      return List.of();
    }
    double floor = ranked.getFirst().score() * HIT_FLOOR_RATIO;
    return ranked.stream().filter(hit -> hit.score() >= floor).toList();
  }
}
