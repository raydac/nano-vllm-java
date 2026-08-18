package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.LlmModel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Dense passage index: chunk vectors from an embedding {@link LlmModel} (e.g. BERT-family GGUF),
 * query-time cosine (dot product on L2-normalized vectors).
 *
 * <p>Does not own {@code encoder} — close the model after this index is unused. Index-time vectors
 * are immutable; query embeds are concurrent (each encoder call uses a fresh step context). Prefer
 * {@link HybridRagIndex} or {@link RagFactory#withEmbeddings} when a BM25 corpus is also available.
 *
 * @since 1.1.0
 */
public final class DenseRagIndex implements RagIndex {

  private static final double HIT_FLOOR_RATIO = 0.45;
  private static final double OUTSIDE_MAX_SIMILARITY = 0.28;

  private final List<TextChunk> chunks;
  private final float[][] vectors;
  private final LlmModel encoder;
  private final int dimensions;

  private DenseRagIndex(
    final List<TextChunk> chunks,
    final float[][] vectors,
    final LlmModel encoder
  ) {
    this.chunks = List.copyOf(requireNonNull(chunks, "chunks"));
    this.vectors = requireNonNull(vectors, "vectors");
    this.encoder = requireNonNull(encoder, "encoder");
    if (this.chunks.isEmpty()) {
      throw new IllegalArgumentException("chunks must not be empty");
    }
    if (this.vectors.length != this.chunks.size()) {
      throw new IllegalArgumentException("vectors length must match chunks");
    }
    this.dimensions = this.vectors[0].length;
    for (int i = 0; i < this.vectors.length; i++) {
      if (this.vectors[i] == null || this.vectors[i].length != this.dimensions) {
        throw new IllegalArgumentException("vector[" + i + "] must have length " + this.dimensions);
      }
    }
  }

  /**
   * Embeds every passage in {@code lexical} with {@code embeddingModel}.
   *
   * @since 1.1.0
   */
  public static DenseRagIndex of(final PreparedRag lexical, final LlmModel embeddingModel) {
    requireNonNull(lexical, "lexical");
    return of(lexical.chunks(), embeddingModel);
  }

  /**
   * Embeds each chunk text with {@code embeddingModel} (must be an embedding encoder).
   *
   * @since 1.1.0
   */
  public static DenseRagIndex of(final List<TextChunk> chunks, final LlmModel embeddingModel) {
    requireNonNull(chunks, "chunks");
    requireEmbeddingModel(embeddingModel);
    if (chunks.isEmpty()) {
      throw new IllegalArgumentException("chunks must not be empty");
    }
    List<TextChunk> sealed = List.copyOf(chunks);
    List<String> texts = sealed.stream()
      .map(DenseRagIndex::embedText)
      .toList();
    float[][] vectors = embeddingModel.embed(texts);
    return new DenseRagIndex(sealed, vectors, embeddingModel);
  }

  private static String embedText(final TextChunk chunk) {
    String text = chunk.text();
    return text.isBlank() ? chunk.id() : text;
  }

  private static void requireEmbeddingModel(final LlmModel model) {
    requireNonNull(model, "embeddingModel");
    if (!model.isEmbeddingModel()) {
      throw new IllegalArgumentException(
        "embeddingModel must be an embedding encoder (got " + model.architectureName() + ")");
    }
    if (model.isClosed()) {
      throw new IllegalStateException("embeddingModel is closed");
    }
  }

  private static List<RagHit> keepStrongHits(final List<RagHit> scored, final int topK) {
    if (scored.isEmpty()) {
      return List.of();
    }
    double floor = scored.getFirst().score() * HIT_FLOOR_RATIO;
    return scored.stream()
      .filter(hit -> hit.score() >= floor)
      .limit(topK)
      .toList();
  }

  private static double dot(final float[] a, final float[] b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      sum += (double) a[i] * b[i];
    }
    return sum;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int size() {
    return this.chunks.size();
  }

  /**
   * Embedding vector length of every stored passage.
   *
   * @since 1.1.0
   */
  public int dimensions() {
    return this.dimensions;
  }

  /**
   * Embedding model used at index time (not owned; close it after this index is unused).
   *
   * @since 1.1.0
   */
  public LlmModel encoder() {
    return this.encoder;
  }

  /** {@inheritDoc} Off-topic when the best cosine similarity is below a dense floor. */
  @Override
  public boolean isOutsideCorpus(final String query) {
    requireNonNull(query, "query");
    if (query.isBlank()) {
      return true;
    }
    return this.bestSimilarity(query) < OUTSIDE_MAX_SIMILARITY;
  }

  /** {@inheritDoc} Cosine (dot of L2-normalized embeddings), then a relative score floor. */
  @Override
  public List<RagHit> retrieve(final String query, final int topK) {
    requireNonNull(query, "query");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    if (query.isBlank()) {
      return List.of();
    }

    float[] queryVector = this.embedQuery(query);
    List<RagHit> scored = new ArrayList<>(this.chunks.size());
    for (int i = 0; i < this.chunks.size(); i++) {
      double score = dot(queryVector, this.vectors[i]);
      if (score > 0.0) {
        scored.add(new RagHit(this.chunks.get(i), score));
      }
    }
    scored.sort(Comparator
      .comparingDouble(RagHit::score).reversed()
      .thenComparingInt(hit -> hit.chunk().text().length()));
    return List.copyOf(keepStrongHits(scored, topK));
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return "DenseRagIndex{passages=%d, dim=%d, encoder=%s}".formatted(
      this.size(),
      this.dimensions,
      this.encoder.architectureName());
  }

  private double bestSimilarity(final String query) {
    float[] queryVector = this.embedQuery(query);
    double best = Double.NEGATIVE_INFINITY;
    for (float[] vector : this.vectors) {
      best = Math.max(best, dot(queryVector, vector));
    }
    return best;
  }

  private float[] embedQuery(final String query) {
    return this.encoder.embed(query.isBlank() ? "_" : query);
  }
}
