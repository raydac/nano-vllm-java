package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.LlmModel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

/**
 * Dense passage index: chunk vectors from an embedding {@link LlmModel} (e.g. BERT-family GGUF),
 * query-time cosine (dot product on L2-normalized vectors).
 *
 * <p>Does not own {@code encoder} — close the model after this index is unused. Index-time vectors
 * are immutable. Passage embedding at {@link #of} is sequential unless the caller supplies an
 * {@link Executor}. Query embeds are concurrent (each encoder call uses a fresh step context). Prefer
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
   * Embeds every passage in {@code lexical} with {@code embeddingModel} on the calling thread.
   *
   * @param lexical         BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel  embedding encoder kept open for query-time embed; must not be {@code null}
   * @return dense index over {@code lexical} passages
   * @throws NullPointerException     if {@code lexical} or {@code embeddingModel} is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed
   * @since 1.1.0
   */
  public static DenseRagIndex of(final PreparedRag lexical, final LlmModel embeddingModel) {
    requireNonNull(lexical, "lexical");
    return of(lexical.chunks(), embeddingModel);
  }

  /**
   * {@link #of(PreparedRag, LlmModel)} and invokes {@code onPassageEmbedded} after each vector
   * with the 1-based completed count on the calling thread.
   *
   * @param lexical             BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel      embedding encoder kept open for query-time embed; must not be {@code null}
   * @param onPassageEmbedded   called with {@code 1..N} after each passage vector; must not be {@code null}
   * @return dense index over {@code lexical} passages
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed
   * @since 1.1.1
   */
  public static DenseRagIndex of(
    final PreparedRag lexical,
    final LlmModel embeddingModel,
    final IntConsumer onPassageEmbedded
  ) {
    requireNonNull(lexical, "lexical");
    return of(lexical.chunks(), embeddingModel, onPassageEmbedded);
  }

  /**
   * {@link #of(PreparedRag, LlmModel)} with one BERT forward submitted per passage on
   * {@code executor}. The caller owns the executor; this index does not shut it down.
   *
   * @param lexical         BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel  embedding encoder kept open for query-time embed; must not be {@code null}
   * @param executor        runs each passage embed; must not be {@code null}; not shut down here
   * @return dense index over {@code lexical} passages
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
   * @since 1.1.1
   */
  public static DenseRagIndex of(
    final PreparedRag lexical,
    final LlmModel embeddingModel,
    final Executor executor
  ) {
    requireNonNull(lexical, "lexical");
    return of(lexical.chunks(), embeddingModel, executor);
  }

  /**
   * {@link #of(PreparedRag, LlmModel, Executor)} and invokes {@code onPassageEmbedded} after each
   * vector with the 1-based completed count (may run on executor threads).
   *
   * @param lexical           BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel    embedding encoder kept open for query-time embed; must not be {@code null}
   * @param executor          runs each passage embed; must not be {@code null}; not shut down here
   * @param onPassageEmbedded called with {@code 1..N} after each passage vector; must not be {@code null}
   * @return dense index over {@code lexical} passages
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
   * @since 1.1.1
   */
  public static DenseRagIndex of(
    final PreparedRag lexical,
    final LlmModel embeddingModel,
    final Executor executor,
    final IntConsumer onPassageEmbedded
  ) {
    requireNonNull(lexical, "lexical");
    return of(lexical.chunks(), embeddingModel, executor, onPassageEmbedded);
  }

  /**
   * Embeds each chunk text with {@code embeddingModel} on the calling thread.
   *
   * @param chunks          passages to index; must not be empty
   * @param embeddingModel  embedding encoder kept open for query-time embed; must not be {@code null}
   * @return dense index over {@code chunks}
   * @throws NullPointerException     if {@code chunks} or {@code embeddingModel} is {@code null}
   * @throws IllegalArgumentException if {@code chunks} is empty or {@code embeddingModel} is not an
   *                                  embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed
   * @since 1.1.0
   */
  public static DenseRagIndex of(final List<TextChunk> chunks, final LlmModel embeddingModel) {
    return indexChunks(chunks, embeddingModel, DenseRagIndex::ignorePassageProgress, null);
  }

  /**
   * {@link #of(List, LlmModel)} and invokes {@code onPassageEmbedded} after each vector with the
   * 1-based completed count on the calling thread. Prefer a method reference or an
   * {@code IntConsumer} variable; a raw lambda as the third argument can be ambiguous with the
   * {@link Executor} overload.
   *
   * @param chunks            passages to index; must not be empty
   * @param embeddingModel    embedding encoder kept open for query-time embed; must not be {@code null}
   * @param onPassageEmbedded called with {@code 1..N} after each passage vector; must not be {@code null}
   * @return dense index over {@code chunks}
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code chunks} is empty or {@code embeddingModel} is not an
   *                                  embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed
   * @since 1.1.1
   */
  public static DenseRagIndex of(
    final List<TextChunk> chunks,
    final LlmModel embeddingModel,
    final IntConsumer onPassageEmbedded
  ) {
    return indexChunks(chunks, embeddingModel, onPassageEmbedded, null);
  }

  /**
   * {@link #of(List, LlmModel)} with one BERT forward submitted per passage on {@code executor}.
   * The caller owns the executor; this index does not shut it down.
   *
   * @param chunks         passages to index; must not be empty
   * @param embeddingModel embedding encoder kept open for query-time embed; must not be {@code null}
   * @param executor       runs each passage embed; must not be {@code null}; not shut down here
   * @return dense index over {@code chunks}
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code chunks} is empty or {@code embeddingModel} is not an
   *                                  embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
   * @since 1.1.1
   */
  public static DenseRagIndex of(
    final List<TextChunk> chunks,
    final LlmModel embeddingModel,
    final Executor executor
  ) {
    return indexChunks(
      chunks,
      embeddingModel,
      DenseRagIndex::ignorePassageProgress,
      requireNonNull(executor, "executor"));
  }

  /**
   * {@link #of(List, LlmModel, Executor)} and invokes {@code onPassageEmbedded} after each vector
   * with the 1-based completed count (may run on executor threads).
   *
   * @param chunks            passages to index; must not be empty
   * @param embeddingModel    embedding encoder kept open for query-time embed; must not be {@code null}
   * @param executor          runs each passage embed; must not be {@code null}; not shut down here
   * @param onPassageEmbedded called with {@code 1..N} after each passage vector; must not be {@code null}
   * @return dense index over {@code chunks}
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code chunks} is empty or {@code embeddingModel} is not an
   *                                  embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
   * @since 1.1.1
   */
  public static DenseRagIndex of(
    final List<TextChunk> chunks,
    final LlmModel embeddingModel,
    final Executor executor,
    final IntConsumer onPassageEmbedded
  ) {
    return indexChunks(
      chunks,
      embeddingModel,
      onPassageEmbedded,
      requireNonNull(executor, "executor"));
  }

  private static void ignorePassageProgress(final int ignored) {
  }

  private static DenseRagIndex indexChunks(
    final List<TextChunk> chunks,
    final LlmModel embeddingModel,
    final IntConsumer onPassageEmbedded,
    final Executor executor
  ) {
    requireNonNull(chunks, "chunks");
    requireNonNull(onPassageEmbedded, "onPassageEmbedded");
    requireEmbeddingModel(embeddingModel);
    if (chunks.isEmpty()) {
      throw new IllegalArgumentException("chunks must not be empty");
    }
    List<TextChunk> sealed = List.copyOf(chunks);
    float[][] vectors = executor == null
      ? embedPassagesOnCaller(sealed, embeddingModel, onPassageEmbedded)
      : embedPassagesInParallel(sealed, embeddingModel, onPassageEmbedded, executor);
    return new DenseRagIndex(sealed, vectors, embeddingModel);
  }

  private static float[][] embedPassagesOnCaller(
    final List<TextChunk> chunks,
    final LlmModel embeddingModel,
    final IntConsumer onPassageEmbedded
  ) {
    float[][] vectors = new float[chunks.size()][];
    for (int i = 0; i < chunks.size(); i++) {
      vectors[i] = embeddingModel.embed(embedText(chunks.get(i)));
      onPassageEmbedded.accept(i + 1);
    }
    return vectors;
  }

  private static float[][] embedPassagesInParallel(
    final List<TextChunk> chunks,
    final LlmModel embeddingModel,
    final IntConsumer onPassageEmbedded,
    final Executor executor
  ) {
    float[][] vectors = new float[chunks.size()][];
    AtomicInteger completed = new AtomicInteger();
    CompletableFuture<?>[] jobs = new CompletableFuture<?>[chunks.size()];
    for (int i = 0; i < chunks.size(); i++) {
      int index = i;
      jobs[i] = CompletableFuture.runAsync(() -> {
        vectors[index] = embeddingModel.embed(embedText(chunks.get(index)));
        onPassageEmbedded.accept(completed.incrementAndGet());
      }, executor);
    }
    awaitEmbeds(jobs);
    return vectors;
  }

  private static void awaitEmbeds(final CompletableFuture<?>[] jobs) {
    try {
      CompletableFuture.allOf(jobs).get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      cancelEmbeds(jobs);
      throw new IllegalStateException("embedding interrupted", e);
    } catch (ExecutionException e) {
      cancelEmbeds(jobs);
      Throwable cause = e.getCause();
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IllegalStateException("embedding failed", cause);
    }
  }

  private static void cancelEmbeds(final CompletableFuture<?>[] jobs) {
    for (CompletableFuture<?> job : jobs) {
      job.cancel(true);
    }
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
