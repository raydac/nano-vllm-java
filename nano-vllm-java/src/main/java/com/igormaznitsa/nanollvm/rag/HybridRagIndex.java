package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.LlmModel;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.function.IntConsumer;

/**
 * Hybrid retrieval: BM25 ({@link PreparedRag}) plus dense embeddings ({@link DenseRagIndex}),
 * fused with reciprocal rank fusion (RRF). Off-topic gating requires both indexes to agree.
 *
 * @since 1.1.0
 */
public final class HybridRagIndex implements RagIndex {

  private static final int RRF_K = 60;

  private final PreparedRag lexical;
  private final DenseRagIndex dense;

  private HybridRagIndex(final PreparedRag lexical, final DenseRagIndex dense) {
    this.lexical = requireNonNull(lexical, "lexical");
    this.dense = requireNonNull(dense, "dense");
    if (this.lexical.size() != this.dense.size()) {
      throw new IllegalArgumentException(
        "lexical and dense indexes must cover the same passage count");
    }
  }

  /**
   * Wraps an existing BM25 index and a dense index over the same passages.
   *
   * @param lexical BM25 half; must not be {@code null}
   * @param dense   dense half over the same passage count; must not be {@code null}
   * @return hybrid index fusing both halves with RRF
   * @throws NullPointerException     if either argument is {@code null}
   * @throws IllegalArgumentException if the two indexes have different passage counts
   * @since 1.1.0
   */
  public static HybridRagIndex of(final PreparedRag lexical, final DenseRagIndex dense) {
    return new HybridRagIndex(lexical, dense);
  }

  /**
   * Builds a dense index over {@code lexical} passages on the calling thread, then wraps both
   * in hybrid retrieval.
   *
   * @param lexical         BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel  embedding encoder kept open for query-time embed; must not be {@code null}
   * @return hybrid index over the same passages
   * @throws NullPointerException     if either argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed
   * @since 1.1.0
   */
  public static HybridRagIndex of(final PreparedRag lexical, final LlmModel embeddingModel) {
    return of(lexical, DenseRagIndex.of(lexical, embeddingModel));
  }

  /**
   * {@link #of(PreparedRag, LlmModel)} and forwards {@code onPassageEmbedded} to dense indexing.
   *
   * @param lexical           BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel    embedding encoder kept open for query-time embed; must not be {@code null}
   * @param onPassageEmbedded called with {@code 1..N} after each dense vector; must not be {@code null}
   * @return hybrid index over the same passages
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed
   * @since 1.1.1
   */
  public static HybridRagIndex of(
    final PreparedRag lexical,
    final LlmModel embeddingModel,
    final IntConsumer onPassageEmbedded
  ) {
    return of(lexical, DenseRagIndex.of(lexical, embeddingModel, onPassageEmbedded));
  }

  /**
   * {@link #of(PreparedRag, LlmModel)} with dense passage embeds submitted on {@code executor}.
   * The caller owns the executor; this index does not shut it down.
   *
   * @param lexical        BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel embedding encoder kept open for query-time embed; must not be {@code null}
   * @param executor       runs each dense passage embed; must not be {@code null}; not shut down here
   * @return hybrid index over the same passages
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
   * @since 1.1.1
   */
  public static HybridRagIndex of(
    final PreparedRag lexical,
    final LlmModel embeddingModel,
    final Executor executor
  ) {
    return of(lexical,
      DenseRagIndex.of(lexical, embeddingModel, requireNonNull(executor, "executor")));
  }

  /**
   * {@link #of(PreparedRag, LlmModel, Executor)} and forwards {@code onPassageEmbedded}.
   *
   * @param lexical           BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel    embedding encoder kept open for query-time embed; must not be {@code null}
   * @param executor          runs each dense passage embed; must not be {@code null}; not shut down here
   * @param onPassageEmbedded called with {@code 1..N} after each dense vector; must not be {@code null}
   * @return hybrid index over the same passages
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
   * @since 1.1.1
   */
  public static HybridRagIndex of(
    final PreparedRag lexical,
    final LlmModel embeddingModel,
    final Executor executor,
    final IntConsumer onPassageEmbedded
  ) {
    return of(lexical, DenseRagIndex.of(lexical, embeddingModel, executor, onPassageEmbedded));
  }

  static List<RagHit> fuse(
    final List<RagHit> lexicalHits,
    final List<RagHit> denseHits,
    final int topK
  ) {
    requireNonNull(lexicalHits, "lexicalHits");
    requireNonNull(denseHits, "denseHits");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    Map<String, Acc> byId = HashMap.newHashMap(lexicalHits.size() + denseHits.size());
    addRanks(byId, lexicalHits);
    addRanks(byId, denseHits);
    return byId.values().stream()
      .sorted(Comparator
        .comparingDouble(Acc::score).reversed()
        .thenComparingInt(acc -> acc.chunk().text().length()))
      .limit(topK)
      .map(acc -> new RagHit(acc.chunk(), acc.score()))
      .toList();
  }

  private static void addRanks(final Map<String, Acc> byId, final List<RagHit> hits) {
    for (int rank = 0; rank < hits.size(); rank++) {
      RagHit hit = hits.get(rank);
      double rrf = 1.0 / (RRF_K + rank + 1);
      byId.compute(hit.chunk().id(), (id, existing) -> {
        if (existing == null) {
          return new Acc(hit.chunk(), rrf);
        }
        return new Acc(existing.chunk(), existing.score() + rrf);
      });
    }
  }

  /**
   * BM25 half of this hybrid index.
   *
   * @since 1.1.0
   */
  public PreparedRag lexical() {
    return this.lexical;
  }

  /**
   * Dense half of this hybrid index.
   *
   * @since 1.1.0
   */
  public DenseRagIndex dense() {
    return this.dense;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int size() {
    return this.lexical.size();
  }

  /**
   * {@inheritDoc} Off-topic only when <em>both</em> BM25 and dense agree.
   */
  @Override
  public boolean isOutsideCorpus(final String query) {
    return this.lexical.isOutsideCorpus(query) && this.dense.isOutsideCorpus(query);
  }

  /** {@inheritDoc} Reciprocal-rank fusion of BM25 and dense rankings. */
  @Override
  public List<RagHit> retrieve(final String query, final int topK) {
    requireNonNull(query, "query");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    int pool = Math.max(topK * 4, topK);
    return fuse(this.lexical.retrieve(query, pool), this.dense.retrieve(query, pool), topK);
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return "HybridRagIndex{passages=%d, dense=%s}".formatted(
      this.size(),
      this.dense);
  }

  private record Acc(TextChunk chunk, double score) {
  }
}
