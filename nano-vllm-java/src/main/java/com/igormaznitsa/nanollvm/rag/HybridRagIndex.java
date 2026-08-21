package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.LlmModel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.function.IntConsumer;

/**
 * Fuses two or more {@link RagIndex} rankings with reciprocal rank fusion (RRF). This does not
 * concatenate corpora — mix folders, classpath files, and inline strings on one
 * {@link RagFactory#builder()} instead. The usual pair is BM25 ({@link PreparedRag}) plus dense
 * embeddings ({@link DenseRagIndex}); any {@link RagIndex} list works the same way. Nested hybrids
 * flatten to their leaves so each source ranks once. Off-topic gating requires every source to
 * agree.
 *
 * @since 1.1.0
 */
public final class HybridRagIndex implements RagIndex {

  private static final int RRF_K = 60;

  private final List<RagIndex> indexes;

  private HybridRagIndex(final List<RagIndex> indexes) {
    this.indexes = List.copyOf(indexes);
  }

  /**
   * Fuses {@code indexes} with RRF (at least two, no nulls, no duplicate instances). Nested
   * {@link HybridRagIndex} values flatten to their leaves.
   *
   * @param indexes sources to fuse; must not be {@code null}
   * @return hybrid over the flattened leaves
   * @throws NullPointerException     if {@code indexes} or an element is {@code null}
   * @throws IllegalArgumentException if fewer than two distinct sources remain
   * @since 1.2.0
   */
  public static HybridRagIndex of(final List<? extends RagIndex> indexes) {
    return new HybridRagIndex(sealedIndexes(indexes));
  }

  /**
   * {@link #of(List)} from a varargs list.
   *
   * @param indexes sources to fuse; must not be {@code null}
   * @return hybrid over the flattened leaves
   * @throws NullPointerException     if {@code indexes} or an element is {@code null}
   * @throws IllegalArgumentException if fewer than two distinct sources remain
   * @since 1.2.0
   */
  public static HybridRagIndex of(final RagIndex... indexes) {
    requireNonNull(indexes, "indexes");
    return of(List.of(indexes));
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
    requireNonNull(lexical, "lexical");
    requireNonNull(dense, "dense");
    if (lexical.size() != dense.size()) {
      throw new IllegalArgumentException(
        "lexical and dense indexes must cover the same passage count");
    }
    return of(List.of(lexical, dense));
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
   * @since 1.2.0
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
   * @since 1.2.0
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
   * @since 1.2.0
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
    return fuse(List.of(lexicalHits, denseHits), topK);
  }

  static List<RagHit> fuse(final List<List<RagHit>> rankings, final int topK) {
    requireNonNull(rankings, "rankings");
    if (rankings.isEmpty()) {
      throw new IllegalArgumentException("rankings must not be empty");
    }
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    Map<String, Acc> byId = HashMap.newHashMap(
      rankings.stream().mapToInt(List::size).sum());
    rankings.forEach(
      hits -> addRanks(byId, requireNonNull(hits, "rankings must not contain null")));
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
      byId.compute(hit.chunk().id(), (id, existing) -> existing == null
        ? new Acc(hit.chunk(), rrf)
        : new Acc(existing.chunk(), existing.score() + rrf));
    }
  }

  private static List<RagIndex> sealedIndexes(final List<? extends RagIndex> indexes) {
    requireNonNull(indexes, "indexes");
    List<RagIndex> leaves = new ArrayList<>();
    IdentityHashMap<RagIndex, Boolean> seen = new IdentityHashMap<>();
    indexes.forEach(index -> appendLeaves(requireNonNull(index, "indexes must not contain null"),
      leaves, seen));
    if (leaves.size() < 2) {
      throw new IllegalArgumentException("hybrid needs at least two indexes");
    }
    return List.copyOf(leaves);
  }

  private static void appendLeaves(
    final RagIndex index,
    final List<RagIndex> leaves,
    final IdentityHashMap<RagIndex, Boolean> seen
  ) {
    if (index instanceof HybridRagIndex hybrid) {
      hybrid.indexes.forEach(leaf -> appendLeaves(leaf, leaves, seen));
      return;
    }
    if (seen.put(index, Boolean.TRUE) != null) {
      throw new IllegalArgumentException("hybrid indexes must be distinct");
    }
    leaves.add(index);
  }

  /**
   * Flattened sources fused by this index, in registration order.
   *
   * @return unmodifiable list of at least two indexes
   * @since 1.2.0
   */
  public List<RagIndex> indexes() {
    return this.indexes;
  }

  /**
   * First BM25 source in {@link #indexes()}, when present.
   *
   * @return the BM25 half
   * @throws IllegalStateException if no {@link PreparedRag} is in the chain
   * @since 1.1.0
   */
  public PreparedRag lexical() {
    return this.firstOf(PreparedRag.class, "BM25 index");
  }

  /**
   * First dense source in {@link #indexes()}, when present.
   *
   * @return the dense half
   * @throws IllegalStateException if no {@link DenseRagIndex} is in the chain
   * @since 1.1.0
   */
  public DenseRagIndex dense() {
    return this.firstOf(DenseRagIndex.class, "dense index");
  }

  /**
   * {@inheritDoc} Same reported size when every source agrees; otherwise {@code -1}.
   */
  @Override
  public int size() {
    int first = this.indexes.getFirst().size();
    return this.indexes.stream().allMatch(index -> index.size() == first) ? first : -1;
  }

  /**
   * {@inheritDoc} Off-topic only when every source agrees.
   */
  @Override
  public boolean isOutsideCorpus(final String query) {
    return this.indexes.stream().allMatch(index -> index.isOutsideCorpus(query));
  }

  /**
   * {@inheritDoc} Reciprocal-rank fusion of every source ranking.
   */
  @Override
  public List<RagHit> retrieve(final String query, final int topK) {
    requireNonNull(query, "query");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    int pool = Math.max(topK * 4, topK);
    return fuse(this.indexes.stream()
      .map(index -> index.retrieve(query, pool))
      .toList(), topK);
  }

  /** {@inheritDoc} */
  @Override
  public String toString() {
    return "HybridRagIndex{passages=%d, sources=%d}".formatted(
      this.size(),
      this.indexes.size());
  }

  private <T extends RagIndex> T firstOf(final Class<T> type, final String role) {
    return this.indexes.stream()
      .filter(type::isInstance)
      .map(type::cast)
      .findFirst()
      .orElseThrow(() -> new IllegalStateException("hybrid has no " + role));
  }

  private record Acc(TextChunk chunk, double score) {
  }
}
