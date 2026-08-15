package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.LlmModel;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
   * @since 1.1.0
   */
  public static HybridRagIndex of(final PreparedRag lexical, final DenseRagIndex dense) {
    return new HybridRagIndex(lexical, dense);
  }

  /**
   * Builds a dense index over {@code lexical} passages, then wraps both in hybrid retrieval.
   *
   * @since 1.1.0
   */
  public static HybridRagIndex of(final PreparedRag lexical, final LlmModel embeddingModel) {
    return of(lexical, DenseRagIndex.of(lexical, embeddingModel));
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

  @Override
  public int size() {
    return this.lexical.size();
  }

  @Override
  public boolean isOutsideCorpus(final String query) {
    return this.lexical.isOutsideCorpus(query) && this.dense.isOutsideCorpus(query);
  }

  @Override
  public List<RagHit> retrieve(final String query, final int topK) {
    requireNonNull(query, "query");
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    int pool = Math.max(topK * 4, topK);
    return fuse(this.lexical.retrieve(query, pool), this.dense.retrieve(query, pool), topK);
  }

  @Override
  public String toString() {
    return "HybridRagIndex{passages=%d, dense=%s}".formatted(
      this.size(),
      this.dense);
  }

  private record Acc(TextChunk chunk, double score) {
  }
}
