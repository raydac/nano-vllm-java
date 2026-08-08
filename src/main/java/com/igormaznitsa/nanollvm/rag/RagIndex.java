package com.igormaznitsa.nanollvm.rag;

import java.util.List;

/**
 * Retrieves passages for a natural-language query (BM25 or other text indexes).
 *
 * <p>Implementations used with library sessions should be safe to share across threads.
 * {@link #retrieve} must not return {@code null}; prefer an unmodifiable list (empty when no hits).
 */
public interface RagIndex {

  /**
   * Top passages for {@code query}, highest score first. Never {@code null}.
   */
  List<RagHit> retrieve(final String query, final int topK);

  /**
   * When {@code true}, the query should skip retrieval (mostly OOV / off-topic for this index).
   */
  default boolean isOutsideCorpus(final String query) {
    return false;
  }

  default int size() {
    return -1;
  }
}
