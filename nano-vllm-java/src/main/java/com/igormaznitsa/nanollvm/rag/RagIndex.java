package com.igormaznitsa.nanollvm.rag;

import java.util.List;

/**
 * Retrieves passages for a natural-language query (BM25, dense embeddings, or hybrid).
 *
 * <p>Implementations used with library sessions should be safe to share across threads.
 * {@link #retrieve} must not return {@code null}; prefer an unmodifiable list (empty when no hits).
 * Dense/hybrid indexes that call an embedding {@link com.igormaznitsa.nanollvm.models.LlmModel}
 * additionally require that model to stay open for the index lifetime.
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
