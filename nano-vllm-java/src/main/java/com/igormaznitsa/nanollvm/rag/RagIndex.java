package com.igormaznitsa.nanollvm.rag;

import java.util.List;

/**
 * Retrieves passages for a natural-language query (BM25, dense embeddings, or a fused
 * {@link HybridRagIndex} of any {@link RagIndex} list).
 *
 * <p>Implementations used with library sessions should be safe to share across threads.
 * {@link #retrieve} must not return {@code null}; prefer an unmodifiable list (empty when no hits).
 * Dense/hybrid indexes that call an embedding {@link com.igormaznitsa.nanollvm.models.LlmModel}
 * additionally require that model to stay open for the index lifetime.
 */
public interface RagIndex {

  /**
   * Top passages for {@code query}, highest score first. Never {@code null}.
   *
   * @param query natural-language search string; must not be {@code null}
   * @param topK  maximum hits to return; implementations may return fewer
   * @return unmodifiable list, empty when nothing matches
   */
  List<RagHit> retrieve(final String query, final int topK);

  /**
   * When {@code true}, the query should skip retrieval (mostly OOV / off-topic for this index).
   *
   * @param query natural-language search string
   * @return {@code true} to skip rewrite, retrieve, and RAG prompts
   */
  default boolean isOutsideCorpus(final String query) {
    return false;
  }

  /**
   * Indexed passage count, or {@code -1} when the size is unknown (still use the RAG path).
   *
   * @return {@code 0} for an empty corpus, {@code -1} if the implementation does not report size
   */
  default int size() {
    return -1;
  }
}
