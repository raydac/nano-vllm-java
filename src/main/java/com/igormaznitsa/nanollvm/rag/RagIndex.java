package com.igormaznitsa.nanollvm.rag;

import java.util.List;

/**
 * Retrieves passages for a natural-language query (BM25 or other text indexes).
 */
public interface RagIndex {

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
