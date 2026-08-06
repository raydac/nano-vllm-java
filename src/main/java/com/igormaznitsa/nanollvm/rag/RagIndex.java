package com.igormaznitsa.nanollvm.rag;

import java.util.List;

/**
 * Retrieves passages for a natural-language query (BM25 or other text indexes).
 */
public interface RagIndex {

  List<RagHit> retrieve(final String query, final int topK);

  default int size() {
    return -1;
  }
}
