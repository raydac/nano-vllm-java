package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.Map;

/**
 * One document passage after {@link PassagePreparser}: model-facing text plus pre-tokenized
 * search terms (built once at load, shared across queries and LLMs).
 */
public record PreparedPassage(
    TextChunk chunk,
    String searchText,
    Map<String, Integer> termFreqs,
    int tokenCount
) {

  public PreparedPassage {
    requireNonNull(chunk, "chunk");
    requireNonNull(searchText, "searchText");
    termFreqs = Map.copyOf(requireNonNull(termFreqs, "termFreqs"));
    if (tokenCount < 0) {
      throw new IllegalArgumentException("tokenCount must be >= 0");
    }
  }

  public String id() {
    return this.chunk.id();
  }

  public String source() {
    return this.chunk.source();
  }

  /**
   * Text shown to the model in the RAG prompt.
   */
  public String modelText() {
    return this.chunk.text();
  }
}
