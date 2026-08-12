package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

/**
 * One retrieval hit: scored {@link TextChunk}.
 */
public record RagHit(TextChunk chunk, double score) {

  public RagHit {
    requireNonNull(chunk, "chunk");
  }
}
