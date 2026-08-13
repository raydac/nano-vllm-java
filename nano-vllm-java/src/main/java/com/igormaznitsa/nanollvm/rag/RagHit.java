package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

/**
 * One retrieval hit: a {@link TextChunk} plus a relevance {@link #score()}.
 *
 * <p>{@link RagIndex#retrieve} returns hits highest-score first. Score meaning depends on the
 * index: BM25 ({@link PreparedRag}), cosine/dot after L2 ({@link DenseRagIndex}), or reciprocal
 * rank fusion ({@link HybridRagIndex}). Scores are not comparable across index types. Immutable;
 * safe to share.
 *
 * <pre>{@code
 * for (RagHit hit : index.retrieve(question, 3)) {
 *   System.out.println(hit.chunk().source() + "  " + hit.score());
 *   System.out.println(hit.chunk().text());
 * }
 * }</pre>
 *
 * @param chunk passage that matched; never {@code null}
 * @param score relevance from the index that produced this hit (higher is better)
 */
public record RagHit(TextChunk chunk, double score) {

  public RagHit {
    requireNonNull(chunk, "chunk");
  }
}
