package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

/**
 * One indexed text passage with a stable id and optional source label.
 *
 * <p>{@link #id()} must be unique within a corpus (chunker uses {@code file#n} style ids).
 * {@link #source()} is a path, classpath URI, or tag shown in RAG citations; {@link #of} copies
 * {@code id} into {@code source} when you have no separate label. {@link #text()} is the model-
 * facing chunk body (null becomes {@code ""}). Immutable; safe to share.
 *
 * @param id     non-blank stable identifier; never {@code null}
 * @param source origin label (file path, {@code classpath:…}, tag); never {@code null}
 * @param text   passage body fed to BM25 / embeddings / the generator; never {@code null}
 */
public record TextChunk(String id, String source, String text) {

  public TextChunk {
    requireNonNull(id, "id");
    requireNonNull(source, "source");
    text = text == null ? "" : text;
    if (id.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
  }

  /**
   * Chunk whose source label is the same as {@code id}.
   *
   * @param id   non-blank identifier
   * @param text passage body; {@code null} becomes {@code ""}
   */
  public static TextChunk of(final String id, final String text) {
    return new TextChunk(id, id, text);
  }

  /**
   * {@code true} when {@link #text()} is empty or whitespace-only.
   */
  public boolean isBlank() {
    return this.text.isBlank();
  }
}
