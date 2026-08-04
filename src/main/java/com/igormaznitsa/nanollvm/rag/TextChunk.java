package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

/**
 * One indexed text passage with a stable id and optional source label (file path, tag, …).
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

  public static TextChunk of(String id, String text) {
    return new TextChunk(id, id, text);
  }

  public boolean isBlank() {
    return this.text.isBlank();
  }
}
