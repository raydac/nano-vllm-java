package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

/**
 * One document after extract and {@link RagTuner#preprocessRagText(String)}, before it is cut
 * into {@link TextChunk}s.
 *
 * <p>{@link #id()} is the stable base the built-in cutter uses ({@code text-1}, a caller id, or
 * the file / classpath source). {@link #source()} is the citation label. {@link #text()} is the
 * body the splitter sees, already rewritten by tuners.
 *
 * @param id     non-blank document id; never {@code null}
 * @param source citation label; never {@code null}
 * @param text   body to cut; never {@code null} ({@code null} becomes empty)
 * @since 1.5.0
 */
public record RagDocument(String id, String source, String text) {

  public RagDocument {
    requireNonNull(id, "id");
    requireNonNull(source, "source");
    text = text == null ? "" : text;
    if (id.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
  }
}
