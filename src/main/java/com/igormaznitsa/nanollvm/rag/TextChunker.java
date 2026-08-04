package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits raw text into {@link TextChunk}s for indexing.
 * Prefers section-aware sentence passages from {@link TextPreprocessor}.
 */
final class TextChunker {

  private TextChunker() {
  }

  static List<TextChunk> split(
      String baseId,
      String source,
      String text,
      int maxChunkChars,
      int overlap,
      boolean preprocess,
      boolean atomicSentences
  ) {
    requireNonNull(baseId, "baseId");
    requireNonNull(source, "source");
    if (text == null || text.isBlank()) {
      return List.of();
    }
    if (preprocess) {
      List<String> units = TextPreprocessor.passages(text);
      if (!units.isEmpty()) {
        return atomicSentences
            ? atomicSplit(baseId, source, units, maxChunkChars)
            : packUnits(baseId, source, units, maxChunkChars);
      }
    }
    return windowSplit(baseId, source, text.strip(), maxChunkChars, overlap);
  }

  private static List<TextChunk> atomicSplit(
      String baseId,
      String source,
      List<String> units,
      int maxChunkChars
  ) {
    List<TextChunk> chunks = new ArrayList<>();
    int part = 0;
    for (String unit : units) {
      if (unit.length() <= maxChunkChars) {
        part++;
        chunks.add(new TextChunk(baseId + "#" + part, source, unit));
        continue;
      }
      List<TextChunk> windows =
          windowSplit(baseId + "#" + (part + 1), source, unit, maxChunkChars, 0);
      for (TextChunk window : windows) {
        part++;
        chunks.add(new TextChunk(baseId + "#" + part, source, window.text()));
      }
    }
    if (chunks.size() == 1) {
      return List.of(new TextChunk(baseId, source, chunks.getFirst().text()));
    }
    return List.copyOf(chunks);
  }

  private static List<TextChunk> packUnits(
      String baseId,
      String source,
      List<String> units,
      int maxChunkChars
  ) {
    List<TextChunk> chunks = new ArrayList<>();
    StringBuilder buf = new StringBuilder();
    for (String unit : units) {
      if (unit.length() > maxChunkChars) {
        emitPacked(buf, baseId, source, chunks);
        chunks.addAll(
            windowSplit(baseId + "#w" + (chunks.size() + 1), source, unit, maxChunkChars, 40));
        continue;
      }
      if (buf.isEmpty()) {
        buf.append(unit);
        continue;
      }
      if (buf.length() + 1 + unit.length() <= maxChunkChars) {
        buf.append(' ').append(unit);
        continue;
      }
      emitPacked(buf, baseId, source, chunks);
      buf.append(unit);
    }
    emitPacked(buf, baseId, source, chunks);
    if (chunks.size() == 1) {
      TextChunk only = chunks.getFirst();
      return List.of(new TextChunk(baseId, only.source(), only.text()));
    }
    return List.copyOf(chunks);
  }

  private static void emitPacked(
      StringBuilder buf,
      String baseId,
      String source,
      List<TextChunk> chunks
  ) {
    if (buf.isEmpty()) {
      return;
    }
    int part = chunks.size() + 1;
    chunks.add(new TextChunk(baseId + "#" + part, source, buf.toString()));
    buf.setLength(0);
  }

  private static List<TextChunk> windowSplit(
      String baseId,
      String source,
      String body,
      int maxChunkChars,
      int overlap
  ) {
    if (body.length() <= maxChunkChars) {
      return List.of(new TextChunk(baseId, source, body));
    }
    int step = Math.max(1, maxChunkChars - Math.min(overlap, maxChunkChars - 1));
    List<TextChunk> chunks = new ArrayList<>();
    int start = 0;
    int part = 0;
    while (start < body.length()) {
      int end = Math.min(body.length(), start + maxChunkChars);
      if (end < body.length()) {
        end = preferBreak(body, start, end);
      }
      String slice = body.substring(start, end).strip();
      if (!slice.isEmpty()) {
        part++;
        String id = part == 1 && end >= body.length() ? baseId : baseId + "#" + part;
        chunks.add(new TextChunk(id, source, slice));
      }
      if (end >= body.length()) {
        break;
      }
      start = Math.max(start + 1, end - (maxChunkChars - step));
      start = Math.min(start, end);
    }
    return List.copyOf(chunks);
  }

  private static int preferBreak(String body, int start, int end) {
    int windowStart = Math.max(start + (end - start) / 2, start);
    int nl = body.lastIndexOf('\n', end - 1);
    if (nl >= windowStart) {
      return nl + 1;
    }
    int space = body.lastIndexOf(' ', end - 1);
    if (space >= windowStart) {
      return space + 1;
    }
    return end;
  }
}
