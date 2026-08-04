package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Formats retrieved passages plus the user question into one model-facing user message.
 * Compact layout puts the question first so tiny models answer instead of continuing a story.
 */
public final class RagPrompt {

  private RagPrompt() {
  }

  public static String format(List<RagHit> hits, String question) {
    return format(hits, question, Integer.MAX_VALUE, false);
  }

  public static String format(List<RagHit> hits, String question, int maxContextChars) {
    return format(hits, question, maxContextChars, false);
  }

  /**
   * @param compact when {@code true}, question first, then short passage lines (tiny models)
   */
  public static String format(
      List<RagHit> hits,
      String question,
      int maxContextChars,
      boolean compact
  ) {
    requireNonNull(hits, "hits");
    requireNonNull(question, "question");
    String q = question.strip();
    if (q.isEmpty()) {
      throw new IllegalArgumentException("question must not be blank");
    }
    if (maxContextChars < 64) {
      throw new IllegalArgumentException("maxContextChars must be >= 64");
    }

    String context = truncateContext(hits, maxContextChars, compact);
    if (context.isBlank()) {
      return q;
    }
    if (compact) {
      return q + "\n\n" + context;
    }
    return """
        Context:
        %s
        
        Question: %s
        
        Answer using only the context. Be concise.
        """.formatted(context, q).strip();
  }

  private static String truncateContext(List<RagHit> hits, int maxContextChars, boolean compact) {
    if (hits.isEmpty()) {
      return "";
    }
    List<String> parts = new ArrayList<>();
    int used = 0;
    int n = 0;
    for (RagHit hit : hits) {
      n++;
      String block = compact
          ? "- " + hit.chunk().text().strip()
          : "[%d] (%s)%n%s".formatted(n, hit.chunk().source(), hit.chunk().text().strip());
      int next = used == 0 ? block.length() : used + 2 + block.length();
      if (next > maxContextChars && used > 0) {
        break;
      }
      if (block.length() > maxContextChars && used == 0) {
        parts.add(block.substring(0, maxContextChars));
        break;
      }
      parts.add(block);
      used = next;
    }
    return String.join("\n", parts);
  }
}
