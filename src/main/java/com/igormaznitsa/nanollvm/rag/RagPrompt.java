package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Formats retrieved passages plus the user question into one model-facing user message.
 * Compact layout puts the question first so tiny models answer instead of continuing a story.
 * When passages are present, the prompt asks for a short answer from those lines; when none
 * are present, it forbids invention (tiny models otherwise hallucinate freely).
 */
public final class RagPrompt {

  private RagPrompt() {
  }

  public static String format(final List<RagHit> hits, final String question) {
    return format(hits, question, Integer.MAX_VALUE, false);
  }

  public static String format(final List<RagHit> hits, final String question,
                              final int maxContextChars) {
    return format(hits, question, maxContextChars, false);
  }

  /**
   * @param compact when {@code true}, question first, then short passage lines (tiny models)
   */
  public static String format(
      final List<RagHit> hits,
      final String question,
      final int maxContextChars,
      final boolean compact
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
      return noHitPrompt(q, compact);
    }
    if (compact) {
      // Positive instruction last — tiny models latch onto "say you do not know" if it leads
      return q + """


        Context:
        """ + context + """

        Answer in one short sentence using names and facts from the Context above.
        """;
    }
    return """
        Context:
        %s

        Question: %s

      Answer using only the context. Do not invent names, dates, or other details.
      If the context does not contain the answer, say you do not know. Be concise.
      """.formatted(context, q).strip();
  }

  private static String noHitPrompt(final String question, final boolean compact) {
    if (compact) {
      return question + """


        No context documents were found for this question.
        Reply with exactly: I do not know.
        Do not invent places, names, or stories.
        """;
    }
    return """
      Question: %s

      No context documents were retrieved. Say you do not know. Do not invent facts.
      """.formatted(question).strip();
  }

  private static String truncateContext(final List<RagHit> hits, final int maxContextChars,
                                        final boolean compact) {
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
