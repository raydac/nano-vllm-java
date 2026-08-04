package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * Structural retrieval-query rules: when to keep the previous user turn as an anchor.
 * Uses token counts only — not reply dictionaries or language-specific words.
 */
final class RagRetrieval {

  /**
   * Follow-ups shorter than this reuse {@code anchor + question} for BM25.
   */
  static final int EXPAND_BELOW_TOKENS = 6;

  /**
   * Only turns at least this long replace the retrieval anchor.
   */
  static final int ANCHOR_MIN_TOKENS = 5;

  private RagRetrieval() {
  }

  static String retrievalQuery(String question, String anchor) {
    requireNonNull(question, "question");
    if (anchor == null || anchor.isBlank() || !needsAnchor(question)) {
      return question;
    }
    return anchor + '\n' + question;
  }

  static boolean needsAnchor(String question) {
    return Bm25Index.tokenize(question).size() < EXPAND_BELOW_TOKENS;
  }

  static boolean shouldUpdateAnchor(String question) {
    return Bm25Index.tokenize(question).size() >= ANCHOR_MIN_TOKENS;
  }

  /**
   * For short follow-ups, prefer hits from the previous source when their score stays
   * competitive — keeps anaphora on the same document instead of a larger dominating file.
   */
  static List<RagHit> preferPriorSource(
      List<RagHit> candidates,
      String priorSource,
      int topK
  ) {
    requireNonNull(candidates, "candidates");
    if (candidates.isEmpty()) {
      return List.of();
    }
    if (priorSource == null || priorSource.isBlank()) {
      return clip(candidates, topK);
    }
    List<RagHit> same = candidates.stream()
        .filter(hit -> priorSource.equals(hit.chunk().source()))
        .toList();
    if (same.isEmpty()) {
      return clip(candidates, topK);
    }
    double best = candidates.getFirst().score();
    if (same.getFirst().score() >= best * 0.55) {
      return clip(same, topK);
    }
    return clip(candidates, topK);
  }

  private static List<RagHit> clip(List<RagHit> hits, int topK) {
    if (hits.size() <= topK) {
      return List.copyOf(hits);
    }
    return List.copyOf(hits.subList(0, topK));
  }
}
