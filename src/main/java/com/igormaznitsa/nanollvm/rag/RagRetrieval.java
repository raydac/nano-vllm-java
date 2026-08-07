package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.Comparator;
import java.util.List;

/**
 * Structural retrieval-query rules: when short follow-ups need rewrite / anchor
 * expansion, prior-source continuity, and compact-passage preference for the prompt.
 * Uses token counts and passage length only — no corpus-specific filenames or topics.
 */
final class RagRetrieval {

  /**
   * Follow-ups shorter than this need LLM rewrite (or, without an LLM, {@code anchor + question}).
   */
  static final int EXPAND_BELOW_TOKENS = 6;

  private static final double PRIOR_SOURCE_COMPETITIVE = 0.55;
  private static final double COMPACT_COMPETITIVE = 0.55;

  private RagRetrieval() {
  }

  static String retrievalQuery(final String question, final String anchor) {
    requireNonNull(question, "question");
    if (anchor == null || anchor.isBlank() || !needsAnchor(question)) {
      return question;
    }
    return anchor + '\n' + question;
  }

  static boolean needsAnchor(final String question) {
    return Bm25Index.tokenize(question).size() < EXPAND_BELOW_TOKENS;
  }

  /**
   * Short follow-ups need linguistic rewrite when prior context exists (LLM), or structural
   * anchor concat (no LLM). First short turns with no prior use the raw question.
   */
  static boolean needsRewrite(final String question) {
    return needsAnchor(question);
  }

  /**
   * Standalone (non-short) user turns may replace the retrieval prior with the raw question.
   * Short turns update the prior from rewrite keywords in {@link RagSession} instead.
   */
  static boolean shouldUpdateAnchor(final String question) {
    return !needsAnchor(question);
  }

  /**
   * Among score-competitive hits, prefer shorter passages for the prompt.
   * Dense notes beat long chapters on any corpus without naming conventions.
   */
  static List<RagHit> preferCompactPassages(final List<RagHit> candidates, final int topK) {
    requireNonNull(candidates, "candidates");
    if (candidates.isEmpty()) {
      return List.of();
    }
    RagHit bestHit = candidates.getFirst();
    double floor = bestHit.score() * COMPACT_COMPETITIVE;
    int bestLen = bestHit.chunk().text().length();

    List<RagHit> compact = candidates.stream()
      .filter(hit -> hit.score() >= floor)
      .filter(hit -> hit.chunk().text().length() * 2 <= bestLen)
      .sorted(Comparator
        .comparingDouble(RagHit::score).reversed()
        .thenComparingInt(hit -> hit.chunk().text().length()))
      .toList();

    if (!compact.isEmpty()) {
      return clip(compact, topK);
    }
    return clip(candidates, topK);
  }

  /**
   * For short follow-ups, prefer hits from the previous source when their score stays
   * competitive — keeps anaphora on the same document instead of a larger dominating file.
   */
  static List<RagHit> preferPriorSource(
      final List<RagHit> candidates,
      final String priorSource,
      final int topK
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
    if (same.getFirst().score() >= best * PRIOR_SOURCE_COMPETITIVE) {
      return clip(same, topK);
    }
    return clip(candidates, topK);
  }

  private static List<RagHit> clip(final List<RagHit> hits, final int topK) {
    if (hits.size() <= topK) {
      return List.copyOf(hits);
    }
    return List.copyOf(hits.subList(0, topK));
  }
}
