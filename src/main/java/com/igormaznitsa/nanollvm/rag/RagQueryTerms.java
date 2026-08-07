package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Lexical query-term selection for BM25 — no language stop-lists or paraphrase maps.
 * Linguistic rewriting for short follow-ups belongs in {@link RagQueryRewrite} / {@link RagSession}.
 *
 * <p>Terms absent from the index are dropped. Queries whose distinct tokens are mostly
 * out-of-vocabulary (corpus DF statistics only) return no terms.
 */
final class RagQueryTerms {

  private RagQueryTerms() {
  }

  static List<String> select(final Map<String, Integer> docFreq, final int docCount,
                             final String query) {
    requireNonNull(docFreq, "docFreq");
    requireNonNull(query, "query");
    if (docCount <= 0) {
      return List.of();
    }
    List<String> rawDistinct = List.copyOf(new LinkedHashSet<>(PassagePreparser.tokenize(query)));
    if (rawDistinct.isEmpty()) {
      return List.of();
    }
    if (queryOutsideCorpus(docFreq, rawDistinct)) {
      return List.of();
    }

    List<String> known = rawDistinct.stream()
      .filter(term -> docFreq.getOrDefault(term, 0) > 0)
      .toList();
    if (known.isEmpty()) {
      return List.of();
    }

    int maxDf = maxCommonDocFreq(docCount);
    List<String> discriminative = known.stream()
      .filter(term -> docFreq.getOrDefault(term, 0) <= maxDf)
      .toList();
    if (!discriminative.isEmpty()) {
      return List.copyOf(discriminative);
    }

    return known.stream()
      .sorted(Comparator.comparingInt(term -> docFreq.getOrDefault(term, 0)))
      .limit(Math.min(3, known.size()))
      .toList();
  }

  /**
   * Terms in more passages than this are treated as overly common for selection.
   */
  static int maxCommonDocFreq(final int docCount) {
    return Math.max(3, (docCount * 2) / 3);
  }

  /**
   * Many distinct query tokens never appear in the corpus — e.g. a coding request against a
   * fairy-tale index. Pure DF/OOV ratio; no language word lists.
   */
  static boolean queryOutsideCorpus(final Map<String, Integer> docFreq,
                                    final List<String> rawDistinct) {
    requireNonNull(docFreq, "docFreq");
    requireNonNull(rawDistinct, "rawDistinct");
    if (rawDistinct.size() < 4) {
      return false;
    }
    long unknown = rawDistinct.stream().filter(term -> docFreq.getOrDefault(term, 0) == 0).count();
    return unknown >= 2 && unknown * 2 >= rawDistinct.size();
  }

  static boolean queryTooBroadForCorpus(final int rawDistinctTerms,
                                        final List<String> selectedTerms) {
    return rawDistinctTerms >= 3 && selectedTerms.isEmpty();
  }

  static boolean qualifies(final PreparedPassage passage, final List<String> selectedTerms) {
    requireNonNull(passage, "passage");
    requireNonNull(selectedTerms, "selectedTerms");
    if (selectedTerms.isEmpty()) {
      return false;
    }
    Set<String> passageTerms = new LinkedHashSet<>(passage.termFreqs().keySet());
    long matched = selectedTerms.stream().filter(passageTerms::contains).count();
    int need = Math.max(1, (selectedTerms.size() + 1) / 2);
    return matched >= need;
  }
}
