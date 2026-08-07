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
    if (queryOutsideCorpus(docFreq, docCount, rawDistinct)) {
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

    boolean hasOov = rawDistinct.stream().anyMatch(term -> docFreq.getOrDefault(term, 0) == 0);
    if (hasOov) {
      return List.of();
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
   * Rare enough to count as topical grounding (not fairy-tale stopword noise).
   */
  static int maxRareDocFreq(final int docCount) {
    return Math.max(2, docCount / 10);
  }

  /**
   * Many distinct query tokens never appear in the corpus — e.g. a coding request against a
   * fairy-tale index. Pure DF/OOV ratio; no language word lists.
   *
   * <p>Contentful OOV tokens (length ≥ 5, no inflection hit) reject the query unless another
   * query token is rare in the corpus (e.g. {@code Paris}/{@code France} keeps a capitals-style
   * ask; lone {@code estonia} with only {@code think}/{@code about} does not).
   */
  static boolean queryOutsideCorpus(final Map<String, Integer> docFreq,
                                    final int docCount,
                                    final List<String> rawDistinct) {
    requireNonNull(docFreq, "docFreq");
    requireNonNull(rawDistinct, "rawDistinct");
    if (rawDistinct.isEmpty() || docCount <= 0) {
      return false;
    }
    if (hasUngroundedContentfulOov(docFreq, docCount, rawDistinct)) {
      return true;
    }
    if (rawDistinct.size() < 4) {
      return false;
    }
    long unknown = rawDistinct.stream().filter(term -> docFreq.getOrDefault(term, 0) == 0).count();
    return unknown >= 2 && unknown * 2 >= rawDistinct.size();
  }

  private static boolean hasUngroundedContentfulOov(
    final Map<String, Integer> docFreq,
    final int docCount,
    final List<String> rawDistinct
  ) {
    boolean contentfulOov = rawDistinct.stream().anyMatch(term -> isContentfulOov(docFreq, term));
    if (!contentfulOov) {
      return false;
    }
    return !hasRareKnownContent(docFreq, docCount, rawDistinct);
  }

  private static boolean hasRareKnownContent(
    final Map<String, Integer> docFreq,
    final int docCount,
    final List<String> rawDistinct
  ) {
    int rareDf = maxRareDocFreq(docCount);
    return rawDistinct.stream()
      .flatMap(term -> PassagePreparser.tokenize(term).stream())
      .anyMatch(key -> {
        int df = docFreq.getOrDefault(key, 0);
        return key.length() >= 4 && df > 0 && df <= rareDf;
      });
  }

  private static boolean isContentfulOov(final Map<String, Integer> docFreq, final String term) {
    if (term.length() < 5 || docFreq.getOrDefault(term, 0) > 0) {
      return false;
    }
    return PassagePreparser.tokenize(term).stream()
      .noneMatch(key -> docFreq.getOrDefault(key, 0) > 0);
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
