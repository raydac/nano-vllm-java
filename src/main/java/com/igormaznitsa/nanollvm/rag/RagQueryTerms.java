package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Query-side term selection and hit qualification — no fixed stopword lists.
 *
 * <p>Queries whose tokens are mostly absent from the index are treated as out-of-corpus and
 * return no hits. Otherwise every query token that occurs in the corpus is kept; BM25 IDF
 * down-weights frequent terms. A hit must share enough selected tokens with the passage.
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

    List<String> known = new ArrayList<>();
    for (String term : rawDistinct) {
      if (docFreq.getOrDefault(term, 0) > 0) {
        known.add(term);
      }
    }
    return List.copyOf(known);
  }

  /**
   * Many distinct query tokens never appear in the corpus — e.g. a coding request against a
   * fairy-tale index ({@code java}, {@code program}, {@code file}, …).
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
    if (selectedTerms.size() == 1) {
      return matched >= 1;
    }
    return matched >= 2;
  }
}
