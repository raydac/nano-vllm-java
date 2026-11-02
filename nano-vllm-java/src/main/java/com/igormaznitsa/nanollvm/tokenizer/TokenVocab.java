package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Comparator.comparingInt;
import static java.util.Comparator.naturalOrder;
import static java.util.Objects.requireNonNull;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Token string ↔ id maps plus special and added sets.
 *
 * @since 1.2.0
 */
final class TokenVocab {

  private static final Set<String> KEPT_WHEN_SPECIAL = Set.of(
    "<think>",
    "</think>",
    "<|channel>",
    "<channel|>"
  );

  private final Map<String, Integer> tokenToId;
  private final Map<Integer, String> idToToken;
  private final List<String> addedTokensByLength;
  private final List<String> specialTokens;
  private final Set<Integer> skipTokenIds;

  TokenVocab(
    final Map<String, Integer> tokenToId,
    final Set<String> addedTokenTexts,
    final Set<String> skipTokenTexts
  ) {
    this.tokenToId = Map.copyOf(requireNonNull(tokenToId, "tokenToId"));
    Map<Integer, String> inverse = new HashMap<>();
    for (var e : this.tokenToId.entrySet()) {
      inverse.put(e.getValue(), e.getKey());
    }
    this.idToToken = Map.copyOf(inverse);
    this.addedTokensByLength = addedTokenTexts.stream()
      .sorted(comparingInt(String::length).reversed())
      .toList();
    Set<Integer> skip = new HashSet<>();
    for (String text : skipTokenTexts) {
      Integer id = this.tokenToId.get(text);
      if (id != null) {
        skip.add(id);
      }
    }
    this.skipTokenIds = Set.copyOf(skip);
    this.specialTokens = skipTokenTexts.stream()
      .filter(text -> text != null && !text.isBlank() && this.tokenToId.containsKey(text))
      .distinct()
      .sorted(comparingInt(String::length).reversed().thenComparing(naturalOrder()))
      .toList();
  }

  Integer id(final String token) {
    return this.tokenToId.get(token);
  }

  String token(final int id) {
    return this.idToToken.get(id);
  }

  boolean contains(final String token) {
    return this.tokenToId.containsKey(token);
  }

  List<String> specialTokens() {
    return this.specialTokens;
  }

  boolean skip(final int id) {
    if (!this.skipTokenIds.contains(id)) {
      return false;
    }
    String token = this.idToToken.get(id);
    return token == null || !KEPT_WHEN_SPECIAL.contains(token);
  }

  String matchAdded(final String text, final int index) {
    for (String added : this.addedTokensByLength) {
      if (text.startsWith(added, index)) {
        return added;
      }
    }
    return null;
  }

  int findNextAdded(final String text, final int from) {
    int best = text.length();
    for (String added : this.addedTokensByLength) {
      int at = text.indexOf(added, from);
      if (at >= from && at < best) {
        best = at;
      }
    }
    return best;
  }
}
