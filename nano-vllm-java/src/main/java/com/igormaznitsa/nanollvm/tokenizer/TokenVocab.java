package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import java.util.Comparator;
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

  private final Map<String, Integer> tokenToId;
  private final Map<Integer, String> idToToken;
  private final List<String> addedTokensByLength;
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
      .sorted(Comparator.comparingInt(String::length).reversed())
      .toList();
    Set<Integer> skip = new HashSet<>();
    for (String text : skipTokenTexts) {
      Integer id = this.tokenToId.get(text);
      if (id != null) {
        skip.add(id);
      }
    }
    this.skipTokenIds = Set.copyOf(skip);
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

  boolean skip(final int id) {
    return this.skipTokenIds.contains(id);
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
