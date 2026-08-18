package com.igormaznitsa.nanollvm.tokenizer;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * BPE merge ranks.
 *
 * @since 1.1.1
 */
final class BpeMerges {

  private final Map<String, Integer> ranks;

  BpeMerges(final Map<String, Integer> ranks) {
    this.ranks = Map.copyOf(requireNonNull(ranks, "ranks"));
  }

  static List<Integer> mergeByScore(
    final List<String> symbols,
    final TokenVocab vocab,
    final float[] scores
  ) {
    if (symbols.isEmpty()) {
      return List.of();
    }
    List<String> word = new ArrayList<>(symbols);
    while (word.size() > 1) {
      int bestIndex = -1;
      float bestScore = Float.NEGATIVE_INFINITY;
      for (int i = 0; i < word.size() - 1; i++) {
        String pair = word.get(i) + word.get(i + 1);
        Integer id = vocab.id(pair);
        if (id == null || id < 0 || id >= scores.length) {
          continue;
        }
        float score = scores[id];
        if (score > bestScore) {
          bestScore = score;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) {
        break;
      }
      List<String> next = new ArrayList<>();
      int i = 0;
      while (i < word.size()) {
        if (i == bestIndex) {
          next.add(word.get(i) + word.get(i + 1));
          i += 2;
        } else {
          next.add(word.get(i));
          i++;
        }
      }
      word = next;
    }
    List<Integer> ids = new ArrayList<>(word.size());
    for (String piece : word) {
      Integer id = vocab.id(piece);
      if (id != null) {
        ids.add(id);
      }
    }
    return ids;
  }

  private static void appendFallback(
    final List<Integer> ids,
    final String piece,
    final TokenVocab vocab,
    final boolean metaspaceBytes
  ) {
    if (metaspaceBytes) {
      for (byte b : piece.getBytes(UTF_8)) {
        Integer bid = vocab.id("<0x%02X>".formatted(b & 0xFF));
        if (bid != null) {
          ids.add(bid);
        }
      }
      return;
    }
    for (int i = 0; i < piece.length(); ) {
      int cp = piece.codePointAt(i);
      String symbol = new String(Character.toChars(cp));
      Integer bid = vocab.id(symbol);
      if (bid == null) {
        for (byte b : symbol.getBytes(UTF_8)) {
          Integer bb = vocab.id(Gpt2Bytes.symbol(b & 0xFF));
          if (bb != null) {
            ids.add(bb);
          }
        }
      } else {
        ids.add(bid);
      }
      i += Character.charCount(cp);
    }
  }

  List<String> merge(final List<String> tokens) {
    if (tokens.isEmpty()) {
      return List.of();
    }
    List<String> word = new ArrayList<>(tokens);
    while (word.size() > 1) {
      int bestRank = Integer.MAX_VALUE;
      int bestIndex = -1;
      for (int i = 0; i < word.size() - 1; i++) {
        Integer rank = this.ranks.get(word.get(i) + " " + word.get(i + 1));
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) {
        break;
      }
      List<String> next = new ArrayList<>();
      int i = 0;
      while (i < word.size()) {
        if (i == bestIndex) {
          next.add(word.get(i) + word.get(i + 1));
          i += 2;
        } else {
          next.add(word.get(i));
          i++;
        }
      }
      word = next;
    }
    return word;
  }

  List<Integer> toIds(
    final List<String> merged,
    final TokenVocab vocab,
    final boolean byteFallback,
    final boolean metaspaceBytes
  ) {
    List<Integer> ids = new ArrayList<>(merged.size());
    for (String piece : merged) {
      Integer id = vocab.id(piece);
      if (id != null) {
        ids.add(id);
      } else if (byteFallback) {
        appendFallback(ids, piece, vocab, metaspaceBytes);
      }
    }
    return ids;
  }
}
