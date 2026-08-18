package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

final class UnigramCodec implements TokenCodec {

  private final TokenVocab vocab;
  private final UnigramScores scores;
  private final boolean prependMetaSpace;

  UnigramCodec(
    final TokenVocab vocab,
    final UnigramScores scores,
    final boolean prependMetaSpace
  ) {
    this.vocab = requireNonNull(vocab, "vocab");
    this.scores = requireNonNull(scores, "scores");
    this.prependMetaSpace = prependMetaSpace;
    if (!this.scores.isPresent()) {
      throw new ModelLoadException("Unigram tokenizer is missing piece scores");
    }
  }

  private static void emitUnk(
    final String text,
    final int end,
    final int unkId,
    final float[] scores,
    final double[] best,
    final int[] prev,
    final int[] tokenAt
  ) {
    if (unkId < 0 || unkId >= scores.length) {
      return;
    }
    int start = end - 1;
    if (end >= 2 && Character.isSurrogatePair(text.charAt(end - 2), text.charAt(end - 1))) {
      start = end - 2;
    }
    if (best[start] == Double.NEGATIVE_INFINITY) {
      return;
    }
    best[end] = best[start] + scores[unkId];
    prev[end] = start;
    tokenAt[end] = unkId;
  }

  @Override
  public List<Integer> encode(final String text) {
    List<Integer> ids = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      String special = this.vocab.matchAdded(text, i);
      if (special != null) {
        ids.add(this.vocab.id(special));
        i += special.length();
        continue;
      }
      int nextSpecial = this.vocab.findNextAdded(text, i);
      String chunk = text.substring(i, nextSpecial);
      if (!chunk.isEmpty()) {
        ids.addAll(this.viterbi(MetaspaceText.withWordMarks(chunk, this.prependMetaSpace)));
      }
      i = nextSpecial;
    }
    return ids;
  }

  @Override
  public String decode(final List<String> pieces) {
    return MetaspaceText.decode(MetaspaceText.concat(pieces));
  }

  private List<Integer> viterbi(final String text) {
    if (text.isEmpty()) {
      return List.of();
    }
    int n = text.length();
    int maxPiece = this.scores.maxPieceChars();
    float[] table = this.scores.scores();
    int unkId = this.scores.unkId();
    double[] best = new double[n + 1];
    int[] prev = new int[n + 1];
    int[] tokenAt = new int[n + 1];
    Arrays.fill(best, Double.NEGATIVE_INFINITY);
    Arrays.fill(prev, -1);
    best[0] = 0.0;

    for (int end = 1; end <= n; end++) {
      int startMin = Math.max(0, end - maxPiece);
      for (int start = startMin; start < end; start++) {
        if (best[start] == Double.NEGATIVE_INFINITY) {
          continue;
        }
        Integer id = this.vocab.id(text.substring(start, end));
        if (id == null || id < 0 || id >= table.length) {
          continue;
        }
        double score = best[start] + table[id];
        if (score > best[end]) {
          best[end] = score;
          prev[end] = start;
          tokenAt[end] = id;
        }
      }
      if (best[end] == Double.NEGATIVE_INFINITY) {
        emitUnk(text, end, unkId, table, best, prev, tokenAt);
      }
    }

    if (best[n] == Double.NEGATIVE_INFINITY) {
      throw new IllegalStateException("Unigram tokenizer could not segment text");
    }
    List<Integer> ids = new ArrayList<>();
    for (int pos = n; pos > 0; pos = prev[pos]) {
      ids.add(tokenAt[pos]);
    }
    Collections.reverse(ids);
    return ids;
  }

  @SuppressWarnings("ArrayRecordComponent")
  record UnigramScores(float[] scores, int unkId, int maxPieceChars) {

    static UnigramScores of(
      final Map<String, Integer> vocab,
      final Map<Integer, Float> scoresById,
      final int unkId
    ) {
      int maxId = -1;
      int maxPiece = 1;
      for (var e : vocab.entrySet()) {
        maxId = Math.max(maxId, e.getValue());
        maxPiece = Math.max(maxPiece, e.getKey().length());
      }
      float[] scores = new float[Math.max(0, maxId + 1)];
      for (var e : scoresById.entrySet()) {
        int id = e.getKey();
        if (id >= 0 && id < scores.length) {
          scores[id] = e.getValue();
        }
      }
      int resolvedUnk = unkId >= 0 ? unkId : vocab.getOrDefault("<unk>", -1);
      return new UnigramScores(scores, resolvedUnk, maxPiece);
    }

    boolean isPresent() {
      return this.scores.length > 0;
    }
  }
}
