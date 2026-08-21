package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * SentencePiece-style BPE with {@code ▁} word boundaries.
 *
 * @since 1.2.0
 */
final class MetaspaceBpeCodec implements TokenCodec {

  private final TokenVocab vocab;
  private final BpeMerges merges;
  private final float[] scores;
  private final boolean prependMetaSpace;
  private final boolean byteFallback;

  MetaspaceBpeCodec(
    final TokenVocab vocab,
    final BpeMerges merges,
    final boolean prependMetaSpace,
    final boolean byteFallback
  ) {
    this(vocab, merges, new float[0], prependMetaSpace, byteFallback);
  }

  MetaspaceBpeCodec(
    final TokenVocab vocab,
    final float[] scores,
    final boolean prependMetaSpace,
    final boolean byteFallback
  ) {
    this(vocab, new BpeMerges(Map.of()), scores, prependMetaSpace, byteFallback);
  }

  private MetaspaceBpeCodec(
    final TokenVocab vocab,
    final BpeMerges merges,
    final float[] scores,
    final boolean prependMetaSpace,
    final boolean byteFallback
  ) {
    this.vocab = requireNonNull(vocab, "vocab");
    this.merges = requireNonNull(merges, "merges");
    this.scores = requireNonNull(scores, "scores").clone();
    this.prependMetaSpace = prependMetaSpace;
    this.byteFallback = byteFallback;
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
        List<String> symbols = MetaspaceText.codepoints(
          MetaspaceText.withWordMarks(chunk, this.prependMetaSpace));
        if (this.scores.length > 0) {
          ids.addAll(BpeMerges.mergeByScore(symbols, this.vocab, this.scores));
        } else {
          ids.addAll(this.merges.toIds(
            this.merges.merge(symbols), this.vocab, this.byteFallback, true));
        }
      }
      i = nextSpecial;
    }
    return ids;
  }

  @Override
  public String decode(final List<String> pieces) {
    return MetaspaceText.decode(MetaspaceText.concat(pieces));
  }
}
