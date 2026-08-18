package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Whole-word vocab lookup.
 *
 * @since 1.1.1
 */
final class WordLevelCodec implements TokenCodec {

  private final TokenVocab vocab;
  private final String unkToken;

  WordLevelCodec(final TokenVocab vocab, final String unkToken) {
    this.vocab = requireNonNull(vocab, "vocab");
    this.unkToken = unkToken == null || unkToken.isBlank() ? "[UNK]" : unkToken;
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
      if (Character.isWhitespace(text.codePointAt(i))) {
        i += Character.charCount(text.codePointAt(i));
        continue;
      }
      int next = this.vocab.findNextAdded(text, i);
      int end = i;
      while (end < next && !Character.isWhitespace(text.codePointAt(end))) {
        end += Character.charCount(text.codePointAt(end));
      }
      String word = text.substring(i, end);
      Integer id = this.vocab.id(word);
      if (id == null) {
        id = this.vocab.id(this.unkToken);
      }
      if (id == null) {
        id = this.vocab.id("<unk>");
      }
      if (id == null) {
        throw new IllegalStateException("WordLevel vocab is missing '" + word + "' and UNK");
      }
      ids.add(id);
      i = end;
    }
    return ids;
  }

  @Override
  public String decode(final List<String> pieces) {
    return String.join(" ", pieces);
  }
}
