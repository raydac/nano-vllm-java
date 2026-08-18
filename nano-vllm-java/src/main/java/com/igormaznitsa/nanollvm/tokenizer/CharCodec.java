package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

final class CharCodec implements TokenCodec {

  private final TokenVocab vocab;
  private final String unkToken;

  CharCodec(final TokenVocab vocab, final String unkToken) {
    this.vocab = requireNonNull(vocab, "vocab");
    this.unkToken = unkToken == null || unkToken.isBlank() ? "<unk>" : unkToken;
  }

  @Override
  public List<Integer> encode(final String text) {
    List<Integer> ids = new ArrayList<>();
    Integer unk = this.vocab.id(this.unkToken);
    for (int i = 0; i < text.length(); ) {
      String special = this.vocab.matchAdded(text, i);
      if (special != null) {
        ids.add(this.vocab.id(special));
        i += special.length();
        continue;
      }
      int cp = text.codePointAt(i);
      String ch = new String(Character.toChars(cp));
      Integer id = this.vocab.id(ch);
      if (id == null) {
        id = unk;
      }
      if (id == null) {
        throw new IllegalStateException("Char vocab is missing '" + ch + "' and UNK");
      }
      ids.add(id);
      i += Character.charCount(cp);
    }
    return ids;
  }

  @Override
  public String decode(final List<String> pieces) {
    return MetaspaceText.decode(MetaspaceText.concat(pieces));
  }
}
