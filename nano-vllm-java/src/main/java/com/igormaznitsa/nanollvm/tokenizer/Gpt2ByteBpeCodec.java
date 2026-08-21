package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * GPT-2 byte-level BPE.
 *
 * @since 1.2.0
 */
final class Gpt2ByteBpeCodec implements TokenCodec {

  private static final Pattern GPT2_PATTERN = Pattern.compile(
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  );

  private final TokenVocab vocab;
  private final BpeMerges merges;
  private final boolean byteFallback;

  Gpt2ByteBpeCodec(final TokenVocab vocab, final BpeMerges merges, final boolean byteFallback) {
    this.vocab = requireNonNull(vocab, "vocab");
    this.merges = requireNonNull(merges, "merges");
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
      Matcher matcher = GPT2_PATTERN.matcher(text);
      if (!matcher.find(i) || matcher.start() != i) {
        int cp = text.codePointAt(i);
        String ch = new String(Character.toChars(cp));
        Integer id = this.vocab.id(ch);
        if (id != null) {
          ids.add(id);
        }
        i += Character.charCount(cp);
        continue;
      }
      ids.addAll(this.merges.toIds(
        this.merges.merge(Gpt2Bytes.utf8ToSymbols(matcher.group())),
        this.vocab,
        this.byteFallback,
        false));
      i = matcher.end();
    }
    return ids;
  }

  @Override
  public String decode(final List<String> pieces) {
    String tokenString = MetaspaceText.concat(pieces);
    StringBuilder out = new StringBuilder();
    int i = 0;
    while (i < tokenString.length()) {
      String special = this.vocab.matchAdded(tokenString, i);
      if (special != null) {
        out.append(special);
        i += special.length();
        continue;
      }
      String ch = String.valueOf(tokenString.charAt(i));
      Integer b = Gpt2Bytes.unsignedByte(ch);
      if (b == null) {
        out.append(tokenString.charAt(i));
        i++;
        continue;
      }
      List<Byte> bytes = new ArrayList<>();
      while (i < tokenString.length()) {
        if (this.vocab.matchAdded(tokenString, i) != null) {
          break;
        }
        Integer next = Gpt2Bytes.unsignedByte(String.valueOf(tokenString.charAt(i)));
        if (next == null) {
          break;
        }
        bytes.add(next.byteValue());
        i++;
      }
      byte[] arr = new byte[bytes.size()];
      for (int j = 0; j < bytes.size(); j++) {
        arr[j] = bytes.get(j);
      }
      out.append(Utf8Complete.decode(arr));
    }
    return out.toString();
  }
}
