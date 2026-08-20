package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * BERT WordPiece ({@code ##} or metaspace).
 *
 * @since 1.1.1
 */
final class WordPieceCodec implements TokenCodec {

  private final TokenVocab vocab;
  private final boolean prependMetaSpace;
  private final boolean bertSplit;
  private final String continuingPrefix;
  private final String unkToken;

  private WordPieceCodec(
    final TokenVocab vocab,
    final boolean prependMetaSpace,
    final boolean bertSplit,
    final String continuingPrefix,
    final String unkToken
  ) {
    this.vocab = requireNonNull(vocab, "vocab");
    this.prependMetaSpace = prependMetaSpace;
    this.bertSplit = bertSplit;
    this.continuingPrefix = requireNonNull(continuingPrefix, "continuingPrefix");
    this.unkToken = requireNonNull(unkToken, "unkToken");
  }

  static WordPieceCodec metaspace(final TokenVocab vocab, final boolean prependMetaSpace) {
    return new WordPieceCodec(vocab, prependMetaSpace, false, "", "[UNK]");
  }

  static WordPieceCodec bert(final TokenVocab vocab, final String continuingPrefix) {
    String prefix =
      continuingPrefix == null || continuingPrefix.isBlank() ? "##" : continuingPrefix;
    return new WordPieceCodec(vocab, false, true, prefix, "[UNK]");
  }

  private static List<String> bertWords(final String text) {
    List<String> words = new ArrayList<>();
    StringBuilder current = new StringBuilder();
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      if (Character.isWhitespace(cp)) {
        flush(words, current);
      } else if (isPunctuation(cp)) {
        flush(words, current);
        words.add(new String(Character.toChars(cp)));
      } else {
        current.appendCodePoint(cp);
      }
      i += Character.charCount(cp);
    }
    flush(words, current);
    return words;
  }

  private static void flush(final List<String> words, final StringBuilder current) {
    if (!current.isEmpty()) {
      words.add(current.toString());
      current.setLength(0);
    }
  }

  private static boolean isPunctuation(final int cp) {
    int type = Character.getType(cp);
    return type == Character.CONNECTOR_PUNCTUATION
      || type == Character.DASH_PUNCTUATION
      || type == Character.START_PUNCTUATION
      || type == Character.END_PUNCTUATION
      || type == Character.INITIAL_QUOTE_PUNCTUATION
      || type == Character.FINAL_QUOTE_PUNCTUATION
      || type == Character.OTHER_PUNCTUATION
      || (cp >= 33 && cp <= 47)
      || (cp >= 58 && cp <= 64)
      || (cp >= 91 && cp <= 96)
      || (cp >= 123 && cp <= 126);
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
        if (this.bertSplit) {
          for (String word : bertWords(chunk)) {
            ids.addAll(this.greedyWordPiece(word, true));
          }
        } else {
          ids.addAll(this.greedyWordPiece(
            MetaspaceText.withWordMarks(chunk, this.prependMetaSpace), false));
        }
      }
      i = nextSpecial;
    }
    return ids;
  }

  @Override
  public String decode(final List<String> pieces) {
    if (this.bertSplit) {
      StringBuilder out = new StringBuilder();
      for (String piece : pieces) {
        if (piece.startsWith(this.continuingPrefix) && !this.continuingPrefix.isEmpty()) {
          out.append(piece.substring(this.continuingPrefix.length()));
        } else {
          if (!out.isEmpty()) {
            out.append(' ');
          }
          out.append(piece);
        }
      }
      return out.toString();
    }
    return MetaspaceText.decode(MetaspaceText.concat(pieces));
  }

  private List<Integer> greedyWordPiece(final String text, final boolean useContinuingPrefix) {
    List<Integer> ids = new ArrayList<>();
    Integer unk = this.vocab.id(this.unkToken);
    if (unk == null) {
      unk = this.vocab.id("<unk>");
    }
    int i = 0;
    boolean first = true;
    while (i < text.length()) {
      int matchedEnd = -1;
      Integer matchedId = null;
      for (int end = text.length(); end > i; end--) {
        String candidate = text.substring(i, end);
        if (useContinuingPrefix && !first && !this.continuingPrefix.isEmpty()) {
          candidate = this.continuingPrefix + candidate;
        }
        Integer id = this.vocab.id(candidate);
        if (id != null) {
          matchedEnd = end;
          matchedId = id;
          break;
        }
      }
      if (matchedId == null) {
        if (unk == null) {
          throw new IllegalStateException("missing vocab piece at index " + i + " and no [UNK]");
        }
        ids.add(unk);
        i += Character.charCount(text.codePointAt(i));
      } else {
        ids.add(matchedId);
        i = matchedEnd;
      }
      first = false;
    }
    return ids;
  }
}
