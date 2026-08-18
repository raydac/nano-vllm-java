package com.igormaznitsa.nanollvm.tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 * SentencePiece {@code ▁} word-boundary helpers.
 *
 * @since 1.1.1
 */
final class MetaspaceText {

  static final String MARK = "▁";

  private MetaspaceText() {
  }

  static List<String> codepoints(final String text) {
    List<String> out = new ArrayList<>();
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      out.add(new String(Character.toChars(cp)));
      i += Character.charCount(cp);
    }
    return out;
  }

  static String withWordMarks(final String chunk, final boolean prepend) {
    String prepared = chunk.replace(" ", MARK);
    return prepend ? MARK + prepared : prepared;
  }

  static String decode(final String tokenString) {
    StringBuilder out = new StringBuilder();
    int i = 0;
    List<Byte> bytes = new ArrayList<>();
    while (i < tokenString.length()) {
      if (tokenString.startsWith("<0x", i) && i + 5 < tokenString.length()
        && tokenString.charAt(i + 5) == '>') {
        try {
          int b = Integer.parseInt(tokenString.substring(i + 3, i + 5), 16);
          bytes.add((byte) b);
          i += 6;
          continue;
        } catch (NumberFormatException ignored) {
          flushBytes(out, bytes);
        }
      }
      flushBytes(out, bytes);
      out.append(tokenString.charAt(i));
      i++;
    }
    flushBytes(out, bytes);
    return out.toString().replace(MARK, " ");
  }

  static String concat(final List<String> pieces) {
    StringBuilder sb = new StringBuilder();
    for (String piece : pieces) {
      sb.append(piece);
    }
    return sb.toString();
  }

  private static void flushBytes(final StringBuilder out, final List<Byte> bytes) {
    if (bytes.isEmpty()) {
      return;
    }
    byte[] arr = new byte[bytes.size()];
    for (int j = 0; j < bytes.size(); j++) {
      arr[j] = bytes.get(j);
    }
    out.append(Utf8Complete.decode(arr));
    bytes.clear();
  }
}
