package com.igormaznitsa.nanollvm.tokenizer;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * GPT-2 printable-byte mapping.
 *
 * @since 1.1.1
 */
final class Gpt2Bytes {

  private static final String[] ENCODER = new String[256];
  private static final Map<String, Integer> DECODER = new HashMap<>();

  static {
    List<Integer> bs = new ArrayList<>();
    for (int i = '!'; i <= '~'; i++) {
      bs.add(i);
    }
    for (int i = '¡'; i <= '¬'; i++) {
      bs.add(i);
    }
    for (int i = '®'; i <= 'ÿ'; i++) {
      bs.add(i);
    }
    List<Integer> cs = new ArrayList<>(bs);
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n++;
      }
    }
    for (int i = 0; i < bs.size(); i++) {
      String ch = new String(Character.toChars(cs.get(i)));
      ENCODER[bs.get(i)] = ch;
      DECODER.put(ch, bs.get(i));
    }
  }

  private Gpt2Bytes() {
  }

  static String symbol(final int unsignedByte) {
    return ENCODER[unsignedByte & 0xFF];
  }

  static Integer unsignedByte(final String symbol) {
    return DECODER.get(symbol);
  }

  static List<String> utf8ToSymbols(final String text) {
    byte[] bytes = text.getBytes(UTF_8);
    List<String> tokens = new ArrayList<>(bytes.length);
    for (byte b : bytes) {
      tokens.add(ENCODER[b & 0xFF]);
    }
    return tokens;
  }
}
