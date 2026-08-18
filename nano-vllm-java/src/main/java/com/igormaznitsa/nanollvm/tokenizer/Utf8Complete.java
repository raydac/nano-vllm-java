package com.igormaznitsa.nanollvm.tokenizer;

import static java.nio.charset.StandardCharsets.UTF_8;

final class Utf8Complete {

  private Utf8Complete() {
  }

  static String decode(final byte[] bytes) {
    if (bytes == null || bytes.length == 0) {
      return "";
    }
    int complete = 0;
    int i = 0;
    while (i < bytes.length) {
      int need = sequenceLength(bytes[i]);
      if (need < 1 || i + need > bytes.length) {
        break;
      }
      boolean ok = true;
      for (int j = 1; j < need; j++) {
        if ((bytes[i + j] & 0xC0) != 0x80) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        break;
      }
      i += need;
      complete = i;
    }
    return complete == 0 ? "" : new String(bytes, 0, complete, UTF_8);
  }

  private static int sequenceLength(final byte lead) {
    int v = lead & 0xFF;
    if (v < 0x80) {
      return 1;
    }
    if (v < 0xC2) {
      return -1;
    }
    if (v < 0xE0) {
      return 2;
    }
    if (v < 0xF0) {
      return 3;
    }
    if (v < 0xF5) {
      return 4;
    }
    return -1;
  }
}
