package com.igormaznitsa.nanollvm.tokenizer;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * SentencePiece precompiled charsmap.
 *
 * @since 1.1.1
 */
final class PrecompiledCharsMap {

  private final int[] trie;
  private final byte[] normalized;

  private PrecompiledCharsMap(final int[] trie, final byte[] normalized) {
    this.trie = trie;
    this.normalized = normalized;
  }

  static PrecompiledCharsMap parse(final byte[] blob) {
    requireNonNull(blob, "blob");
    if (blob.length < 4) {
      throw new ModelLoadException("precompiled charsmap is truncated");
    }
    ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
    int trieBytes = buf.getInt();
    if (trieBytes < 0 || trieBytes % 4 != 0 || 4 + trieBytes > blob.length) {
      throw new ModelLoadException("precompiled charsmap trie size is invalid");
    }
    int trieLen = trieBytes / 4;
    int[] trie = new int[trieLen];
    for (int i = 0; i < trieLen; i++) {
      trie[i] = buf.getInt();
    }
    byte[] normalized = new byte[blob.length - 4 - trieBytes];
    buf.get(normalized);
    return new PrecompiledCharsMap(trie, normalized);
  }

  private static boolean hasLeaf(final int unit) {
    return ((unit >>> 8) & 1) == 1;
  }

  private static int value(final int unit) {
    return unit & 0x7FFFFFFF;
  }

  private static int label(final int unit) {
    return unit & 0x800000FF;
  }

  private static int offset(final int unit) {
    return (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
  }

  String normalize(final String original) {
    if (original.isEmpty() || this.trie.length == 0) {
      return original;
    }
    StringBuilder out = new StringBuilder(original.length());
    BreakIterator boundaries = BreakIterator.getCharacterInstance(Locale.ROOT);
    boundaries.setText(original);
    int start = boundaries.first();
    for (int end = boundaries.next(); end != BreakIterator.DONE;
         start = end, end = boundaries.next()) {
      String grapheme = original.substring(start, end);
      if (grapheme.length() < 6) {
        String mapped = this.transform(grapheme);
        if (mapped != null) {
          out.append(mapped);
          continue;
        }
      }
      for (int i = 0; i < grapheme.length(); ) {
        int cp = grapheme.codePointAt(i);
        int n = Character.charCount(cp);
        String part = grapheme.substring(i, i + n);
        String mapped = this.transform(part);
        out.append(mapped == null ? part : mapped);
        i += n;
      }
    }
    return out.toString();
  }

  private String transform(final String chunk) {
    List<Integer> hits = this.commonPrefixSearch(chunk.getBytes(UTF_8));
    if (hits.isEmpty()) {
      return null;
    }
    int index = hits.getFirst();
    if (index < 0 || index >= this.normalized.length) {
      return null;
    }
    int end = index;
    while (end < this.normalized.length && this.normalized[end] != 0) {
      end++;
    }
    return new String(this.normalized, index, end - index, UTF_8);
  }

  private List<Integer> commonPrefixSearch(final byte[] key) {
    if (this.trie.length == 0) {
      return List.of();
    }
    List<Integer> results = new ArrayList<>();
    int nodePos = 0;
    int unit = this.trie[nodePos];
    nodePos ^= offset(unit);
    for (byte raw : key) {
      int c = raw & 0xFF;
      if (c == 0) {
        break;
      }
      nodePos ^= c;
      if (nodePos < 0 || nodePos >= this.trie.length) {
        return List.of();
      }
      unit = this.trie[nodePos];
      if (label(unit) != c) {
        return results;
      }
      nodePos ^= offset(unit);
      if (hasLeaf(unit)) {
        if (nodePos < 0 || nodePos >= this.trie.length) {
          return List.of();
        }
        results.add(value(this.trie[nodePos]));
      }
    }
    return results;
  }
}
