package com.igormaznitsa.nanollvm.models.internal;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Looks up words in a compiled espeak-ng {@code *_dict} hash table.
 */
final class EspeakNgCompiledDict {

  private static final int HASH_BUCKETS = 1024;
  private static final int LATIN_MIN = 0x60;
  private static final int LATIN_MAX = 0x17f;
  private static final int CYRILLIC_MIN = 0x430;
  private static final int CYRILLIC_MAX = 0x451;
  private static final byte[] LATIN_MAP = latinMap();
  private static final int[] PAIRS_RU = {
    0x010c, 0x010e, 0x0113, 0x0301, 0x030f, 0x060e, 0x0611, 0x0903, 0x0b01, 0x0b0f,
    0x0c01, 0x0c09, 0x0e01, 0x0e06, 0x0e09, 0x0e0e, 0x0e0f, 0x0e1c, 0x0f03, 0x0f11,
    0x0f12, 0x100f, 0x1011, 0x1101, 0x1106, 0x1109, 0x110f, 0x1213, 0x1220, 0x7fff
  };
  private static final EspeakNgCompiledDict EMPTY = new EspeakNgCompiledDict(
    new byte[0], new int[0], Map.of(), Set.of());

  private final byte[] data;
  private final int[] buckets;
  private final Map<Integer, String> mnemonics;
  private final Set<Integer> dictRules;

  private EspeakNgCompiledDict(
    final byte[] data,
    final int[] buckets,
    final Map<Integer, String> mnemonics,
    final Set<Integer> dictRules
  ) {
    this.data = data;
    this.buckets = buckets;
    this.mnemonics = Map.copyOf(mnemonics);
    this.dictRules = Set.copyOf(dictRules);
  }

  static EspeakNgCompiledDict none() {
    return EMPTY;
  }

  static EspeakNgCompiledDict load(
    final Path dataDir,
    final List<String> dictionaryNames,
    final List<String> phonemeTables,
    final Set<Integer> dictRules
  ) {
    requireNonNull(dataDir, "dataDir");
    requireNonNull(dictionaryNames, "dictionaryNames");
    requireNonNull(phonemeTables, "phonemeTables");
    requireNonNull(dictRules, "dictRules");
    EspeakNgPhontab phontab = EspeakNgPhontab.load(dataDir);
    if (!phontab.isLoaded()) {
      return EMPTY;
    }
    Optional<Map<Integer, String>> mnemonics = phontab.mnemonicsFor(phonemeTables);
    if (mnemonics.isEmpty()) {
      return EMPTY;
    }
    Optional<Path> dict = dictionaryNames.stream()
      .map(name -> dataDir.resolve(name + "_dict"))
      .filter(Files::isRegularFile)
      .findFirst();
    if (dict.isEmpty()) {
      return EMPTY;
    }
    try {
      byte[] bytes = Files.readAllBytes(dict.get());
      if (bytes.length < 8 + HASH_BUCKETS) {
        return EMPTY;
      }
      ByteBuffer buffer = ByteBuffer.wrap(bytes).order(LITTLE_ENDIAN);
      if (buffer.getInt(0) != HASH_BUCKETS) {
        return EMPTY;
      }
      int rulesOffset = buffer.getInt(4);
      if (rulesOffset <= 8 || rulesOffset > bytes.length) {
        return EMPTY;
      }
      int[] buckets = new int[HASH_BUCKETS];
      int cursor = 8;
      for (int hash = 0; hash < HASH_BUCKETS; hash++) {
        if (cursor >= rulesOffset) {
          return EMPTY;
        }
        buckets[hash] = cursor;
        while (cursor < rulesOffset && (bytes[cursor] & 0xff) != 0) {
          cursor += bytes[cursor] & 0xff;
        }
        cursor++;
      }
      return new EspeakNgCompiledDict(bytes, buckets, mnemonics.get(), dictRules);
    } catch (IOException ignored) {
      return EMPTY;
    }
  }

  private static Optional<KeyedWord> transpose(
    final String word,
    final byte[] original,
    final int min,
    final int max,
    final byte[] map,
    final int[] pairs
  ) {
    int[] letters = word.codePoints().toArray();
    byte[] mapped = new byte[letters.length];
    int offset = min - 1;
    for (int i = 0; i < letters.length; i++) {
      int cp = letters[i];
      if (cp < min || cp > max) {
        return Optional.empty();
      }
      int code;
      if (map == null) {
        code = cp - offset;
      } else {
        int index = cp - min;
        if (index >= map.length || map[index] <= 0) {
          return Optional.empty();
        }
        code = map[index] & 0xff;
      }
      mapped[i] = (byte) code;
    }
    byte[] packed = packSixBit(mapped, pairs, max - min + 2);
    System.arraycopy(packed, 0, original, 0, packed.length);
    return Optional.of(new KeyedWord(original, packed.length | 0x40));
  }

  private static byte[] packSixBit(final byte[] mapped, final int[] pairs, final int pairsStart) {
    int acc = 0;
    int bits = 0;
    byte[] packed = new byte[mapped.length + 1];
    int written = 0;
    int i = 0;
    while (i < mapped.length) {
      int code = mapped[i] & 0x3f;
      if (pairs != null && i + 1 < mapped.length) {
        int pair = code + ((mapped[i + 1] & 0xff) << 8);
        for (int p = 0; pair >= pairs[p]; p++) {
          if (pair == pairs[p]) {
            code = p + pairsStart;
            i++;
            break;
          }
        }
      }
      acc = (acc << 6) + (code & 0x3f);
      bits += 6;
      if (bits >= 8) {
        bits -= 8;
        packed[written++] = (byte) ((acc >> bits) & 0xff);
      }
      i++;
    }
    if (bits > 0) {
      packed[written++] = (byte) ((acc << (8 - bits)) & 0xff);
    }
    return Arrays.copyOf(packed, written);
  }

  private static int hashDictionary(final byte[] bytes) {
    int hash = 0;
    int chars = 0;
    for (int i = 0; i < bytes.length; i++) {
      int c = bytes[i] & 0xff;
      if (c == 0) {
        break;
      }
      hash = hash * 8 + c;
      hash = (hash & 0x3ff) ^ (hash >> 8);
      chars++;
    }
    return (hash + chars) & 0x3ff;
  }

  private static byte[] latinMap() {
    int[] values = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      27, 28, 29, 0, 0, 30, 31, 32, 33, 34, 35, 36, 0, 37, 38, 0,
      0, 0, 0, 39, 0, 0, 40, 0, 41, 0, 42, 0, 43, 0, 0, 0,
      0, 0, 0, 44, 0, 45, 0, 46, 0, 0, 0, 0, 0, 47, 0, 0,
      0, 48, 0, 0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 50, 0, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 0, 0,
      0, 53, 0, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 56, 0, 57, 0, 0
    };
    byte[] map = new byte[values.length];
    for (int i = 0; i < values.length; i++) {
      map[i] = (byte) values[i];
    }
    return map;
  }

  boolean isLoaded() {
    return this.buckets.length == HASH_BUCKETS;
  }

  Optional<String> lookup(final String word) {
    return this.lookupHit(word)
      .map(Hit::phonemes)
      .filter(phonemes -> !phonemes.isEmpty());
  }

  Optional<Hit> lookupHit(final String word) {
    if (!this.isLoaded() || word == null || word.isEmpty()) {
      return Optional.empty();
    }
    List<Optional<Hit>> candidates = List.of(
      this.lookupTransposed(word, LATIN_MIN, LATIN_MAX, LATIN_MAP, null),
      this.lookupTransposed(word, CYRILLIC_MIN, CYRILLIC_MAX, null, PAIRS_RU),
      this.lookupRaw(word));
    return candidates.stream()
      .flatMap(Optional::stream)
      .filter(Hit::hasPhonemes)
      .findFirst()
      .or(() -> candidates.stream().flatMap(Optional::stream).filter(Hit::hasStress).findFirst());
  }

  private Optional<Hit> lookupTransposed(
    final String word,
    final int min,
    final int max,
    final byte[] map,
    final int[] pairs
  ) {
    byte[] utf8 = word.getBytes(UTF_8);
    byte[] original = Arrays.copyOf(utf8, utf8.length + 1);
    Optional<KeyedWord> keyed = transpose(word, original, min, max, map, pairs);
    return keyed.isEmpty() ? Optional.empty() : this.matchBucket(keyed.get());
  }

  private Optional<Hit> lookupRaw(final String word) {
    byte[] utf8 = word.getBytes(UTF_8);
    byte[] original = Arrays.copyOf(utf8, utf8.length + 1);
    return this.matchBucket(new KeyedWord(original, utf8.length));
  }

  private Optional<Hit> matchBucket(final KeyedWord keyed) {
    int hash = hashDictionary(keyed.bytes());
    int cursor = this.buckets[hash];
    byte[] data = this.data;
    Optional<Hit> stressOnly = Optional.empty();
    while (cursor < data.length && (data[cursor] & 0xff) != 0) {
      int next = cursor + (data[cursor] & 0xff);
      int info = data[cursor + 1] & 0xff;
      if ((info & 0x7f) == (keyed.length() & 0x7f)) {
        int stored = keyed.length() & 0x3f;
        if ((stored > 0 &&
          Arrays.equals(keyed.bytes(), 0, stored, data, cursor + 2, cursor + 2 + stored))
          || stored == 0) {
          Optional<Hit> hit = this.hitAt(cursor, info, next);
          if (hit.isPresent() && hit.get().hasPhonemes()) {
            return hit;
          }
          if (hit.isPresent() && hit.get().hasStress() && stressOnly.isEmpty()) {
            stressOnly = hit;
          }
        }
      }
      cursor = next;
    }
    return stressOnly;
  }

  private Optional<Hit> hitAt(final int cursor, final int info, final int next) {
    if ((info & 0x80) != 0) {
      return this.stressOnlyHit(cursor, info, next);
    }
    int phonemeAt = cursor + (info & 0x3f) + 2;
    int end = phonemeAt;
    while (end < next && this.data[end] != 0) {
      end++;
    }
    if (end >= next) {
      return Optional.empty();
    }
    int flagsAt = end + 1;
    if (!this.flagsAllow(flagsAt, next)) {
      return Optional.empty();
    }
    byte[] codes = Arrays.copyOfRange(this.data, phonemeAt, end);
    String mnemonics = EspeakNgPhontab.decodeCodes(codes, this.mnemonics);
    return mnemonics.isEmpty() ? Optional.empty() : Optional.of(new Hit(mnemonics, 0));
  }

  private Optional<Hit> stressOnlyHit(final int cursor, final int info, final int next) {
    int flagsAt = cursor + (info & 0x3f) + 2;
    if (!this.flagsAllow(flagsAt, next)) {
      return Optional.empty();
    }
    int stress = this.stressFromFlags(flagsAt, next);
    return stress == 0 ? Optional.empty() : Optional.of(new Hit("", stress));
  }

  private int stressFromFlags(final int start, final int next) {
    int cursor = start;
    while (cursor < next) {
      int flag = this.data[cursor] & 0xff;
      cursor++;
      if (flag >= 0x41 && flag <= 0x47) {
        return flag - 0x40;
      }
      if (flag == 0x48 || flag == 0x4c) {
        return -1;
      }
      if (flag >= 0x49 && flag <= 0x4b) {
        return flag - 0x48;
      }
      if (flag >= 0x4d && flag <= 0x4f) {
        return flag - 0x4c;
      }
    }
    return 0;
  }

  private boolean flagsAllow(final int start, final int next) {
    int cursor = start;
    while (cursor < next) {
      int flag = this.data[cursor] & 0xff;
      cursor++;
      if (flag >= 100) {
        int condition = flag >= 132 ? flag - 132 : flag - 100;
        boolean present = this.dictRules.contains(condition);
        boolean required = flag < 132;
        if (required != present) {
          return false;
        }
      } else if (flag > 80) {
        return false;
      }
    }
    return true;
  }

  record Hit(String phonemes, int stressSyllable) {
    boolean hasPhonemes() {
      return this.phonemes != null && !this.phonemes.isEmpty();
    }

    boolean hasStress() {
      return this.stressSyllable != 0;
    }
  }

  @SuppressWarnings("ArrayRecordComponent")
  private record KeyedWord(byte[] bytes, int length) {
  }
}
