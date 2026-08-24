package com.igormaznitsa.nanollvm.models.internal;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Reads espeak-ng {@code phontab}: named phoneme tables and {@code includes} chains.
 */
final class EspeakNgPhontab {

  private static final int NAME_BYTES = 32;
  private static final int PHONEME_BYTES = 16;
  private static final EspeakNgPhontab EMPTY = new EspeakNgPhontab(List.of());

  private final List<Table> tables;

  private EspeakNgPhontab(final List<Table> tables) {
    this.tables = List.copyOf(tables);
  }

  static EspeakNgPhontab load(final Path dataDir) {
    requireNonNull(dataDir, "dataDir");
    Path file = dataDir.resolve("phontab");
    if (!Files.isRegularFile(file)) {
      return EMPTY;
    }
    try {
      byte[] bytes = Files.readAllBytes(file);
      if (bytes.length < 4) {
        return EMPTY;
      }
      ByteBuffer buffer = ByteBuffer.wrap(bytes).order(LITTLE_ENDIAN);
      int tableCount = Byte.toUnsignedInt(buffer.get(0));
      int offset = 4;
      List<Table> tables = new ArrayList<>(tableCount);
      for (int i = 0; i < tableCount; i++) {
        if (offset + 4 + NAME_BYTES > bytes.length) {
          return EMPTY;
        }
        int phonemeCount = Byte.toUnsignedInt(bytes[offset]);
        int includes = Byte.toUnsignedInt(bytes[offset + 1]);
        offset += 4;
        String name = readName(bytes, offset);
        offset += NAME_BYTES;
        int recordsEnd = offset + phonemeCount * PHONEME_BYTES;
        if (recordsEnd > bytes.length) {
          return EMPTY;
        }
        Map<Integer, String> mnemonics = new HashMap<>();
        for (int p = 0; p < phonemeCount; p++) {
          int at = offset + p * PHONEME_BYTES;
          int mnemonic = buffer.getInt(at);
          int code = Byte.toUnsignedInt(bytes[at + 10]);
          mnemonics.put(code, unpackMnemonic(mnemonic));
        }
        offset = recordsEnd;
        tables.add(new Table(name, includes, Map.copyOf(mnemonics)));
      }
      return new EspeakNgPhontab(tables);
    } catch (IOException ignored) {
      return EMPTY;
    }
  }

  static String decodeCodes(final byte[] codes, final Map<Integer, String> mnemonics) {
    requireNonNull(codes, "codes");
    requireNonNull(mnemonics, "mnemonics");
    StringBuilder phonemes = new StringBuilder(codes.length * 2);
    for (byte raw : codes) {
      int code = Byte.toUnsignedInt(raw);
      if (code == 0) {
        break;
      }
      String mnemonic = mnemonics.getOrDefault(code, "");
      if (isSpoken(mnemonic)) {
        phonemes.append(mnemonic);
      }
    }
    return phonemes.toString();
  }

  private static boolean isSpoken(final String mnemonic) {
    if (mnemonic.isEmpty()) {
      return false;
    }
    int first = mnemonic.charAt(0);
    return first != '_' && first != '#' && first != '*' && first != '=' && first != '%';
  }

  private static String readName(final byte[] bytes, final int offset) {
    int end = offset;
    int limit = offset + NAME_BYTES;
    while (end < limit && bytes[end] != 0) {
      end++;
    }
    return new String(bytes, offset, end - offset, US_ASCII);
  }

  private static String unpackMnemonic(final int mnemonic) {
    StringBuilder letters = new StringBuilder(4);
    int packed = mnemonic;
    for (int i = 0; i < 4; i++) {
      int cp = packed & 0xff;
      if (cp == 0) {
        break;
      }
      letters.append((char) cp);
      packed >>>= 8;
    }
    return letters.toString();
  }

  boolean isLoaded() {
    return !this.tables.isEmpty();
  }

  Optional<Map<Integer, String>> mnemonicsFor(final List<String> names) {
    requireNonNull(names, "names");
    for (String name : names) {
      Optional<Integer> index = this.indexOf(name);
      if (index.isPresent()) {
        return Optional.of(this.mnemonicsAt(index.get()));
      }
    }
    return Optional.empty();
  }

  private Optional<Integer> indexOf(final String name) {
    for (int i = 0; i < this.tables.size(); i++) {
      if (this.tables.get(i).name().equalsIgnoreCase(name)) {
        return Optional.of(i);
      }
    }
    return Optional.empty();
  }

  private Map<Integer, String> mnemonicsAt(final int index) {
    Map<Integer, String> merged = new HashMap<>();
    this.fillIncludes(index, merged);
    return Map.copyOf(merged);
  }

  private void fillIncludes(final int index, final Map<Integer, String> merged) {
    if (index < 0 || index >= this.tables.size()) {
      return;
    }
    Table table = this.tables.get(index);
    if (table.includes() > 0) {
      this.fillIncludes(table.includes() - 1, merged);
    }
    merged.putAll(table.mnemonics());
  }

  private record Table(String name, int includes, Map<Integer, String> mnemonics) {
  }
}
