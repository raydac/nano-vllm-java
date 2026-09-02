package com.igormaznitsa.nanollvm.models.internal.fasttext;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

final class Dictionary {

  static final byte TYPE_WORD = 0;
  static final byte TYPE_LABEL = 1;

  private static final byte[] EOS = "</s>".getBytes(UTF_8);
  private static final byte BOW = (byte) '<';
  private static final byte EOW = (byte) '>';
  private static final byte[] LABEL_PREFIX = Args.DEFAULT_LABEL.getBytes(UTF_8);

  private final Args args;
  private final Entry[] words;
  private final int[] word2int;
  private final int size;
  private final int nwords;
  private final int nlabels;
  private final long pruneidxSize;
  private final Map<Integer, Integer> pruneidx;

  private Dictionary(
    final Args args,
    final Entry[] words,
    final int[] word2int,
    final int size,
    final int nwords,
    final int nlabels,
    final long pruneidxSize,
    final Map<Integer, Integer> pruneidx
  ) {
    this.args = args;
    this.words = words;
    this.word2int = word2int;
    this.size = size;
    this.nwords = nwords;
    this.nlabels = nlabels;
    this.pruneidxSize = pruneidxSize;
    this.pruneidx = pruneidx;
  }

  static Dictionary load(final LittleEndianInput in, final Args args) throws IOException {
    final int size = in.readInt();
    final int nwords = in.readInt();
    final int nlabels = in.readInt();
    in.readLong();
    final long pruneidxSize = in.readLong();

    final Entry[] words = new Entry[size];
    for (int i = 0; i < size; i++) {
      final byte[] word = in.readNullTerminatedUtf8();
      final long count = in.readLong();
      final byte type = in.readByte();
      words[i] = new Entry(word, count, type);
    }

    final Map<Integer, Integer> pruneidx = new HashMap<>();
    for (long i = 0; i < pruneidxSize; i++) {
      pruneidx.put(in.readInt(), in.readInt());
    }

    final Dictionary dictionary = new Dictionary(
      args,
      words,
      new int[0],
      size,
      nwords,
      nlabels,
      pruneidxSize,
      pruneidx);
    dictionary.initNgrams();
    return dictionary.withWord2Int();
  }

  private static int consumeUtf8Char(final byte[] word, final int start) {
    int j = start + 1;
    while (j < word.length && (word[j] & 0xC0) == 0x80) {
      j++;
    }
    return j;
  }

  private static boolean startsWith(final byte[] token, final byte[] prefix) {
    if (token.length < prefix.length) {
      return false;
    }
    for (int i = 0; i < prefix.length; i++) {
      if (token[i] != prefix[i]) {
        return false;
      }
    }
    return true;
  }

  private static boolean isSpace(final byte value) {
    return value == ' '
      || value == '\n'
      || value == '\r'
      || value == '\t'
      || value == 0x0B
      || value == 0x0C
      || value == 0;
  }

  static int hash(final byte[] str) {
    return hash(str, 0, str.length);
  }

  static int hash(final byte[] str, final int begin, final int end) {
    int h = 0x811C9DC5;
    for (int i = begin; i < end; i++) {
      h ^= (int) str[i];
      h *= 16777619;
    }
    return h;
  }

  private static int push(final int[] line, final int size, final int capacity, final int value) {
    if (size >= capacity) {
      throw new IllegalStateException("fastText line buffer overflow");
    }
    line[size] = value;
    return size + 1;
  }

  boolean isPruned() {
    return this.pruneidxSize >= 0;
  }

  int nwords() {
    return this.nwords;
  }

  int nlabels() {
    return this.nlabels;
  }

  String getLabel(final int labelId) {
    if (labelId < 0 || labelId >= this.nlabels) {
      throw new IllegalArgumentException(
        "Label id is out of range [0, " + this.nlabels + ")");
    }
    return LittleEndianInput.utf8(this.words[labelId + this.nwords].word);
  }

  long[] labelCounts() {
    final long[] counts = new long[this.nlabels];
    for (int i = 0; i < this.nlabels; i++) {
      counts[i] = this.words[this.nwords + i].count;
    }
    return counts;
  }

  int getLine(final CharSequence text, final int[] out, final int capacity) {
    final byte[] bytes = text.toString().getBytes(UTF_8);
    int outSize = 0;
    final int[] wordHashes = new int[capacity];
    int hashCount = 0;

    int index = 0;
    while (index < bytes.length) {
      while (index < bytes.length && isSpace(bytes[index])) {
        index++;
      }
      if (index >= bytes.length) {
        break;
      }
      final int begin = index;
      while (index < bytes.length && !isSpace(bytes[index])) {
        index++;
      }
      final byte[] token = Arrays.copyOfRange(bytes, begin, index);
      if (Arrays.equals(token, EOS)) {
        break;
      }

      final int hash = hash(token);
      final int wordId = this.getId(token, hash);
      final byte type = wordId < 0 ? this.typeOf(token) : this.words[wordId].type;
      if (type == TYPE_WORD) {
        outSize = this.addSubwords(out, outSize, capacity, token, wordId);
        if (hashCount < wordHashes.length) {
          wordHashes[hashCount++] = hash;
        }
      }
    }

    return this.addWordNgrams(out, outSize, capacity, wordHashes, hashCount);
  }

  private Dictionary withWord2Int() {
    final int word2intSize = (int) Math.ceil(this.size / 0.7);
    final int[] table = new int[word2intSize];
    Arrays.fill(table, -1);
    for (int i = 0; i < this.size; i++) {
      table[this.findSlot(this.words[i].word, hash(this.words[i].word), table)] = i;
    }
    return new Dictionary(
      this.args,
      this.words,
      table,
      this.size,
      this.nwords,
      this.nlabels,
      this.pruneidxSize,
      this.pruneidx);
  }

  private void initNgrams() {
    for (int i = 0; i < this.size; i++) {
      final Entry entry = this.words[i];
      IntBuffer buffer = new IntBuffer(32);
      buffer.push(i);
      if (!Arrays.equals(entry.word, EOS)) {
        this.computeSubwordsGrowing(this.bowWordEow(entry.word), buffer);
      }
      entry.subwords = buffer.toArray();
    }
  }

  private int addSubwords(
    final int[] line,
    final int lineSize,
    final int capacity,
    final byte[] token,
    final int wordId
  ) {
    if (wordId < 0) {
      return this.computeSubwords(this.bowWordEow(token), line, lineSize, capacity);
    }
    if (this.args.maxn <= 0) {
      return this.push(line, lineSize, capacity, wordId);
    }
    int size = lineSize;
    for (final int subword : this.words[wordId].subwords) {
      size = this.push(line, size, capacity, subword);
    }
    return size;
  }

  private int addWordNgrams(
    final int[] line,
    final int lineSize,
    final int capacity,
    final int[] hashes,
    final int hashCount
  ) {
    int size = lineSize;
    final int n = this.args.wordNgrams;
    for (int i = 0; i < hashCount; i++) {
      long h = Integer.toUnsignedLong(hashes[i]);
      for (int j = i + 1; j < hashCount && j < i + n; j++) {
        h = h * 116049371L + Integer.toUnsignedLong(hashes[j]);
        size = this.pushHash(line, size, capacity, (int) (h % this.args.bucket));
      }
    }
    return size;
  }

  private int pushHash(final int[] line, final int lineSize, final int capacity, final int id) {
    if (this.pruneidxSize == 0 || id < 0) {
      return lineSize;
    }
    int mapped = id;
    if (this.pruneidxSize > 0) {
      final Integer pruned = this.pruneidx.get(id);
      if (pruned == null) {
        return lineSize;
      }
      mapped = pruned;
    }
    return this.push(line, lineSize, capacity, this.nwords + mapped);
  }

  private void computeSubwordsGrowing(final byte[] word, final IntBuffer out) {
    for (int i = 0; i < word.length; i++) {
      if ((word[i] & 0xC0) == 0x80) {
        continue;
      }
      int j = i;
      int n = 1;
      while (j < word.length && n <= this.args.maxn) {
        final int ngramEnd = consumeUtf8Char(word, j);
        j = ngramEnd;
        if (n >= this.args.minn && !(n == 1 && (i == 0 || j == word.length))) {
          final int bucket = Integer.remainderUnsigned(hash(word, i, ngramEnd), this.args.bucket);
          this.pushHashGrowing(out, bucket);
        }
        n++;
      }
    }
  }

  private int computeSubwords(
    final byte[] word,
    final int[] out,
    final int outSize,
    final int capacity
  ) {
    int size = outSize;
    for (int i = 0; i < word.length; i++) {
      if ((word[i] & 0xC0) == 0x80) {
        continue;
      }
      int j = i;
      int n = 1;
      while (j < word.length && n <= this.args.maxn) {
        final int ngramEnd = consumeUtf8Char(word, j);
        j = ngramEnd;
        if (n >= this.args.minn && !(n == 1 && (i == 0 || j == word.length))) {
          final int bucket = Integer.remainderUnsigned(hash(word, i, ngramEnd), this.args.bucket);
          size = this.pushHash(out, size, capacity, bucket);
        }
        n++;
      }
    }
    return size;
  }

  private void pushHashGrowing(final IntBuffer line, final int id) {
    if (this.pruneidxSize == 0 || id < 0) {
      return;
    }
    int mapped = id;
    if (this.pruneidxSize > 0) {
      final Integer pruned = this.pruneidx.get(id);
      if (pruned == null) {
        return;
      }
      mapped = pruned;
    }
    line.push(this.nwords + mapped);
  }

  private byte[] bowWordEow(final byte[] word) {
    final byte[] wrapped = new byte[word.length + 2];
    wrapped[0] = BOW;
    System.arraycopy(word, 0, wrapped, 1, word.length);
    wrapped[wrapped.length - 1] = EOW;
    return wrapped;
  }

  private int getId(final byte[] word, final int hash) {
    return this.word2int[this.findSlot(word, hash, this.word2int)];
  }

  private int findSlot(final byte[] word, final int hash, final int[] table) {
    final int size = table.length;
    int id = Integer.remainderUnsigned(hash, size);
    while (table[id] != -1 && !Arrays.equals(this.words[table[id]].word, word)) {
      id = (id + 1) % size;
    }
    return id;
  }

  private byte typeOf(final byte[] token) {
    return startsWith(token, LABEL_PREFIX) ? TYPE_LABEL : TYPE_WORD;
  }

  private static final class Entry {
    final byte[] word;
    final long count;
    final byte type;
    int[] subwords = new int[0];

    Entry(final byte[] word, final long count, final byte type) {
      this.word = word;
      this.count = count;
      this.type = type;
    }
  }

  private static final class IntBuffer {
    private int[] data;
    private int size;

    IntBuffer(final int capacity) {
      this.data = new int[capacity];
    }

    void push(final int value) {
      if (this.size == this.data.length) {
        this.data = Arrays.copyOf(this.data, this.data.length << 1);
      }
      this.data[this.size++] = value;
    }

    int[] toArray() {
      return Arrays.copyOf(this.data, this.size);
    }
  }
}
