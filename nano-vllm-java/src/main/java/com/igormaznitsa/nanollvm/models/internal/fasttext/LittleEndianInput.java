package com.igormaznitsa.nanollvm.models.internal.fasttext;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

final class LittleEndianInput {

  private final InputStream in;
  private final byte[] scratch8 = new byte[8];

  LittleEndianInput(final InputStream in) {
    this.in = in;
  }

  static String utf8(final byte[] bytes) {
    return new String(bytes, StandardCharsets.UTF_8);
  }

  boolean readBoolean() throws IOException {
    final int value = this.in.read();
    if (value < 0) {
      throw new EOFException();
    }
    return value != 0;
  }

  byte readByte() throws IOException {
    final int value = this.in.read();
    if (value < 0) {
      throw new EOFException();
    }
    return (byte) value;
  }

  int readInt() throws IOException {
    this.readFully(this.scratch8, 4);
    return (this.scratch8[0] & 0xFF)
      | ((this.scratch8[1] & 0xFF) << 8)
      | ((this.scratch8[2] & 0xFF) << 16)
      | ((this.scratch8[3] & 0xFF) << 24);
  }

  long readLong() throws IOException {
    this.readFully(this.scratch8, 8);
    return (this.scratch8[0] & 0xFFL)
      | ((this.scratch8[1] & 0xFFL) << 8)
      | ((this.scratch8[2] & 0xFFL) << 16)
      | ((this.scratch8[3] & 0xFFL) << 24)
      | ((this.scratch8[4] & 0xFFL) << 32)
      | ((this.scratch8[5] & 0xFFL) << 40)
      | ((this.scratch8[6] & 0xFFL) << 48)
      | ((this.scratch8[7] & 0xFFL) << 56);
  }

  float readFloat() throws IOException {
    return Float.intBitsToFloat(this.readInt());
  }

  double readDouble() throws IOException {
    return Double.longBitsToDouble(this.readLong());
  }

  void readFully(final byte[] dest) throws IOException {
    this.readFully(dest, dest.length);
  }

  void readFully(final byte[] dest, final int length) throws IOException {
    int offset = 0;
    while (offset < length) {
      final int read = this.in.read(dest, offset, length - offset);
      if (read < 0) {
        throw new EOFException();
      }
      offset += read;
    }
  }

  void readFloats(final float[] dest) throws IOException {
    for (int i = 0; i < dest.length; i++) {
      dest[i] = this.readFloat();
    }
  }

  byte[] readNullTerminatedUtf8() throws IOException {
    byte[] buffer = new byte[64];
    int size = 0;
    while (true) {
      final int value = this.in.read();
      if (value < 0) {
        throw new EOFException("truncated null-terminated string");
      }
      if (value == 0) {
        final byte[] word = new byte[size];
        System.arraycopy(buffer, 0, word, 0, size);
        return word;
      }
      if (size == buffer.length) {
        final byte[] grown = new byte[buffer.length << 1];
        System.arraycopy(buffer, 0, grown, 0, size);
        buffer = grown;
      }
      buffer[size++] = (byte) value;
    }
  }
}
