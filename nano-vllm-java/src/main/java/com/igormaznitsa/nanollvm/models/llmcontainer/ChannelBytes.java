package com.igormaznitsa.nanollvm.models.llmcontainer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

final class ChannelBytes {

  private ChannelBytes() {
  }

  static ByteBuffer readHeap(final FileChannel channel, final long size) throws IOException {
    if (size < 0 || size > Integer.MAX_VALUE) {
      throw new IOException("file larger than 2GiB heap read limit");
    }
    ByteBuffer buf = ByteBuffer.allocate((int) size).order(ByteOrder.LITTLE_ENDIAN);
    readFully(channel, 0L, buf);
    buf.clear();
    return buf;
  }

  static void readFully(
    final FileChannel channel,
    final long position,
    final ByteBuffer dest
  ) throws IOException {
    long at = position;
    while (dest.hasRemaining()) {
      int n = channel.read(dest, at);
      if (n < 0) {
        throw new IOException("unexpected EOF at offset " + at);
      }
      at += n;
    }
  }
}
