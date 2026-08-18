package com.igormaznitsa.nanollvm.tokenizer;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class SentencePieceModel {

  static final int UNIGRAM = 1;
  static final int BPE = 2;
  static final int WORD = 3;
  static final int CHAR = 4;

  private static final int TYPE_NORMAL = 1;
  private static final int TYPE_UNKNOWN = 2;
  private static final int TYPE_CONTROL = 3;
  private static final int TYPE_USER_DEFINED = 4;

  private SentencePieceModel() {
  }

  static Parsed parse(final byte[] bytes) {
    requireNonNull(bytes, "bytes");
    try {
      ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
      List<Piece> pieces = new ArrayList<>();
      Trainer trainer = new Trainer();
      NormalizerSpec normalizer = new NormalizerSpec();
      while (buf.hasRemaining()) {
        long tag = readVarint(buf);
        int field = (int) (tag >>> 3);
        int wire = (int) (tag & 7L);
        switch (field) {
          case 1 -> {
            if (wire != 2) {
              skip(buf, wire);
            } else {
              pieces.add(readPiece(readLengthDelimited(buf)));
            }
          }
          case 2 -> {
            if (wire != 2) {
              skip(buf, wire);
            } else {
              trainer = readTrainer(readLengthDelimited(buf));
            }
          }
          case 3 -> {
            if (wire != 2) {
              skip(buf, wire);
            } else {
              normalizer = readNormalizer(readLengthDelimited(buf));
            }
          }
          default -> skip(buf, wire);
        }
      }
      if (pieces.isEmpty()) {
        throw new ModelLoadException("SentencePiece model has no pieces");
      }
      return Parsed.of(pieces, trainer, normalizer);
    } catch (IOException e) {
      throw new ModelLoadException("failed to parse tokenizer.model", e);
    }
  }

  private static Piece readPiece(final ByteBuffer buf) throws IOException {
    String piece = "";
    float score = 0f;
    int type = TYPE_NORMAL;
    while (buf.hasRemaining()) {
      long tag = readVarint(buf);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      switch (field) {
        case 1 -> {
          if (wire != 2) {
            skip(buf, wire);
          } else {
            piece = new String(readBytes(buf), UTF_8);
          }
        }
        case 2 -> {
          if (wire == 5) {
            score = Float.intBitsToFloat(buf.getInt());
          } else {
            skip(buf, wire);
          }
        }
        case 3 -> {
          if (wire == 0) {
            type = (int) readVarint(buf);
          } else {
            skip(buf, wire);
          }
        }
        default -> skip(buf, wire);
      }
    }
    return new Piece(piece, score, type);
  }

  private static Trainer readTrainer(final ByteBuffer buf) throws IOException {
    int modelType = UNIGRAM;
    boolean byteFallback = false;
    int unkId = 0;
    while (buf.hasRemaining()) {
      long tag = readVarint(buf);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      if (wire == 0) {
        long value = readVarint(buf);
        switch (field) {
          case 3 -> modelType = (int) value;
          case 35 -> byteFallback = value != 0;
          case 40 -> unkId = (int) value;
          default -> {
          }
        }
      } else {
        skip(buf, wire);
      }
    }
    return new Trainer(modelType, byteFallback, unkId);
  }

  private static NormalizerSpec readNormalizer(final ByteBuffer buf) throws IOException {
    byte[] charsmap = new byte[0];
    boolean addDummyPrefix = true;
    boolean escapeWhitespaces = true;
    while (buf.hasRemaining()) {
      long tag = readVarint(buf);
      int field = (int) (tag >>> 3);
      int wire = (int) (tag & 7L);
      switch (field) {
        case 2 -> {
          if (wire != 2) {
            skip(buf, wire);
          } else {
            charsmap = readBytes(buf);
          }
        }
        case 3 -> addDummyPrefix = wire == 0 && readVarint(buf) != 0;
        case 5 -> escapeWhitespaces = wire == 0 && readVarint(buf) != 0;
        default -> skip(buf, wire);
      }
    }
    return new NormalizerSpec(charsmap, addDummyPrefix, escapeWhitespaces);
  }

  private static ByteBuffer readLengthDelimited(final ByteBuffer buf) throws IOException {
    long len = readVarint(buf);
    if (len < 0 || len > buf.remaining()) {
      throw new IOException("invalid length-delimited field in tokenizer.model");
    }
    int start = buf.position();
    ByteBuffer slice = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    slice.position(start);
    slice.limit(start + (int) len);
    buf.position(start + (int) len);
    return slice.slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static byte[] readBytes(final ByteBuffer buf) throws IOException {
    long len = readVarint(buf);
    if (len < 0 || len > buf.remaining()) {
      throw new IOException("invalid bytes field in tokenizer.model");
    }
    byte[] out = new byte[(int) len];
    buf.get(out);
    return out;
  }

  private static void skip(final ByteBuffer buf, final int wire) throws IOException {
    switch (wire) {
      case 0 -> readVarint(buf);
      case 1 -> {
        if (buf.remaining() < 8) {
          throw new IOException("truncated 64-bit field in tokenizer.model");
        }
        buf.position(buf.position() + 8);
      }
      case 2 -> readLengthDelimited(buf);
      case 5 -> {
        if (buf.remaining() < 4) {
          throw new IOException("truncated 32-bit field in tokenizer.model");
        }
        buf.position(buf.position() + 4);
      }
      default -> throw new IOException("unsupported protobuf wire type " + wire);
    }
  }

  private static long readVarint(final ByteBuffer buf) throws IOException {
    long result = 0L;
    int shift = 0;
    while (true) {
      if (!buf.hasRemaining()) {
        throw new IOException("truncated varint in tokenizer.model");
      }
      int b = buf.get() & 0xFF;
      result |= (long) (b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        return result;
      }
      shift += 7;
      if (shift > 63) {
        throw new IOException("varint too long in tokenizer.model");
      }
    }
  }

  @SuppressWarnings("ArrayRecordComponent")
  record Parsed(
    int modelType,
    Map<String, Integer> vocab,
    Map<Integer, Float> scores,
    Set<String> added,
    Set<String> specials,
    int unkId,
    boolean byteFallback,
    boolean addDummyPrefix,
    byte[] charsmap
  ) {
    private static Parsed of(
      final List<Piece> pieces,
      final Trainer trainer,
      final NormalizerSpec normalizer
    ) {
      Map<String, Integer> vocab = new LinkedHashMap<>(pieces.size() * 2);
      Map<Integer, Float> scores = new HashMap<>(pieces.size() * 2);
      Set<String> added = new HashSet<>();
      Set<String> specials = new HashSet<>();
      int unkId = trainer.unkId;
      for (int i = 0; i < pieces.size(); i++) {
        Piece piece = pieces.get(i);
        if (piece.text.isEmpty()) {
          continue;
        }
        vocab.put(piece.text, i);
        scores.put(i, piece.score);
        if (piece.type == TYPE_CONTROL || piece.type == TYPE_USER_DEFINED) {
          added.add(piece.text);
          specials.add(piece.text);
        }
        if (piece.type == TYPE_UNKNOWN) {
          unkId = i;
        }
      }
      return new Parsed(
        trainer.modelType,
        vocab,
        scores,
        added,
        specials,
        unkId,
        trainer.byteFallback,
        normalizer.addDummyPrefix,
        normalizer.charsmap);
    }
  }

  private record Piece(String text, float score, int type) {
  }

  private record Trainer(int modelType, boolean byteFallback, int unkId) {
    Trainer() {
      this(UNIGRAM, false, 0);
    }
  }

  @SuppressWarnings("ArrayRecordComponent")
  private record NormalizerSpec(byte[] charsmap, boolean addDummyPrefix,
                                boolean escapeWhitespaces) {
    NormalizerSpec() {
      this(new byte[0], true, true);
    }
  }
}
