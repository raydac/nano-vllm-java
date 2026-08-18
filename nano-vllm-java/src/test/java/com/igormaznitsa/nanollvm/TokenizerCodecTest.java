package com.igormaznitsa.nanollvm;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class TokenizerCodecTest {

  private static byte[] unigramModel(final byte[]... pieces) {
    try {
      ByteArrayOutputStream model = new ByteArrayOutputStream();
      for (byte[] piece : pieces) {
        writeLengthDelimited(model, 1, piece);
      }
      ByteArrayOutputStream trainer = new ByteArrayOutputStream();
      writeVarint(trainer, (3 << 3));
      writeVarint(trainer, 1);
      writeLengthDelimited(model, 2, trainer.toByteArray());
      ByteArrayOutputStream normalizer = new ByteArrayOutputStream();
      writeVarint(normalizer, (3 << 3));
      writeVarint(normalizer, 1);
      writeVarint(normalizer, (5 << 3));
      writeVarint(normalizer, 1);
      writeLengthDelimited(model, 3, normalizer.toByteArray());
      return model.toByteArray();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static byte[] piece(final String text, final float score, final int type) {
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      writeLengthDelimited(out, 1, text.getBytes(UTF_8));
      out.write((2 << 3) | 5);
      ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      buf.putFloat(score);
      out.write(buf.array());
      writeVarint(out, 3 << 3);
      writeVarint(out, type);
      return out.toByteArray();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static void writeLengthDelimited(
    final ByteArrayOutputStream out,
    final int field,
    final byte[] bytes
  ) throws IOException {
    writeVarint(out, (field << 3) | 2);
    writeVarint(out, bytes.length);
    out.write(bytes);
  }

  private static void writeVarint(final ByteArrayOutputStream out, final int value)
    throws IOException {
    int current = value;
    while ((current & ~0x7F) != 0) {
      out.write((current & 0x7F) | 0x80);
      current >>>= 7;
    }
    out.write(current);
  }

  @Test
  void wordLevelSplitsOnWhitespace() {
    Tokenizer tok = Tokenizer.fromJsonDocuments("""
        {
          "model": {
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "</s>": 3}
          }
        }
        """,
      "{\"eos_token\":\"</s>\"}",
      null,
      "{\"vocab_size\":4}");
    assertEquals(List.of(1, 2), tok.encode("hello world"));
    assertEquals("hello world", tok.decode(List.of(1, 2)));
    assertEquals(List.of(0), tok.encode("missing"));
  }

  @Test
  void hfWordPieceUsesContinuingPrefix() {
    Tokenizer tok = Tokenizer.fromJsonDocuments("""
        {
          "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "vocab": {
              "[UNK]": 0,
              "[CLS]": 1,
              "[SEP]": 2,
              "hello": 3,
              "world": 4,
              "##ing": 5,
              "un": 6,
              "</s>": 7
            }
          },
          "added_tokens": [
            {"id": 1, "content": "[CLS]", "special": true},
            {"id": 2, "content": "[SEP]", "special": true}
          ]
        }
        """,
      "{\"eos_token\":\"</s>\",\"unk_token\":\"[UNK]\"}",
      null,
      "{\"vocab_size\":8}");
    assertEquals(List.of(3, 4), tok.encode("hello world"));
    assertEquals("hello world", tok.decode(List.of(3, 4)));
    assertEquals(List.of(6, 5), tok.encode("uning"));
    assertEquals("uning", tok.decode(List.of(6, 5)));
  }

  @Test
  void bertNormalizerLowercases() {
    Tokenizer tok = Tokenizer.fromJsonDocuments("""
        {
          "normalizer": {"type": "BertNormalizer", "lowercase": true, "clean_text": false,
            "handle_chinese_chars": false, "strip_accents": false},
          "model": {
            "type": "WordPiece",
            "vocab": {"[UNK]": 0, "hello": 1, "</s>": 2}
          }
        }
        """,
      "{\"eos_token\":\"</s>\"}",
      null,
      "{\"vocab_size\":3}");
    assertEquals(List.of(1), tok.encode("HELLO"));
  }

  @Test
  void replaceNormalizerCollapsesSpaces() {
    Tokenizer tok = Tokenizer.fromJsonDocuments("""
        {
          "normalizer": {
            "type": "Replace",
            "pattern": {"Regex": " {2,}"},
            "content": " "
          },
          "model": {
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "a": 1, "b": 2, "</s>": 3}
          }
        }
        """,
      "{\"eos_token\":\"</s>\"}",
      null,
      "{\"vocab_size\":4}");
    assertEquals(List.of(1, 2), tok.encode("a    b"));
  }

  @Test
  void sentencePieceUnigramLoadsFromProtobuf(@TempDir final Path dir) throws IOException {
    byte[] model = unigramModel(
      piece("<unk>", 0f, 2),
      piece("<s>", 0f, 3),
      piece("</s>", 0f, 3),
      piece("▁Hello", -1.0f, 1),
      piece("▁world", -1.2f, 1),
      piece("▁", -3.0f, 1));
    Files.write(dir.resolve("tokenizer.model"), model);
    Files.writeString(dir.resolve("tokenizer_config.json"),
      "{\"eos_token\":\"</s>\",\"bos_token\":\"<s>\"}");
    Tokenizer tok = Tokenizer.fromPretrained(dir);
    assertEquals(List.of(3, 4), tok.encode("Hello world"));
    assertEquals(" Hello world", tok.decode(List.of(3, 4)));
    assertEquals(Optional.of(1), tok.tokenId("<s>"));
  }

  @Test
  void sentencePieceFactoryMatchesDirectoryLoad() {
    byte[] model = unigramModel(
      piece("<unk>", 0f, 2),
      piece("</s>", 0f, 3),
      piece("▁hi", -0.5f, 1),
      piece("▁", -2.0f, 1));
    Tokenizer tok = Tokenizer.fromSentencePiece(model, "{\"eos_token\":\"</s>\"}", null);
    assertEquals(List.of(2), tok.encode("hi"));
  }

  @Test
  void e5UnigramCharsmapLeavesAsciiStable() {
    Path path = OptionalModelAssumptions.requireMultilingualE5Small();
    Tokenizer tok = Tokenizer.fromPretrained(path);
    List<Integer> ids = tok.encode("hello world");
    assertFalse(ids.isEmpty());
    assertTrue(tok.decode(ids).toLowerCase(java.util.Locale.ROOT).contains("hello"));
    assertFalse(tok.encode("Café ™").isEmpty());
  }
}
