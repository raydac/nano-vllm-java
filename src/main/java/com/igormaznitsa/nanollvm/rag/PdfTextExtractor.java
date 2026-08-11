package com.igormaznitsa.nanollvm.rag;

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_16BE;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * Pure-Java plain-text extractor for RAG corpus loading of {@code .pdf} files.
 *
 * <p>Scans page content streams ({@code FlateDecode} / {@code ASCII85Decode} / {@code ASCIIHexDecode}),
 * pulls {@code Tj} / {@code TJ} / {@code '} / {@code "} operands, and maps CID/OCR text through
 * embedded {@code ToUnicode} CMaps when present. Glyph pieces are concatenated (spaces kept) so
 * OCR text layers stay searchable. Not a full PDF engine — no image OCR, encryption, or exotic
 * filters.
 */
public final class PdfTextExtractor {

  private static final byte[] PDF_HEADER = {'%', 'P', 'D', 'F'};
  private static final Pattern LENGTH = Pattern.compile("/Length\\s+(\\d+)(?!\\s+\\d+\\s+R)");
  private static final Pattern FILTER_BLOCK = Pattern.compile("/Filter\\s*(/\\w+|\\[[^\\]]*])");
  private static final Pattern FILTER_NAME = Pattern.compile("/(\\w+)");
  private static final Pattern BF_CHAR = Pattern.compile(
    "(\\d+)\\s+beginbfchar(.*?)endbfchar", Pattern.DOTALL);
  private static final Pattern BF_RANGE = Pattern.compile(
    "(\\d+)\\s+beginbfrange(.*?)endbfrange", Pattern.DOTALL);
  private static final Pattern CODE_SPACE = Pattern.compile(
    "begincodespacerange(.*?)endcodespacerange", Pattern.DOTALL);
  private static final Pattern HEX_TOKEN = Pattern.compile("<([0-9A-Fa-f]+)>");

  private PdfTextExtractor() {
  }

  /**
   * {@code true} when the path's file name ends with {@code .pdf} (case-insensitive).
   */
  public static boolean isPdf(final Path path) {
    requireNonNull(path, "path");
    Path name = path.getFileName();
    return name != null && name.toString().toLowerCase(Locale.ROOT).endsWith(".pdf");
  }

  /**
   * Reads {@code pdf} and returns extractable text in stream order.
   *
   * @throws UncheckedIOException if the file cannot be read or is not a PDF
   */
  public static String extract(final Path pdf) {
    return extract(pdf, ResourceLimits.current());
  }

  /**
   * Reads {@code pdf} under {@code limits} (file size and inflate caps).
   */
  public static String extract(final Path pdf, final ResourceLimits limits) {
    requireNonNull(pdf, "pdf");
    requireNonNull(limits, "limits");
    Path path = pdf.toAbsolutePath().normalize();
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException("not a regular file: " + path);
    }
    try {
      long size = Files.size(path);
      if (size > limits.maxFileBytes()) {
        throw new IllegalArgumentException(
          "PDF exceeds maxFileBytes (" + limits.maxFileBytes() + "): " + path);
      }
      return extract(Files.readAllBytes(path), limits);
    } catch (IOException e) {
      throw new UncheckedIOException("failed to read PDF " + path, e);
    }
  }

  /**
   * Consumes {@code in} fully and returns extractable text in stream order.
   *
   * @throws UncheckedIOException if the stream cannot be read or is not a PDF
   */
  public static String extract(final InputStream in) {
    return extract(in, ResourceLimits.current());
  }

  public static String extract(final InputStream in, final ResourceLimits limits) {
    requireNonNull(in, "in");
    requireNonNull(limits, "limits");
    try {
      return extract(in.readAllBytes(), limits);
    } catch (IOException e) {
      throw new UncheckedIOException("failed to read PDF stream", e);
    }
  }

  /**
   * Parses {@code pdfBytes} and returns extractable text in stream order.
   *
   * @throws UncheckedIOException if the bytes are not a PDF or cannot be decoded
   */
  public static String extract(final byte[] pdfBytes) {
    return extract(pdfBytes, ResourceLimits.current());
  }

  public static String extract(final byte[] pdfBytes, final ResourceLimits limits) {
    requireNonNull(pdfBytes, "pdfBytes");
    requireNonNull(limits, "limits");
    if (pdfBytes.length > limits.maxFileBytes()) {
      throw new IllegalArgumentException(
        "PDF bytes exceed maxFileBytes (" + limits.maxFileBytes() + ")");
    }
    requirePdfHeader(pdfBytes);
    return joinFragments(collectTextFragments(pdfBytes, limits)).strip();
  }

  private static void requirePdfHeader(final byte[] pdfBytes) {
    if (pdfBytes.length < 5
      || pdfBytes[0] != PDF_HEADER[0]
      || pdfBytes[1] != PDF_HEADER[1]
      || pdfBytes[2] != PDF_HEADER[2]
      || pdfBytes[3] != PDF_HEADER[3]) {
      throw new UncheckedIOException("not a PDF document", new IOException("missing %PDF header"));
    }
  }

  /**
   * Concatenate glyph/`Tj` pieces. OCR layers emit one character (or a space) per operator —
   * joining with newlines destroyed words ({@code Ш к о л а} instead of {@code Школа}).
   */
  private static String joinFragments(final List<String> fragments) {
    return repairGluedWords(String.join("", fragments));
  }

  /**
   * OCR layers often omit spaces at glyph boundaries; insert breaks before capitals / digits.
   */
  private static String repairGluedWords(final String text) {
    return text
      .replaceAll("(?<=\\p{Ll})(?=\\p{Lu})", " ")
      .replaceAll("(?<=\\p{N})(?=\\p{L})", " ")
      .replaceAll("(?<=\\p{L})(?=\\p{N})", " ");
  }

  private static List<String> collectTextFragments(final byte[] pdf, final ResourceLimits limits) {
    List<byte[]> decodedStreams = findStreams(pdf).stream()
      .map(stream -> decodeStream(stream, limits))
      .filter(data -> data != null && data.length > 0)
      .toList();

    List<ToUnicode> cmaps = decodedStreams.stream()
      .filter(PdfTextExtractor::looksLikeCMap)
      .map(data -> parseToUnicode(data, limits))
      .filter(cmap -> !cmap.isEmpty())
      .toList();

    List<String> fragments = new ArrayList<>();
    for (byte[] decoded : decodedStreams) {
      if (looksLikeCMap(decoded) || !looksLikeContentStream(decoded)) {
        continue;
      }
      int before = fragments.size();
      extractShowsFromContent(decoded, fragments, cmaps);
      if (fragments.size() > before) {
        fragments.add("\n");
      }
    }
    return fragments;
  }

  private static List<StreamPayload> findStreams(final byte[] pdf) {
    List<StreamPayload> streams = new ArrayList<>();
    int cursor = 0;
    while (cursor < pdf.length) {
      int streamKw = indexOfKeyword(pdf, cursor, "stream");
      if (streamKw < 0) {
        break;
      }

      DictInfo dict = parseDictBefore(pdf, streamKw);
      int dataStart = skipEolAfterStreamKeyword(pdf, streamKw + "stream".length());
      int dataEnd = resolveStreamEnd(pdf, dataStart, dict.length());
      if (dataEnd < dataStart || dataEnd > pdf.length) {
        cursor = streamKw + "stream".length();
        continue;
      }

      streams.add(new StreamPayload(copyOfRange(pdf, dataStart, dataEnd), dict.filters()));
      int endstream = indexOfKeyword(pdf, dataEnd, "endstream");
      cursor = endstream >= 0 ? endstream + "endstream".length() : dataEnd + 1;
    }
    return streams;
  }

  private static int resolveStreamEnd(final byte[] pdf, final int dataStart, final int length) {
    if (length >= 0) {
      int end = dataStart + length;
      return indexOfKeyword(pdf, end, "endstream") >= 0 ? end : -1;
    }
    return indexOfKeyword(pdf, dataStart, "endstream");
  }

  private static byte[] decodeStream(final StreamPayload stream, final ResourceLimits limits) {
    byte[] data = stream.data();
    for (String filter : stream.filters()) {
      data = applyFilter(data, filter, limits);
      if (data == null) {
        return null;
      }
    }
    return data;
  }

  private static byte[] applyFilter(
    final byte[] data,
    final String filter,
    final ResourceLimits limits
  ) {
    return switch (filter) {
      case "", "Identity" -> data;
      case "FlateDecode", "Fl" -> inflate(data, limits);
      case "ASCII85Decode", "A85" -> decodeAscii85(data);
      case "ASCIIHexDecode", "AHx" -> decodeAsciiHex(data);
      default -> null;
    };
  }

  private static byte[] inflate(final byte[] compressed, final ResourceLimits limits) {
    byte[] zlib = inflateWith(compressed, false, limits);
    if (zlib != null && zlib.length > 0) {
      return zlib;
    }
    return inflateWith(compressed, true, limits);
  }

  private static byte[] inflateWith(
    final byte[] compressed,
    final boolean nowrap,
    final ResourceLimits limits
  ) {
    Inflater inflater = new Inflater(nowrap);
    try {
      inflater.setInput(compressed);
      ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(64, compressed.length * 2));
      byte[] buffer = new byte[8192];
      while (!inflater.finished()) {
        int n = inflater.inflate(buffer);
        if (n == 0) {
          if (inflater.needsInput() || inflater.needsDictionary()) {
            break;
          }
        } else {
          if (out.size() + (long) n > limits.maxPdfInflateBytes()) {
            throw new UncheckedIOException(
              "PDF FlateDecode exceeds maxPdfInflateBytes (" + limits.maxPdfInflateBytes() + ")",
              new IOException("inflate budget exceeded"));
          }
          out.write(buffer, 0, n);
        }
      }
      return out.toByteArray();
    } catch (DataFormatException e) {
      return null;
    } finally {
      inflater.end();
    }
  }

  private static byte[] decodeAsciiHex(final byte[] data) {
    ByteArrayOutputStream out = new ByteArrayOutputStream(data.length / 2);
    int nibble = -1;
    for (byte value : data) {
      if (value == '>') {
        break;
      }
      if (isWhitespace(value)) {
        continue;
      }
      if (!isHexDigit(value)) {
        return null;
      }
      int digit = hexValue((char) (value & 0xFF));
      if (nibble < 0) {
        nibble = digit;
      } else {
        out.write((nibble << 4) | digit);
        nibble = -1;
      }
    }
    if (nibble >= 0) {
      out.write(nibble << 4);
    }
    return out.toByteArray();
  }

  private static byte[] decodeAscii85(final byte[] data) {
    int start = 0;
    int end = data.length;
    if (end >= 2 && data[0] == '<' && data[1] == '~') {
      start = 2;
    }
    for (int i = start; i + 1 < end; i++) {
      if (data[i] == '~' && data[i + 1] == '>') {
        end = i;
        break;
      }
    }

    ByteArrayOutputStream out = new ByteArrayOutputStream(data.length);
    long value = 0;
    int count = 0;
    for (int i = start; i < end; i++) {
      byte b = data[i];
      if (isWhitespace(b)) {
        continue;
      }
      if (b == 'z') {
        if (count != 0) {
          return null;
        }
        out.write(0);
        out.write(0);
        out.write(0);
        out.write(0);
        continue;
      }
      if (b < '!' || b > 'u') {
        return null;
      }
      value = value * 85 + (b - '!');
      count++;
      if (count == 5) {
        writeAscii85Word(out, value, 4);
        value = 0;
        count = 0;
      }
    }
    if (count > 0) {
      for (int i = count; i < 5; i++) {
        value = value * 85 + 84;
      }
      writeAscii85Word(out, value, count - 1);
    }
    return out.toByteArray();
  }

  private static void writeAscii85Word(
    final ByteArrayOutputStream out,
    final long value,
    final int bytes
  ) {
    out.write((int) ((value >> 24) & 0xFF));
    if (bytes > 1) {
      out.write((int) ((value >> 16) & 0xFF));
    }
    if (bytes > 2) {
      out.write((int) ((value >> 8) & 0xFF));
    }
    if (bytes > 3) {
      out.write((int) (value & 0xFF));
    }
  }

  private static boolean looksLikeContentStream(final byte[] data) {
    String sample = new String(data, 0, Math.min(data.length, 8192), ISO_8859_1);
    return sample.contains("Tj") || sample.contains("TJ") || sample.contains("BT");
  }

  private static boolean looksLikeCMap(final byte[] data) {
    String sample = new String(data, 0, Math.min(data.length, 8192), ISO_8859_1);
    return sample.contains("begincmap")
      && (sample.contains("beginbfchar") || sample.contains("beginbfrange"));
  }

  private static ToUnicode parseToUnicode(final byte[] data, final ResourceLimits limits) {
    String cmap = new String(data, ISO_8859_1);
    ToUnicode toUnicode = new ToUnicode(limits);
    Matcher space = CODE_SPACE.matcher(cmap);
    if (space.find()) {
      Matcher hex = HEX_TOKEN.matcher(space.group(1));
      while (hex.find()) {
        toUnicode.noteCodeWidth(hex.group(1).length() / 2);
      }
    }
    Matcher chars = BF_CHAR.matcher(cmap);
    while (chars.find()) {
      parseBfCharBlock(chars.group(2), toUnicode);
    }
    Matcher ranges = BF_RANGE.matcher(cmap);
    while (ranges.find()) {
      parseBfRangeBlock(ranges.group(2), toUnicode);
    }
    return toUnicode;
  }

  private static void parseBfCharBlock(final String block, final ToUnicode toUnicode) {
    List<String> tokens = hexTokens(block);
    for (int i = 0; i + 1 < tokens.size(); i += 2) {
      toUnicode.put(parseHexCode(tokens.get(i)), decodeUtf16Hex(tokens.get(i + 1)));
      toUnicode.noteCodeWidth(tokens.get(i).length() / 2);
    }
  }

  private static void parseBfRangeBlock(final String block, final ToUnicode toUnicode) {
    Matcher line = Pattern.compile(
        "<([0-9A-Fa-f]+)>\\s*<([0-9A-Fa-f]+)>\\s*(?:<([0-9A-Fa-f]+)>|\\[(.*?)])",
        Pattern.DOTALL)
      .matcher(block);
    while (line.find()) {
      int from = parseHexCode(line.group(1));
      int to = parseHexCode(line.group(2));
      toUnicode.noteCodeWidth(line.group(1).length() / 2);
      if (line.group(3) != null) {
        int span = to - from + 1;
        if (span < 0 || span > toUnicode.maxRangeSpan()) {
          throw new UncheckedIOException(
            "PDF ToUnicode bfRange span exceeds maxCmapRangeSpan ("
              + toUnicode.maxRangeSpan() + ")",
            new IOException("cmap range too large"));
        }
        int dst = parseHexCode(line.group(3));
        for (int code = from; code <= to; code++) {
          toUnicode.put(code, new String(Character.toChars(dst + (code - from))));
        }
      } else if (line.group(4) != null) {
        List<String> values = hexTokens(line.group(4));
        for (int i = 0; i < values.size() && from + i <= to; i++) {
          toUnicode.put(from + i, decodeUtf16Hex(values.get(i)));
        }
      }
    }
  }

  private static List<String> hexTokens(final String block) {
    List<String> tokens = new ArrayList<>();
    Matcher matcher = HEX_TOKEN.matcher(block);
    while (matcher.find()) {
      tokens.add(matcher.group(1));
    }
    return tokens;
  }

  private static int parseHexCode(final String hex) {
    return Integer.parseInt(hex, 16);
  }

  private static String decodeUtf16Hex(final String hex) {
    String padded = (hex.length() & 1) == 1 ? hex + "0" : hex;
    byte[] bytes = new byte[padded.length() / 2];
    for (int i = 0; i < bytes.length; i++) {
      bytes[i] = (byte) Integer.parseInt(padded.substring(i * 2, i * 2 + 2), 16);
    }
    return new String(bytes, UTF_16BE);
  }

  private static void extractShowsFromContent(
    final byte[] data,
    final List<String> fragments,
    final List<ToUnicode> cmaps
  ) {
    int i = 0;
    while (i < data.length) {
      byte b = data[i];
      if (b == '(') {
        ByteString lit = readLiteralString(data, i);
        int op = skipWhitespace(data, lit.end());
        if (isOperator(data, op, "Tj") || isOperator(data, op, "'")) {
          if (data[op] == '\'') {
            appendFragment(fragments, "\n");
          }
          appendFragment(fragments, decodeShow(lit.bytes(), cmaps));
          i = op + (data[op] == '\'' ? 1 : 2);
          continue;
        }
        if (isOperator(data, op, "\"")) {
          appendFragment(fragments, "\n");
          appendFragment(fragments, decodeShow(lit.bytes(), cmaps));
          i = op + 1;
          continue;
        }
        i = lit.end();
        continue;
      }
      if (b == '[') {
        int close = indexOfByte(data, i + 1, (byte) ']');
        if (close > i) {
          int op = skipWhitespace(data, close + 1);
          if (isOperator(data, op, "TJ")) {
            appendFragment(fragments, extractLiteralsFromArray(data, i + 1, close, cmaps));
            i = op + 2;
            continue;
          }
        }
      }
      if (b == '<' && i + 1 < data.length && data[i + 1] != '<') {
        ByteString hex = readHexString(data, i);
        if (hex != null) {
          int op = skipWhitespace(data, hex.end());
          if (isOperator(data, op, "Tj") || isOperator(data, op, "'")) {
            if (data[op] == '\'') {
              appendFragment(fragments, "\n");
            }
            appendFragment(fragments, decodeShow(hex.bytes(), cmaps));
            i = op + (data[op] == '\'' ? 1 : 2);
            continue;
          }
        }
      }
      i++;
    }
  }

  private static String extractLiteralsFromArray(
    final byte[] data,
    final int from,
    final int to,
    final List<ToUnicode> cmaps
  ) {
    StringBuilder piece = new StringBuilder();
    int i = from;
    while (i < to) {
      if (data[i] == '(') {
        ByteString lit = readLiteralString(data, i);
        piece.append(decodeShow(lit.bytes(), cmaps));
        i = lit.end();
      } else if (data[i] == '<' && i + 1 < to && data[i + 1] != '<') {
        ByteString hex = readHexString(data, i);
        if (hex != null) {
          piece.append(decodeShow(hex.bytes(), cmaps));
          i = hex.end();
          continue;
        }
      }
      i++;
    }
    return piece.toString();
  }

  private static String decodeShow(final byte[] raw, final List<ToUnicode> cmaps) {
    if (raw.length == 0) {
      return "";
    }
    if (cmaps.isEmpty()) {
      return new String(raw, ISO_8859_1);
    }

    String best = "";
    int bestScore = -1;
    for (ToUnicode cmap : cmaps) {
      String mapped = cmap.mapBytes(raw);
      int score = printableScore(mapped);
      if (score > bestScore) {
        bestScore = score;
        best = mapped;
      }
    }

    String latin = new String(raw, ISO_8859_1);
    return printableScore(latin) > bestScore && mostlyPrintableLatin1(latin) ? latin : best;
  }

  private static int printableScore(final String text) {
    int score = 0;
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      if (Character.isLetterOrDigit(c) || c == ' ' || c == '\n' || c == '\t') {
        score++;
      } else if (c > 0x20 && c < 0x7F) {
        score++;
      }
    }
    return score;
  }

  private static boolean mostlyPrintableLatin1(final String text) {
    if (text.isEmpty()) {
      return false;
    }
    int printable = 0;
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      if (c == '\n' || c == '\r' || c == '\t' || (c >= 0x20 && c <= 0x7E) || c >= 0xA0) {
        printable++;
      }
    }
    return printable * 4 >= text.length() * 3;
  }

  private static void appendFragment(final List<String> fragments, final String text) {
    if (text == null || text.isEmpty()) {
      return;
    }
    // Keep whitespace-only pieces — OCR layers use them as word separators between glyphs
    fragments.add(text);
  }

  private static ByteString readLiteralString(final byte[] data, final int openParen) {
    ByteArrayOutputStream text = new ByteArrayOutputStream();
    int i = openParen + 1;
    int depth = 1;
    while (i < data.length) {
      byte b = data[i++];
      if (b == '\\' && i < data.length) {
        text.write(decodeEscape(data, i));
        i += escapeAdvance(data, i);
        continue;
      }
      if (b == '(') {
        depth++;
        text.write('(');
        continue;
      }
      if (b == ')') {
        depth--;
        if (depth == 0) {
          return new ByteString(text.toByteArray(), i);
        }
        text.write(')');
        continue;
      }
      text.write(b);
    }
    return new ByteString(text.toByteArray(), data.length);
  }

  private static int decodeEscape(final byte[] data, final int escapeByte) {
    byte e = data[escapeByte];
    return switch (e) {
      case 'n' -> '\n';
      case 'r' -> '\r';
      case 't' -> '\t';
      case 'b' -> '\b';
      case 'f' -> '\f';
      case '(', ')', '\\' -> e & 0xFF;
      default -> {
        if (e >= '0' && e <= '7') {
          yield readOctal(data, escapeByte);
        }
        yield e & 0xFF;
      }
    };
  }

  private static int escapeAdvance(final byte[] data, final int escapeByte) {
    byte e = data[escapeByte];
    if (e >= '0' && e <= '7') {
      int digits = 1;
      if (escapeByte + 1 < data.length && isOctal(data[escapeByte + 1])) {
        digits++;
      }
      if (digits == 2 && escapeByte + 2 < data.length && isOctal(data[escapeByte + 2])) {
        digits++;
      }
      return digits;
    }
    return 1;
  }

  private static int readOctal(final byte[] data, final int start) {
    int value = 0;
    int i = start;
    for (int n = 0; n < 3 && i < data.length && isOctal(data[i]); n++, i++) {
      value = (value << 3) + (data[i] - '0');
    }
    return value;
  }

  private static boolean isOctal(final byte b) {
    return b >= '0' && b <= '7';
  }

  /**
   * Reads a PDF hex string {@code <...>}. Returns {@code null} when the brackets do not form a
   * valid hex string so callers can skip the {@code <} without failing the file.
   */
  private static ByteString readHexString(final byte[] data, final int openAngle) {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    int i = openAngle + 1;
    int nibble = -1;
    while (i < data.length && data[i] != '>') {
      byte b = data[i++];
      if (isWhitespace(b)) {
        continue;
      }
      if (!isHexDigit(b)) {
        return null;
      }
      int digit = hexValue((char) (b & 0xFF));
      if (nibble < 0) {
        nibble = digit;
      } else {
        bytes.write((nibble << 4) | digit);
        nibble = -1;
      }
    }
    if (i >= data.length || data[i] != '>') {
      return null;
    }
    i++;
    if (nibble >= 0) {
      bytes.write(nibble << 4);
    }
    return new ByteString(bytes.toByteArray(), i);
  }

  private static boolean isHexDigit(final byte b) {
    return (b >= '0' && b <= '9')
      || (b >= 'A' && b <= 'F')
      || (b >= 'a' && b <= 'f');
  }

  private static int hexValue(final char c) {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'A' && c <= 'F') {
      return 10 + (c - 'A');
    }
    return 10 + (c - 'a');
  }

  private static DictInfo parseDictBefore(final byte[] pdf, final int streamAt) {
    int end = streamAt;
    while (end > 0 && isWhitespace(pdf[end - 1])) {
      end--;
    }
    if (end < 2 || pdf[end - 2] != '>' || pdf[end - 1] != '>') {
      return DictInfo.EMPTY;
    }
    int depth = 1;
    int i = end - 2;
    while (i > 0 && depth > 0) {
      i--;
      if (pdf[i] == '>' && pdf[i + 1] == '>') {
        depth++;
      } else if (pdf[i] == '<' && pdf[i + 1] == '<') {
        depth--;
      }
    }
    if (depth != 0) {
      return DictInfo.EMPTY;
    }
    String dict = new String(pdf, i, end - i, ISO_8859_1);
    return new DictInfo(parseLength(dict), parseFilters(dict));
  }

  private static int parseLength(final String dict) {
    Matcher matcher = LENGTH.matcher(dict);
    return matcher.find() ? Integer.parseInt(matcher.group(1)) : -1;
  }

  private static List<String> parseFilters(final String dict) {
    Matcher block = FILTER_BLOCK.matcher(dict);
    if (!block.find()) {
      return List.of();
    }
    List<String> filters = new ArrayList<>();
    Matcher names = FILTER_NAME.matcher(block.group(1));
    while (names.find()) {
      filters.add(names.group(1));
    }
    return List.copyOf(filters);
  }

  private static int indexOfKeyword(final byte[] data, final int from, final String keyword) {
    byte[] needle = keyword.getBytes(ISO_8859_1);
    outer:
    for (int i = from; i <= data.length - needle.length; i++) {
      for (int j = 0; j < needle.length; j++) {
        if (data[i + j] != needle[j]) {
          continue outer;
        }
      }
      if (isKeywordBoundary(data, i - 1) && isKeywordBoundary(data, i + needle.length)) {
        return i;
      }
    }
    return -1;
  }

  private static boolean isKeywordBoundary(final byte[] data, final int index) {
    if (index < 0 || index >= data.length) {
      return true;
    }
    byte b = data[index];
    return isWhitespace(b) || b == '<' || b == '>' || b == '[' || b == ']' || b == '/' || b == '(';
  }

  private static int skipEolAfterStreamKeyword(final byte[] data, final int afterKeyword) {
    int i = afterKeyword;
    if (i < data.length && data[i] == '\r') {
      i++;
      if (i < data.length && data[i] == '\n') {
        i++;
      }
      return i;
    }
    if (i < data.length && data[i] == '\n') {
      return i + 1;
    }
    return i;
  }

  private static int skipWhitespace(final byte[] data, final int from) {
    int i = from;
    while (i < data.length && isWhitespace(data[i])) {
      i++;
    }
    return i;
  }

  private static boolean isWhitespace(final byte b) {
    return b == 0 || b == '\t' || b == '\n' || b == '\f' || b == '\r' || b == ' ';
  }

  private static boolean isOperator(final byte[] data, final int from, final String op) {
    byte[] needle = op.getBytes(ISO_8859_1);
    if (from + needle.length > data.length) {
      return false;
    }
    for (int j = 0; j < needle.length; j++) {
      if (data[from + j] != needle[j]) {
        return false;
      }
    }
    int after = from + needle.length;
    return after >= data.length || isWhitespace(data[after]) || !isNameChar(data[after]);
  }

  private static boolean isNameChar(final byte b) {
    return (b >= 'A' && b <= 'Z')
      || (b >= 'a' && b <= 'z')
      || (b >= '0' && b <= '9');
  }

  private static int indexOfByte(final byte[] data, final int from, final byte target) {
    for (int i = from; i < data.length; i++) {
      if (data[i] == target) {
        return i;
      }
    }
    return -1;
  }

  private static byte[] copyOfRange(final byte[] data, final int from, final int to) {
    byte[] copy = new byte[to - from];
    System.arraycopy(data, from, copy, 0, copy.length);
    return copy;
  }

  private static final class ToUnicode {
    private final Map<Integer, String> map = new HashMap<>();
    private final int maxEntries;
    private final int maxRangeSpan;
    private int codeWidth = 1;

    private ToUnicode(final ResourceLimits limits) {
      this.maxEntries = limits.maxCmapEntries();
      this.maxRangeSpan = limits.maxCmapRangeSpan();
    }

    private static int bytesToCode(final byte[] raw, final int offset, final int width) {
      int code = 0;
      for (int i = 0; i < width; i++) {
        code = (code << 8) | (raw[offset + i] & 0xFF);
      }
      return code;
    }

    int maxRangeSpan() {
      return this.maxRangeSpan;
    }

    void noteCodeWidth(final int width) {
      if (width > this.codeWidth) {
        this.codeWidth = width;
      }
    }

    void put(final int code, final String value) {
      if (value != null && !value.isEmpty()) {
        if (this.map.size() >= this.maxEntries && !this.map.containsKey(code)) {
          throw new UncheckedIOException(
            "PDF ToUnicode map exceeds maxCmapEntries (" + this.maxEntries + ")",
            new IOException("cmap too large"));
        }
        this.map.put(code, value);
      }
    }

    boolean isEmpty() {
      return this.map.isEmpty();
    }

    String mapBytes(final byte[] raw) {
      StringBuilder out = new StringBuilder(raw.length);
      int i = 0;
      while (i < raw.length) {
        String mapped = null;
        int used = 0;
        int max = Math.min(this.codeWidth, raw.length - i);
        for (int width = max; width >= 1; width--) {
          String value = this.map.get(bytesToCode(raw, i, width));
          if (value != null) {
            mapped = value;
            used = width;
            break;
          }
        }
        if (mapped != null) {
          out.append(mapped);
          i += used;
        } else {
          i += this.codeWidth > 1 && i + this.codeWidth <= raw.length ? this.codeWidth : 1;
        }
      }
      return out.toString();
    }
  }

  private record StreamPayload(byte[] data, List<String> filters) {
  }

  private record DictInfo(int length, List<String> filters) {
    static final DictInfo EMPTY = new DictInfo(-1, List.of());
  }

  private record ByteString(byte[] bytes, int end) {
  }
}
