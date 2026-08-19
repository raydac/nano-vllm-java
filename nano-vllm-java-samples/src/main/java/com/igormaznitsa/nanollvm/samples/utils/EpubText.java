package com.igormaznitsa.nanollvm.samples.utils;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.rag.RagResource;
import io.documentnode.epub4j.domain.Author;
import io.documentnode.epub4j.domain.Book;
import io.documentnode.epub4j.domain.MediaType;
import io.documentnode.epub4j.domain.Metadata;
import io.documentnode.epub4j.domain.Resource;
import io.documentnode.epub4j.epub.EpubReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.Charset;
import java.nio.charset.IllegalCharsetNameException;
import java.nio.charset.UnsupportedCharsetException;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Plain text from an EPUB via <a href="https://github.com/documentnode/epub4j">epub4j</a>
 * (Maven Central fork of <a href="https://github.com/psiegman/epublib">epublib</a>), for
 * {@link com.igormaznitsa.nanollvm.rag.RagTuner} extract / filter hooks.
 */
public final class EpubText {

  public static final String RUR_EPUB = "pg59112.epub";

  private static final Pattern SCRIPT_OR_STYLE = Pattern.compile(
    "(?is)<(script|style)\\b[^>]*>.*?</\\1>");
  private static final Pattern TAG = Pattern.compile("(?s)<[^>]+>");
  private static final Pattern NAMED_ENTITY = Pattern.compile("&([A-Za-z][A-Za-z0-9]+);");
  private static final Pattern NUMERIC_ENTITY = Pattern.compile("&#(x[0-9A-Fa-f]+|\\d+);");
  private static final Pattern WHITESPACE = Pattern.compile("\\s+");
  private static final Map<String, String> NAMED_ENTITIES = Map.ofEntries(
    Map.entry("amp", "&"),
    Map.entry("apos", "'"),
    Map.entry("gt", ">"),
    Map.entry("lt", "<"),
    Map.entry("nbsp", " "),
    Map.entry("quot", "\""),
    Map.entry("mdash", "—"),
    Map.entry("ndash", "–"),
    Map.entry("lsquo", "‘"),
    Map.entry("rsquo", "’"),
    Map.entry("ldquo", "“"),
    Map.entry("rdquo", "”"),
    Map.entry("hellip", "…"),
    Map.entry("copy", "©"),
    Map.entry("trade", "™"),
    Map.entry("bull", "•"),
    Map.entry("deg", "°")
  );

  private EpubText() {
  }

  public static boolean isEpub(final RagResource resource) {
    requireNonNull(resource, "resource");
    return resource.fileName().toLowerCase(Locale.ROOT).endsWith(".epub");
  }

  public static Optional<String> extract(final RagResource resource) {
    requireNonNull(resource, "resource");
    if (!isEpub(resource) || !resource.hasContent()) {
      return Optional.empty();
    }
    byte[] bytes = resource.content().orElseThrow();
    try {
      Book book = new EpubReader().readEpub(new ByteArrayInputStream(bytes));
      return Optional.of(plainText(book));
    } catch (IOException e) {
      throw new UncheckedIOException("failed to extract EPUB " + resource.source(), e);
    }
  }

  public static String normalizeWhitespace(final String text) {
    requireNonNull(text, "text");
    return WHITESPACE.matcher(text.strip()).replaceAll(" ");
  }

  private static String plainText(final Book book) {
    String chapters = htmlResources(book)
      .map(EpubText::htmlBody)
      .flatMap(Optional::stream)
      .collect(joining("\n"));
    String preface = metadataPreface(book);
    return preface.isEmpty() ? chapters : preface + "\n" + chapters;
  }

  private static String metadataPreface(final Book book) {
    Metadata metadata = book.getMetadata();
    String titles = metadata.getTitles().stream()
      .filter(title -> title != null && !title.isBlank())
      .map(String::strip)
      .collect(joining("; "));
    String authors = metadata.getAuthors().stream()
      .map(EpubText::formatAuthor)
      .filter(name -> !name.isBlank())
      .collect(joining(", "));
    StringBuilder preface = new StringBuilder();
    if (!titles.isEmpty()) {
      preface.append("Title: ").append(titles).append('\n');
    }
    if (!authors.isEmpty()) {
      preface.append("Author: ").append(authors).append('\n');
    }
    return preface.toString();
  }

  private static String formatAuthor(final Author author) {
    String first = author.getFirstname() == null ? "" : author.getFirstname().strip();
    String last = author.getLastname() == null ? "" : author.getLastname().strip();
    return (first + " " + last).strip();
  }

  private static Stream<Resource> htmlResources(final Book book) {
    return book.getContents().stream().filter(EpubText::isHtml);
  }

  private static boolean isHtml(final Resource resource) {
    if (resource == null) {
      return false;
    }
    MediaType type = resource.getMediaType();
    if (type != null && type.getName() != null
      && type.getName().toLowerCase(Locale.ROOT).contains("html")) {
      return true;
    }
    String href = resource.getHref();
    if (href == null) {
      return false;
    }
    String lower = href.toLowerCase(Locale.ROOT);
    return lower.endsWith(".html") || lower.endsWith(".xhtml") || lower.endsWith(".htm");
  }

  private static Optional<String> htmlBody(final Resource resource) {
    try {
      byte[] data = resource.getData();
      if (data == null || data.length == 0) {
        return Optional.empty();
      }
      String html = new String(data, charsetOf(resource));
      String text = stripMarkup(html);
      return text.isBlank() ? Optional.empty() : Optional.of(text);
    } catch (IOException e) {
      throw new UncheckedIOException("failed to read EPUB resource " + resource.getHref(), e);
    }
  }

  private static Charset charsetOf(final Resource resource) {
    String encoding = resource.getInputEncoding();
    if (encoding == null || encoding.isBlank()) {
      return UTF_8;
    }
    try {
      return Charset.forName(encoding);
    } catch (IllegalCharsetNameException | UnsupportedCharsetException e) {
      return UTF_8;
    }
  }

  private static String stripMarkup(final String html) {
    String withoutBlocks = SCRIPT_OR_STYLE.matcher(html).replaceAll(" ");
    String withoutTags = TAG.matcher(withoutBlocks).replaceAll(" ");
    return decodeEntities(withoutTags);
  }

  private static String decodeEntities(final String text) {
    String named = NAMED_ENTITY.matcher(text).replaceAll(EpubText::namedEntity);
    return NUMERIC_ENTITY.matcher(named).replaceAll(EpubText::numericEntity);
  }

  private static String namedEntity(final MatchResult match) {
    return NAMED_ENTITIES.getOrDefault(match.group(1), match.group());
  }

  private static String numericEntity(final MatchResult match) {
    String body = match.group(1);
    int code = body.charAt(0) == 'x' || body.charAt(0) == 'X'
      ? Integer.parseInt(body.substring(1), 16)
      : Integer.parseInt(body, 10);
    return code < 0 || code > Character.MAX_CODE_POINT ? match.group() : Character.toString(code);
  }
}
