package com.igormaznitsa.nanollvm.samples.utils;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;
import static javax.xml.stream.XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES;
import static javax.xml.stream.XMLInputFactory.SUPPORT_DTD;
import static javax.xml.stream.XMLStreamConstants.END_ELEMENT;
import static javax.xml.stream.XMLStreamConstants.START_ELEMENT;

import com.igormaznitsa.nanollvm.rag.RagResource;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;

/**
 * Plain text from an EPUB (ZIP + OPF + XHTML) using JDK zip and StAX, for
 * {@link com.igormaznitsa.nanollvm.rag.RagTuner} extract / filter hooks.
 */
public final class EpubText {

  public static final String RUR_EPUB = "pg59112.epub";

  private static final String CONTAINER_PATH = "META-INF/container.xml";
  private static final XMLInputFactory XML = secureXmlFactory();

  private static final Pattern SCRIPT_OR_STYLE = Pattern.compile(
    "(?is)<(script|style)\\b[^>]*>.*?</\\1>");
  private static final Pattern TAG = Pattern.compile("(?s)<[^>]+>");
  private static final Pattern NAMED_ENTITY = Pattern.compile("&([A-Za-z][A-Za-z0-9]+);");
  private static final Pattern NUMERIC_ENTITY = Pattern.compile("&#(x[0-9A-Fa-f]+|\\d+);");
  private static final Pattern WHITESPACE = Pattern.compile("\\s+");
  private static final Pattern ZIP_SLASH = Pattern.compile("/");
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
    try {
      return Optional.of(plainText(unzip(resource.content().orElseThrow())));
    } catch (IOException e) {
      throw new UncheckedIOException("failed to extract EPUB " + resource.source(), e);
    }
  }

  public static String normalizeWhitespace(final String text) {
    requireNonNull(text, "text");
    return WHITESPACE.matcher(text.strip()).replaceAll(" ");
  }

  private static XMLInputFactory secureXmlFactory() {
    XMLInputFactory factory = XMLInputFactory.newFactory();
    factory.setProperty(SUPPORT_DTD, false);
    factory.setProperty(IS_SUPPORTING_EXTERNAL_ENTITIES, false);
    return factory;
  }

  private static Map<String, byte[]> unzip(final byte[] epub) throws IOException {
    Map<String, byte[]> files = new LinkedHashMap<>();
    try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(epub), UTF_8)) {
      ZipEntry entry;
      while ((entry = zip.getNextEntry()) != null) {
        if (entry.isDirectory()) {
          continue;
        }
        files.put(normalizeZipPath(entry.getName()), zip.readAllBytes());
      }
    }
    return Map.copyOf(files);
  }

  private static String plainText(final Map<String, byte[]> files) {
    String opfPath = rootfilePath(requireEntry(files, CONTAINER_PATH, "EPUB container"));
    Publication publication = readOpf(requireEntry(files, opfPath, "EPUB package"), opfPath);
    String chapters = publication.htmlHrefs().stream()
      .map(href -> htmlBody(files.get(href)))
      .flatMap(Optional::stream)
      .collect(joining("\n"));
    String preface = metadataPreface(publication);
    return preface.isEmpty() ? chapters : preface + "\n" + chapters;
  }

  private static byte[] requireEntry(
    final Map<String, byte[]> files,
    final String path,
    final String label
  ) {
    byte[] bytes = files.get(path);
    if (bytes == null || bytes.length == 0) {
      throw new IllegalArgumentException(label + " missing: " + path);
    }
    return bytes;
  }

  private static String rootfilePath(final byte[] containerXml) {
    try {
      XMLStreamReader reader = openXml(containerXml);
      try {
        while (reader.hasNext()) {
          if (reader.next() == START_ELEMENT && "rootfile".equals(reader.getLocalName())) {
            String path = attr(reader, "full-path");
            if (path != null && !path.isBlank()) {
              return normalizeZipPath(path);
            }
          }
        }
      } finally {
        reader.close();
      }
    } catch (XMLStreamException e) {
      throw new UncheckedIOException(new IOException("invalid EPUB container.xml", e));
    }
    throw new IllegalArgumentException("EPUB container.xml has no rootfile");
  }

  private static Publication readOpf(final byte[] opf, final String opfPath) {
    List<String> titles = new ArrayList<>();
    List<String> authors = new ArrayList<>();
    Map<String, ManifestItem> manifest = new LinkedHashMap<>();
    List<String> spine = new ArrayList<>();
    String section = "";

    try {
      XMLStreamReader reader = openXml(opf);
      try {
        while (reader.hasNext()) {
          int event = reader.next();
          if (event == START_ELEMENT) {
            String local = reader.getLocalName();
            if ("metadata".equals(local) || "manifest".equals(local) || "spine".equals(local)) {
              section = local;
            } else if ("title".equals(local) && "metadata".equals(section)) {
              addNonBlank(titles, reader.getElementText());
            } else if ("creator".equals(local) && "metadata".equals(section)) {
              addNonBlank(authors, reader.getElementText());
            } else if ("item".equals(local) && "manifest".equals(section)) {
              putManifestItem(manifest, reader);
            } else if ("itemref".equals(local) && "spine".equals(section)) {
              addNonBlank(spine, attr(reader, "idref"));
            }
          } else if (event == END_ELEMENT) {
            String local = reader.getLocalName();
            if (local.equals(section)) {
              section = "";
            }
          }
        }
      } finally {
        reader.close();
      }
    } catch (XMLStreamException e) {
      throw new UncheckedIOException(new IOException("invalid EPUB package " + opfPath, e));
    }

    List<String> htmlHrefs = spine.stream()
      .map(manifest::get)
      .filter(Objects::nonNull)
      .filter(ManifestItem::isHtml)
      .map(item -> resolveHref(opfPath, item.href()))
      .toList();
    return new Publication(List.copyOf(titles), List.copyOf(authors), htmlHrefs);
  }

  private static void putManifestItem(
    final Map<String, ManifestItem> manifest,
    final XMLStreamReader reader
  ) {
    String id = attr(reader, "id");
    String href = attr(reader, "href");
    if (id == null || id.isBlank() || href == null || href.isBlank()) {
      return;
    }
    String mediaType = Optional.ofNullable(attr(reader, "media-type")).orElse("");
    manifest.put(id, new ManifestItem(href, mediaType));
  }

  private static void addNonBlank(final List<String> values, final String value) {
    if (value != null && !value.isBlank()) {
      values.add(value.strip());
    }
  }

  private static String metadataPreface(final Publication publication) {
    String titles = String.join("; ", publication.titles());
    String authors = String.join(", ", publication.authors());
    StringBuilder preface = new StringBuilder();
    if (!titles.isEmpty()) {
      preface.append("Title: ").append(titles).append('\n');
    }
    if (!authors.isEmpty()) {
      preface.append("Author: ").append(authors).append('\n');
    }
    return preface.toString();
  }

  private static Optional<String> htmlBody(final byte[] data) {
    if (data == null || data.length == 0) {
      return Optional.empty();
    }
    String text = stripMarkup(new String(data, UTF_8));
    return text.isBlank() ? Optional.empty() : Optional.of(text);
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

  private static XMLStreamReader openXml(final byte[] xml) throws XMLStreamException {
    return XML.createXMLStreamReader(new ByteArrayInputStream(xml));
  }

  private static String attr(final XMLStreamReader reader, final String localName) {
    for (int i = 0; i < reader.getAttributeCount(); i++) {
      if (localName.equals(reader.getAttributeLocalName(i))) {
        return reader.getAttributeValue(i);
      }
    }
    return null;
  }

  private static String resolveHref(final String opfPath, final String href) {
    String relative = stripFragment(href.strip()).replace('\\', '/');
    if (relative.startsWith("/")) {
      return normalizeZipPath(relative.substring(1));
    }
    int slash = opfPath.lastIndexOf('/');
    String dir = slash < 0 ? "" : opfPath.substring(0, slash + 1);
    return normalizeZipPath(dir + relative);
  }

  private static String stripFragment(final String href) {
    int hash = href.indexOf('#');
    return hash < 0 ? href : href.substring(0, hash);
  }

  private static String normalizeZipPath(final String path) {
    List<String> parts = new ArrayList<>();
    for (String part : ZIP_SLASH.split(path.replace('\\', '/'), -1)) {
      if (part.isEmpty() || ".".equals(part)) {
        continue;
      }
      if ("..".equals(part)) {
        if (!parts.isEmpty()) {
          parts.removeLast();
        }
        continue;
      }
      parts.add(part);
    }
    return String.join("/", parts);
  }

  private record ManifestItem(String href, String mediaType) {

    boolean isHtml() {
      if (this.mediaType.toLowerCase(Locale.ROOT).contains("html")) {
        return true;
      }
      String lower = this.href.toLowerCase(Locale.ROOT);
      return lower.endsWith(".html") || lower.endsWith(".xhtml") || lower.endsWith(".htm");
    }
  }

  private record Publication(List<String> titles, List<String> authors, List<String> htmlHrefs) {
  }
}
