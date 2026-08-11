package com.igormaznitsa.nanollvm.rag;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Loads strings, files, classpath resources, and folder trees into {@link TextChunk}s
 * (preprocess + chunk). Used only from {@link RagFactory}.
 */
final class CorpusLoader {

  private static final Set<String> DEFAULT_EXTENSIONS = Set.of(
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".xml", ".html", ".htm",
    ".properties", ".yml", ".yaml", ".log", ".java", ".kt", ".py", ".js", ".ts", ".css",
    ".pdf");

  private CorpusLoader() {
  }

  static Builder builder() {
    return new Builder();
  }

  private static ClassLoader defaultClassLoader() {
    ClassLoader loader = CorpusLoader.class.getClassLoader();
    return loader == null ? ClassLoader.getSystemClassLoader() : loader;
  }

  private static String normalizeClasspathPath(final String resourcePath, final String paramName) {
    requireNonNull(resourcePath, paramName);
    String trimmed = resourcePath.strip();
    if (trimmed.isEmpty()) {
      throw new IllegalArgumentException(paramName + " must not be blank");
    }
    while (trimmed.startsWith("/")) {
      trimmed = trimmed.substring(1);
    }
    while (trimmed.endsWith("/")) {
      trimmed = trimmed.substring(0, trimmed.length() - 1);
    }
    if (trimmed.isEmpty()) {
      throw new IllegalArgumentException(paramName + " must not be blank");
    }
    return trimmed;
  }

  private static String packageRelativeClasspath(final Class<?> anchor, final String relative) {
    String pkg = anchor.getPackageName();
    if (pkg.isBlank()) {
      return relative;
    }
    return pkg.replace('.', '/') + "/" + relative;
  }

  private static boolean isPdfName(final String path) {
    return path.toLowerCase(Locale.ROOT).endsWith(".pdf");
  }

  static final class Builder {

    private final List<TextChunk> pending = new ArrayList<>();
    private final AtomicInteger anon = new AtomicInteger();
    private int maxChunkChars = 1200;
    private int chunkOverlap = 150;
    private boolean preprocess = true;
    private boolean atomicSentences = false;
    private boolean dedupe = true;
    private ResourceLimits resourceLimits = ResourceLimits.current();
    private long totalBytesRead;
    private int filesRead;
    private Set<String> folderExtensions = DEFAULT_EXTENSIONS;
    private LlmListener io = LlmListeners.silent();
    private Path reportRoot;

    private static String fingerprint(final String text) {
      return text.toLowerCase(Locale.ROOT).replaceAll("\\s+", " ").strip();
    }

    public Builder maxChunkChars(final int maxChunkChars) {
      if (maxChunkChars < 64) {
        throw new IllegalArgumentException("maxChunkChars must be >= 64");
      }
      this.maxChunkChars = maxChunkChars;
      return this;
    }

    public Builder chunkOverlap(final int chunkOverlap) {
      if (chunkOverlap < 0) {
        throw new IllegalArgumentException("chunkOverlap must be >= 0");
      }
      this.chunkOverlap = chunkOverlap;
      return this;
    }

    public Builder preprocess(final boolean preprocess) {
      this.preprocess = preprocess;
      return this;
    }

    public Builder atomicSentences(final boolean atomicSentences) {
      this.atomicSentences = atomicSentences;
      return this;
    }

    public Builder dedupe(final boolean dedupe) {
      this.dedupe = dedupe;
      return this;
    }

    /**
     * Progress sink for per-file load lines. {@code null} → {@link LlmListeners#silent()}.
     */
    public Builder listen(final LlmListener io) {
      this.io = io == null ? LlmListeners.silent() : io;
      return this;
    }

    public Builder apply(final RagLoadOptions options) {
      requireNonNull(options, "options");
      return this.maxChunkChars(options.maxChunkChars())
        .chunkOverlap(options.chunkOverlap())
        .preprocess(options.preprocess())
        .atomicSentences(options.atomicSentences())
        .dedupe(options.dedupe())
        .resourceLimits(options.resourceLimits());
    }

    public Builder resourceLimits(final ResourceLimits resourceLimits) {
      this.resourceLimits = requireNonNull(resourceLimits, "resourceLimits");
      return this;
    }

    public Builder folderExtensions(final Set<String> extensions) {
      requireNonNull(extensions, "extensions");
      this.folderExtensions = extensions.stream()
        .map(ext -> ext.startsWith(".") ? ext.toLowerCase(Locale.ROOT)
          : ("." + ext).toLowerCase(Locale.ROOT))
        .collect(java.util.stream.Collectors.toUnmodifiableSet());
      return this;
    }

    public Builder add(final String text) {
      return this.add("text-" + this.anon.incrementAndGet(), text);
    }

    public Builder add(final String id, final String text) {
      return this.add(id, id, text);
    }

    public Builder add(final String id, final String source, final String text) {
      requireNonNull(id, "id");
      requireNonNull(source, "source");
      this.pending.addAll(Chunking.split(
        id,
        source,
        text,
        this.maxChunkChars,
        this.chunkOverlap,
        this.preprocess,
        this.atomicSentences));
      return this;
    }

    public Builder addAll(final Iterable<String> texts) {
      requireNonNull(texts, "texts");
      for (String text : texts) {
        this.add(text);
      }
      return this;
    }

    public Builder addFile(final Path file) {
      requireNonNull(file, "file");
      Path path = file.toAbsolutePath().normalize();
      if (!Files.isRegularFile(path)) {
        throw new IllegalArgumentException("not a regular file: " + path);
      }
      try {
        long size = Files.size(path);
        this.requireBudget(path.toString(), size);
        String body = this.readFileText(path);
        this.accountRead(size);
        String source = path.toString();
        List<TextChunk> chunks = Chunking.split(
          source,
          source,
          body,
          this.maxChunkChars,
          this.chunkOverlap,
          this.preprocess,
          this.atomicSentences);
        this.pending.addAll(chunks);
        this.reportLoaded(this.displayPath(path), body, chunks.size());
      } catch (IOException e) {
        throw new UncheckedIOException("failed to read " + path, e);
      }
      return this;
    }

    /**
     * Classpath resource via this module's class loader (absolute path, no leading {@code /}).
     */
    public Builder addResource(final String resourcePath) {
      return this.addResource(CorpusLoader.defaultClassLoader(), resourcePath);
    }

    /**
     * Absolute classpath resource (no leading {@code /}), e.g. {@code rag/facts.md}.
     */
    public Builder addResource(final ClassLoader loader, final String resourcePath) {
      requireNonNull(loader, "loader");
      String path = normalizeClasspathPath(resourcePath, "resourcePath");
      try (InputStream in = loader.getResourceAsStream(path)) {
        if (in == null) {
          throw new IllegalArgumentException("classpath resource not found: " + path);
        }
        return this.addResourceBytes(path, in.readAllBytes());
      } catch (IOException e) {
        throw new UncheckedIOException("failed to read classpath:" + path, e);
      }
    }

    /**
     * Classpath resource resolved like {@link Class#getResourceAsStream(String)}:
     * leading {@code /} = absolute from classpath root; otherwise package-relative to
     * {@code anchor}.
     */
    public Builder addResource(final Class<?> anchor, final String resourcePath) {
      requireNonNull(anchor, "anchor");
      requireNonNull(resourcePath, "resourcePath");
      String raw = resourcePath.strip();
      if (raw.isEmpty()) {
        throw new IllegalArgumentException("resourcePath must not be blank");
      }
      try (InputStream in = anchor.getResourceAsStream(raw)) {
        if (in == null) {
          throw new IllegalArgumentException(
            "classpath resource not found for " + anchor.getName() + ": " + raw);
        }
        String label = raw.startsWith("/")
          ? normalizeClasspathPath(raw, "resourcePath")
          : packageRelativeClasspath(anchor, raw);
        return this.addResourceBytes(label, in.readAllBytes());
      } catch (IOException e) {
        throw new UncheckedIOException("failed to read classpath resource: " + raw, e);
      }
    }

    public Builder addResources(final String... resourcePaths) {
      requireNonNull(resourcePaths, "resourcePaths");
      for (String resourcePath : resourcePaths) {
        this.addResource(resourcePath);
      }
      return this;
    }

    public Builder addResources(final ClassLoader loader, final String... resourcePaths) {
      requireNonNull(loader, "loader");
      requireNonNull(resourcePaths, "resourcePaths");
      for (String resourcePath : resourcePaths) {
        this.addResource(loader, resourcePath);
      }
      return this;
    }

    private Builder addResourceBytes(final String classpathPath, final byte[] bytes) {
      requireNonNull(classpathPath, "classpathPath");
      requireNonNull(bytes, "bytes");
      String source = "classpath:" + classpathPath;
      this.requireBudget(source, bytes.length);
      String body = isPdfName(classpathPath)
        ? PdfTextExtractor.extract(bytes, this.resourceLimits)
        : new String(bytes, UTF_8);
      this.accountRead(bytes.length);
      List<TextChunk> chunks = Chunking.split(
        source,
        source,
        body,
        this.maxChunkChars,
        this.chunkOverlap,
        this.preprocess,
        this.atomicSentences);
      this.pending.addAll(chunks);
      this.reportLoaded(source, body, chunks.size());
      return this;
    }

    private void requireBudget(final String label, final long size) {
      if (size > this.resourceLimits.maxFileBytes()) {
        throw new IllegalArgumentException(
          "file exceeds maxFileBytes (" + this.resourceLimits.maxFileBytes() + "): " + label);
      }
      if (this.filesRead >= this.resourceLimits.maxCorpusFiles()) {
        throw new IllegalStateException(
          "corpus exceeds maxCorpusFiles (" + this.resourceLimits.maxCorpusFiles() + ")");
      }
      if (this.totalBytesRead + size > this.resourceLimits.maxTotalCorpusBytes()) {
        throw new IllegalStateException(
          "corpus exceeds maxTotalCorpusBytes ("
            + this.resourceLimits.maxTotalCorpusBytes() + ")");
      }
    }

    private void accountRead(final long size) {
      this.filesRead++;
      this.totalBytesRead += size;
    }

    private String readFileText(final Path path) throws IOException {
      return PdfTextExtractor.isPdf(path)
        ? PdfTextExtractor.extract(path, this.resourceLimits)
        : Files.readString(path, UTF_8);
    }

    public Builder addFiles(Path... files) {
      requireNonNull(files, "files");
      for (Path file : files) {
        this.addFile(file);
      }
      return this;
    }

    public Builder addFolder(final Path folder) {
      requireNonNull(folder, "folder");
      Path root = folder.toAbsolutePath().normalize();
      if (!Files.isDirectory(root)) {
        throw new IllegalArgumentException("not a directory: " + root);
      }
      this.reportRoot = root;
      LlmListeners.infof(io, null, "RAG scanning %s …%n", root);
      try {
        Files.walkFileTree(root, new SimpleFileVisitor<>() {
          @Override
          public FileVisitResult visitFile(final Path file, final BasicFileAttributes attrs) {
            if (attrs.isRegularFile() && Builder.this.isIndexedFile(file)) {
              Builder.this.addFile(file);
            }
            return FileVisitResult.CONTINUE;
          }
        });
      } catch (IOException e) {
        throw new UncheckedIOException("failed to walk " + root, e);
      }
      return this;
    }

    List<TextChunk> build() {
      List<TextChunk> prepared = this.dedupe
        ? this.dedupeChunks(this.pending)
        : this.pending.stream().filter(chunk -> !chunk.isBlank()).toList();
      if (prepared.isEmpty()) {
        throw new IllegalStateException("corpus has no non-blank chunks");
      }
      return List.copyOf(prepared);
    }

    private void reportLoaded(final String display, final String body, final int chunkCount) {
      if (LlmListeners.isSilent(io)) {
        return;
      }
      int chars = body == null ? 0 : body.length();
      LlmListeners.infof(io, null, "RAG %s: %d char(s) → %d chunk(s)%n",
        display, chars, chunkCount);
      if (chars == 0) {
        LlmListeners.info(io, null, "RAG warning: no text extracted from " + display);
      }
    }

    private String displayPath(final Path path) {
      if (this.reportRoot != null) {
        try {
          Path relative = this.reportRoot.relativize(path);
          if (relative.getNameCount() > 0) {
            return relative.toString();
          }
        } catch (IllegalArgumentException ignored) {
          // different roots — fall through
        }
      }
      Path name = path.getFileName();
      return name == null ? path.toString() : name.toString();
    }

    private List<TextChunk> dedupeChunks(final List<TextChunk> chunks) {
      Map<String, TextChunk> unique = new LinkedHashMap<>();
      for (TextChunk chunk : chunks) {
        if (chunk.isBlank()) {
          continue;
        }
        unique.putIfAbsent(fingerprint(chunk.text()), chunk);
      }
      return List.copyOf(unique.values());
    }

    private boolean isIndexedFile(final Path file) {
      String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
      if (name.equals("readme.md") || name.equals("readme.txt") || name.equals("readme.markdown")) {
        return false;
      }
      int dot = name.lastIndexOf('.');
      if (dot < 0) {
        return false;
      }
      return this.folderExtensions.contains(name.substring(dot));
    }
  }

  private static final class DocumentCleanup {

    private static final Pattern HEADING_LINE = Pattern.compile("^#{1,6}\\s+(.+?)\\s*$");
    private static final Pattern LINK = Pattern.compile("\\[([^\\]]+)]\\([^)]*\\)");
    private static final Pattern IMAGE = Pattern.compile("!\\[[^\\]]*]\\([^)]*\\)");
    private static final Pattern CODE_FENCE = Pattern.compile("(?s)```.*?```");
    private static final Pattern INLINE_CODE = Pattern.compile("`([^`]+)`");
    private static final Pattern BULLET = Pattern.compile("^\\s*[-*+]\\s+");
    private static final Pattern SENTENCE_END =
      Pattern.compile("(?<=[.!?。！？])\\s+(?=[\\p{L}\\p{N}\"'(«])");

    private DocumentCleanup() {
    }

    static List<String> passages(final String raw) {
      if (raw == null || raw.isBlank()) {
        return List.of();
      }
      String text =
        CODE_FENCE.matcher(raw.replace("\r\n", "\n").replace('\r', '\n')).replaceAll("\n");
      String section = "";
      StringBuilder paragraph = new StringBuilder();
      List<String> out = new ArrayList<>();

      for (String line : text.split("\n", -1)) {
        Matcher heading = HEADING_LINE.matcher(line.strip());
        if (heading.matches()) {
          flushParagraph(paragraph, section, out);
          section = cleanInline(heading.group(1)).strip();
          continue;
        }
        if (line.isBlank()) {
          flushParagraph(paragraph, section, out);
          continue;
        }
        String cleaned = cleanInline(BULLET.matcher(line).replaceFirst("")).strip();
        if (cleaned.isEmpty()) {
          continue;
        }
        if (!paragraph.isEmpty()) {
          paragraph.append(' ');
        }
        paragraph.append(cleaned);
      }
      flushParagraph(paragraph, section, out);
      return List.copyOf(out);
    }

    private static void flushParagraph(final StringBuilder paragraph, final String section,
                                       final List<String> out) {
      if (paragraph.isEmpty()) {
        return;
      }
      String body = paragraph.toString().replaceAll(" +", " ").strip();
      paragraph.setLength(0);
      if (body.isEmpty()) {
        return;
      }
      for (String sentence : SENTENCE_END.split(body)) {
        String s = sentence.strip();
        if (s.isEmpty()) {
          continue;
        }
        out.add(section.isBlank() ? s : section + " — " + s);
      }
    }

    private static String cleanInline(final String line) {
      String text = IMAGE.matcher(line).replaceAll("");
      text = LINK.matcher(text).replaceAll("$1");
      text = INLINE_CODE.matcher(text).replaceAll("$1");
      text = text.replace('\t', ' ').replaceAll("[ ]{2,}", " ");
      return text;
    }
  }

  private static final class Chunking {

    private Chunking() {
    }

    static List<TextChunk> split(
      final String baseId,
      final String source,
      final String text,
      final int maxChunkChars,
      final int overlap,
      final boolean preprocess,
      final boolean atomicSentences
    ) {
      requireNonNull(baseId, "baseId");
      requireNonNull(source, "source");
      if (text == null || text.isBlank()) {
        return List.of();
      }
      if (preprocess) {
        List<String> units = DocumentCleanup.passages(text);
        if (!units.isEmpty()) {
          return atomicSentences
            ? atomicSplit(baseId, source, units, maxChunkChars)
            : packUnits(baseId, source, units, maxChunkChars);
        }
      }
      return windowSplit(baseId, source, text.strip(), maxChunkChars, overlap);
    }

    private static List<TextChunk> atomicSplit(
      final String baseId,
      final String source,
      final List<String> units,
      final int maxChunkChars
    ) {
      List<TextChunk> chunks = new ArrayList<>();
      int part = 0;
      for (String unit : units) {
        if (unit.length() <= maxChunkChars) {
          part++;
          chunks.add(new TextChunk(baseId + "#" + part, source, unit));
          continue;
        }
        List<TextChunk> windows =
          windowSplit(baseId + "#" + (part + 1), source, unit, maxChunkChars, 0);
        for (TextChunk window : windows) {
          part++;
          chunks.add(new TextChunk(baseId + "#" + part, source, window.text()));
        }
      }
      if (chunks.size() == 1) {
        return List.of(new TextChunk(baseId, source, chunks.getFirst().text()));
      }
      return List.copyOf(chunks);
    }

    private static List<TextChunk> packUnits(
      final String baseId,
      final String source,
      final List<String> units,
      final int maxChunkChars
    ) {
      List<TextChunk> chunks = new ArrayList<>();
      StringBuilder buf = new StringBuilder();
      for (String unit : units) {
        if (unit.length() > maxChunkChars) {
          emitPacked(buf, baseId, source, chunks);
          chunks.addAll(
            windowSplit(baseId + "#w" + (chunks.size() + 1), source, unit, maxChunkChars, 40));
          continue;
        }
        if (buf.isEmpty()) {
          buf.append(unit);
          continue;
        }
        if (buf.length() + 1 + unit.length() <= maxChunkChars) {
          buf.append(' ').append(unit);
          continue;
        }
        emitPacked(buf, baseId, source, chunks);
        buf.append(unit);
      }
      emitPacked(buf, baseId, source, chunks);
      if (chunks.size() == 1) {
        TextChunk only = chunks.getFirst();
        return List.of(new TextChunk(baseId, only.source(), only.text()));
      }
      return List.copyOf(chunks);
    }

    private static void emitPacked(
      final StringBuilder buf,
      final String baseId,
      final String source,
      final List<TextChunk> chunks
    ) {
      if (buf.isEmpty()) {
        return;
      }
      int part = chunks.size() + 1;
      chunks.add(new TextChunk(baseId + "#" + part, source, buf.toString()));
      buf.setLength(0);
    }

    private static List<TextChunk> windowSplit(
      final String baseId,
      final String source,
      final String body,
      final int maxChunkChars,
      final int overlap
    ) {
      if (body.length() <= maxChunkChars) {
        return List.of(new TextChunk(baseId, source, body));
      }
      int step = Math.max(1, maxChunkChars - Math.min(overlap, maxChunkChars - 1));
      List<TextChunk> chunks = new ArrayList<>();
      int start = 0;
      int part = 0;
      while (start < body.length()) {
        int end = Math.min(body.length(), start + maxChunkChars);
        if (end < body.length()) {
          end = preferBreak(body, start, end);
        }
        String slice = body.substring(start, end).strip();
        if (!slice.isEmpty()) {
          part++;
          String id = part == 1 && end >= body.length() ? baseId : baseId + "#" + part;
          chunks.add(new TextChunk(id, source, slice));
        }
        if (end >= body.length()) {
          break;
        }
        start = Math.max(start + 1, end - (maxChunkChars - step));
        start = Math.min(start, end);
      }
      return List.copyOf(chunks);
    }

    private static int preferBreak(final String body, final int start, final int end) {
      int windowStart = Math.max(start + (end - start) / 2, start);
      int nl = body.lastIndexOf('\n', end - 1);
      if (nl >= windowStart) {
        return nl + 1;
      }
      int space = body.lastIndexOf(' ', end - 1);
      if (space >= windowStart) {
        return space + 1;
      }
      return end;
    }
  }
}
