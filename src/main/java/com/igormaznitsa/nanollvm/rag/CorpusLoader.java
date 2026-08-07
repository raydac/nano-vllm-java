package com.igormaznitsa.nanollvm.rag;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.EngineIo;
import java.io.IOException;
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
 * Loads strings, files, and folder trees into {@link TextChunk}s (preprocess + chunk). Used only
 * from {@link RagFactory}.
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

  static final class Builder {

    private final List<TextChunk> pending = new ArrayList<>();
    private final AtomicInteger anon = new AtomicInteger();
    private int maxChunkChars = 1200;
    private int chunkOverlap = 150;
    private boolean preprocess = true;
    private boolean atomicSentences = false;
    private boolean dedupe = true;
    private Set<String> folderExtensions = DEFAULT_EXTENSIONS;
    private EngineIo io = EngineIo.silent();
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
     * Progress sink for per-file load lines. {@code null} → {@link EngineIo#silent()}.
     */
    public Builder io(final EngineIo io) {
      this.io = io == null ? EngineIo.silent() : io;
      return this;
    }

    public Builder apply(final RagLoadOptions options) {
      requireNonNull(options, "options");
      return this.maxChunkChars(options.maxChunkChars())
          .chunkOverlap(options.chunkOverlap())
          .preprocess(options.preprocess())
          .atomicSentences(options.atomicSentences())
          .dedupe(options.dedupe());
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
        String body = this.readFileText(path);
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
        this.reportFileProcessed(path, body, chunks.size());
      } catch (IOException e) {
        throw new UncheckedIOException("failed to read " + path, e);
      }
      return this;
    }

    private String readFileText(final Path path) throws IOException {
      return PdfTextExtractor.isPdf(path)
        ? PdfTextExtractor.extract(path)
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
      this.io.infof("RAG scanning %s …%n", root);
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

    private void reportFileProcessed(final Path path, final String body, final int chunkCount) {
      if (this.io.isSilent()) {
        return;
      }
      int chars = body == null ? 0 : body.length();
      this.io.infof("RAG %s: %d char(s) → %d chunk(s)%n",
        this.displayPath(path), chars, chunkCount);
      if (chars == 0) {
        this.io.info("RAG warning: no text extracted from " + this.displayPath(path));
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
