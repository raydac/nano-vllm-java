package com.igormaznitsa.nanollvm.rag;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

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

/**
 * Immutable bag of {@link TextChunk}s built from strings, files, and/or a folder tree.
 *
 * <p>With preprocessing, documents become section-aware sentence passages; optional atomic
 * mode keeps one sentence per chunk for small models.
 */
public final class TextCorpus {

  private static final Set<String> DEFAULT_EXTENSIONS = Set.of(
      ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".xml", ".html", ".htm",
      ".properties", ".yml", ".yaml", ".log", ".java", ".kt", ".py", ".js", ".ts", ".css");

  private final List<TextChunk> chunks;

  private TextCorpus(List<TextChunk> chunks) {
    this.chunks = List.copyOf(chunks);
  }

  static TextCorpus ofChunks(List<TextChunk> chunks) {
    return new TextCorpus(chunks);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static TextCorpus ofStrings(String... texts) {
    Builder b = builder();
    for (String text : texts) {
      b.add(text);
    }
    return b.build();
  }

  public static TextCorpus fromFile(Path file) {
    return builder().addFile(file).build();
  }

  public static TextCorpus fromFolder(Path folder) {
    return builder().addFolder(folder).build();
  }

  public List<TextChunk> chunks() {
    return this.chunks;
  }

  public int size() {
    return this.chunks.size();
  }

  public boolean isEmpty() {
    return this.chunks.isEmpty();
  }

  public static final class Builder {

    private final List<TextChunk> pending = new ArrayList<>();
    private final AtomicInteger anon = new AtomicInteger();
    private int maxChunkChars = 1200;
    private int chunkOverlap = 150;
    private boolean preprocess = true;
    private boolean atomicSentences = false;
    private boolean dedupe = true;
    private Set<String> folderExtensions = DEFAULT_EXTENSIONS;

    private static String fingerprint(String text) {
      return text.toLowerCase(Locale.ROOT).replaceAll("\\s+", " ").strip();
    }

    public Builder maxChunkChars(int maxChunkChars) {
      if (maxChunkChars < 64) {
        throw new IllegalArgumentException("maxChunkChars must be >= 64");
      }
      this.maxChunkChars = maxChunkChars;
      return this;
    }

    public Builder chunkOverlap(int chunkOverlap) {
      if (chunkOverlap < 0) {
        throw new IllegalArgumentException("chunkOverlap must be >= 0");
      }
      this.chunkOverlap = chunkOverlap;
      return this;
    }

    public Builder preprocess(boolean preprocess) {
      this.preprocess = preprocess;
      return this;
    }

    public Builder atomicSentences(boolean atomicSentences) {
      this.atomicSentences = atomicSentences;
      return this;
    }

    public Builder dedupe(boolean dedupe) {
      this.dedupe = dedupe;
      return this;
    }

    public Builder apply(RagLoadOptions options) {
      requireNonNull(options, "options");
      return this.maxChunkChars(options.maxChunkChars())
          .chunkOverlap(options.chunkOverlap())
          .preprocess(options.preprocess())
          .atomicSentences(options.atomicSentences())
          .dedupe(options.dedupe());
    }

    public Builder folderExtensions(Set<String> extensions) {
      requireNonNull(extensions, "extensions");
      this.folderExtensions = extensions.stream()
          .map(ext -> ext.startsWith(".") ? ext.toLowerCase(Locale.ROOT)
              : ("." + ext).toLowerCase(Locale.ROOT))
          .collect(java.util.stream.Collectors.toUnmodifiableSet());
      return this;
    }

    public Builder add(String text) {
      return this.add("text-" + this.anon.incrementAndGet(), text);
    }

    public Builder add(String id, String text) {
      return this.add(id, id, text);
    }

    public Builder add(String id, String source, String text) {
      requireNonNull(id, "id");
      requireNonNull(source, "source");
      this.pending.addAll(TextChunker.split(
          id,
          source,
          text,
          this.maxChunkChars,
          this.chunkOverlap,
          this.preprocess,
          this.atomicSentences));
      return this;
    }

    public Builder addAll(Iterable<String> texts) {
      requireNonNull(texts, "texts");
      for (String text : texts) {
        this.add(text);
      }
      return this;
    }

    public Builder addFile(Path file) {
      requireNonNull(file, "file");
      Path path = file.toAbsolutePath().normalize();
      if (!Files.isRegularFile(path)) {
        throw new IllegalArgumentException("not a regular file: " + path);
      }
      try {
        String body = Files.readString(path, UTF_8);
        String source = path.toString();
        this.pending.addAll(TextChunker.split(
            source,
            source,
            body,
            this.maxChunkChars,
            this.chunkOverlap,
            this.preprocess,
            this.atomicSentences));
      } catch (IOException e) {
        throw new UncheckedIOException("failed to read " + path, e);
      }
      return this;
    }

    public Builder addFiles(Path... files) {
      requireNonNull(files, "files");
      for (Path file : files) {
        this.addFile(file);
      }
      return this;
    }

    public Builder addFolder(Path folder) {
      requireNonNull(folder, "folder");
      Path root = folder.toAbsolutePath().normalize();
      if (!Files.isDirectory(root)) {
        throw new IllegalArgumentException("not a directory: " + root);
      }
      try {
        Files.walkFileTree(root, new SimpleFileVisitor<>() {
          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
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

    public TextCorpus build() {
      List<TextChunk> prepared = this.dedupe
          ? this.dedupeChunks(this.pending)
          : this.pending.stream().filter(chunk -> !chunk.isBlank()).toList();
      if (prepared.isEmpty()) {
        throw new IllegalStateException("corpus has no non-blank chunks");
      }
      return new TextCorpus(prepared);
    }

    private List<TextChunk> dedupeChunks(List<TextChunk> chunks) {
      Map<String, TextChunk> unique = new LinkedHashMap<>();
      for (TextChunk chunk : chunks) {
        if (chunk.isBlank()) {
          continue;
        }
        unique.putIfAbsent(fingerprint(chunk.text()), chunk);
      }
      return List.copyOf(unique.values());
    }

    private boolean isIndexedFile(Path file) {
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
}
