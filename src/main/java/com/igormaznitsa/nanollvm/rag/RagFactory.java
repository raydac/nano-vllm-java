package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.EngineIo;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Loads and preprocesses documents into a shareable {@link PreparedRag}.
 * Analogous to {@link com.igormaznitsa.nanollvm.models.ModelFactory} for weights.
 *
 * <p>Preprocessing is document-side only: section titles, sentence passages, load-time
 * preparsing (model vs search text, term frequencies), inverted BM25 — not user-reply rules.
 * Pass {@link EngineIo#system()} to print per-file extraction stats while loading.
 */
public final class RagFactory {

  private RagFactory() {
  }

  public static PreparedRag make(final Path folderOrFile) {
    return make(folderOrFile, RagLoadOptions.defaults(), EngineIo.silent());
  }

  public static PreparedRag make(final Path folderOrFile, final RagLoadOptions options) {
    return make(folderOrFile, options, EngineIo.silent());
  }

  public static PreparedRag make(
    final Path folderOrFile,
    final RagLoadOptions options,
    final EngineIo io
  ) {
    return tryMake(folderOrFile, options, io).orElseThrow(() -> new IllegalStateException(
      "corpus has no non-blank chunks: " + folderOrFile.toAbsolutePath().normalize()));
  }

  /**
   * Like {@link #make(Path, RagLoadOptions, EngineIo)} but returns empty when the path exists
   * yet yields no indexable text (empty folder, only README, blank files).
   */
  public static Optional<PreparedRag> tryMake(final Path folderOrFile) {
    return tryMake(folderOrFile, RagLoadOptions.defaults(), EngineIo.silent());
  }

  public static Optional<PreparedRag> tryMake(final Path folderOrFile,
                                              final RagLoadOptions options) {
    return tryMake(folderOrFile, options, EngineIo.silent());
  }

  public static Optional<PreparedRag> tryMake(
    final Path folderOrFile,
    final RagLoadOptions options,
    final EngineIo io
  ) {
    requireNonNull(folderOrFile, "folderOrFile");
    requireNonNull(options, "options");
    EngineIo streams = io == null ? EngineIo.silent() : io;
    Path path = folderOrFile.toAbsolutePath().normalize();
    CorpusLoader.Builder corpus = CorpusLoader.builder().apply(options).io(streams);
    if (Files.isDirectory(path)) {
      corpus.addFolder(path);
    } else if (Files.isRegularFile(path)) {
      corpus.addFile(path);
    } else {
      throw new IllegalArgumentException("path is not a file or directory: " + path);
    }
    try {
      return Optional.of(seal(corpus.build(), path, options, streams));
    } catch (IllegalStateException emptyCorpus) {
      String message = emptyCorpus.getMessage();
      if (message == null || !message.startsWith("corpus has no")) {
        throw emptyCorpus;
      }
      if (!streams.isSilent()) {
        streams.infof("RAG skipped: no indexable text at %s%n", path);
      }
      return Optional.empty();
    }
  }

  public static PreparedRag of(String... texts) {
    return of(RagLoadOptions.defaults(), texts);
  }

  public static PreparedRag of(final RagLoadOptions options, String... texts) {
    requireNonNull(options, "options");
    requireNonNull(texts, "texts");
    CorpusLoader.Builder corpus = CorpusLoader.builder().apply(options);
    for (String text : texts) {
      corpus.add(text);
    }
    return seal(corpus.build(), null, options, EngineIo.silent());
  }

  public static PreparedRag of(final List<String> texts) {
    return of(RagLoadOptions.defaults(), texts.toArray(String[]::new));
  }

  public static Builder builder() {
    return new Builder();
  }

  private static PreparedRag seal(
    final List<TextChunk> chunks,
    final Path sourceRoot,
    final RagLoadOptions options,
    final EngineIo io
  ) {
    PreparedRag prepared = PreparedRag.fromChunks(chunks, sourceRoot, options);
    if (!io.isSilent()) {
      io.infof("RAG ready: %d chunk(s)%s%n",
        prepared.size(),
        sourceRoot == null ? "" : " from " + sourceRoot);
    }
    return prepared;
  }

  public static final class Builder {

    private RagLoadOptions options = RagLoadOptions.defaults();
    private final CorpusLoader.Builder corpus = CorpusLoader.builder().apply(this.options);
    private Path sourceRoot;
    private boolean hasContent;
    private EngineIo io = EngineIo.silent();

    public Builder options(final RagLoadOptions options) {
      if (this.hasContent) {
        throw new IllegalStateException("options must be set before adding documents");
      }
      this.options = requireNonNull(options, "options");
      this.corpus.apply(options);
      return this;
    }

    public Builder forTinyModels() {
      return this.options(RagLoadOptions.forTinyModels());
    }

    /**
     * Progress sink for per-file load lines. {@code null} → {@link EngineIo#silent()}.
     */
    public Builder io(final EngineIo io) {
      this.io = io == null ? EngineIo.silent() : io;
      this.corpus.io(this.io);
      return this;
    }

    public Builder sourceRoot(final Path sourceRoot) {
      this.sourceRoot = requireNonNull(sourceRoot, "sourceRoot").toAbsolutePath().normalize();
      return this;
    }

    public Builder folderExtensions(final Set<String> extensions) {
      this.corpus.folderExtensions(extensions);
      return this;
    }

    public Builder add(final String text) {
      this.hasContent = true;
      this.corpus.add(text);
      return this;
    }

    public Builder add(final String id, final String text) {
      this.hasContent = true;
      this.corpus.add(id, text);
      return this;
    }

    public Builder addFile(final Path file) {
      this.hasContent = true;
      this.corpus.addFile(file);
      return this;
    }

    public Builder addFolder(final Path folder) {
      this.hasContent = true;
      this.corpus.addFolder(folder);
      if (this.sourceRoot == null) {
        this.sourceRoot = folder.toAbsolutePath().normalize();
      }
      return this;
    }

    public Builder addFiles(Path... files) {
      this.hasContent = true;
      this.corpus.addFiles(files);
      return this;
    }

    public PreparedRag build() {
      return seal(this.corpus.build(), this.sourceRoot, this.options, this.io);
    }

    @Override
    public String toString() {
      return "RagFactory.Builder{options=%s, source=%s}".formatted(
          this.options,
          this.sourceRoot == null ? "-" : this.sourceRoot);
    }
  }
}
