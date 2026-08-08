package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Loads and preprocesses documents into a shareable {@link PreparedRag}.
 * Analogous to {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} for weights.
 *
 * <p>Preprocessing is document-side only: section titles, sentence passages, load-time
 * preparsing (model vs search text, term frequencies), inverted BM25 — not user-reply rules.
 * Pass {@link LlmListeners#toSystem()} to print per-file extraction stats while loading.
 *
 * <p>Empty corpora throw {@link ModelLoadException}. The returned index is immutable and safe to
 * share across threads.
 */
public final class RagFactory {

  private RagFactory() {
  }

  public static PreparedRag make(final Path folderOrFile) {
    return make(folderOrFile, RagLoadOptions.defaults(), LlmListeners.silent());
  }

  public static PreparedRag make(final Path folderOrFile, final RagLoadOptions options) {
    return make(folderOrFile, options, LlmListeners.silent());
  }

  public static PreparedRag make(
    final Path folderOrFile,
    final RagLoadOptions options,
    final LlmListener io
  ) {
    return tryMake(folderOrFile, options, io).orElseThrow(() -> new ModelLoadException(
      "corpus has no non-blank chunks: " + folderOrFile.toAbsolutePath().normalize()));
  }

  /**
   * Like {@link #make(Path, RagLoadOptions, LlmListener)} but returns empty when the path exists
   * yet yields no indexable text (empty folder, only README, blank files).
   */
  public static Optional<PreparedRag> tryMake(final Path folderOrFile) {
    return tryMake(folderOrFile, RagLoadOptions.defaults(), LlmListeners.silent());
  }

  public static Optional<PreparedRag> tryMake(final Path folderOrFile,
                                              final RagLoadOptions options) {
    return tryMake(folderOrFile, options, LlmListeners.silent());
  }

  public static Optional<PreparedRag> tryMake(
    final Path folderOrFile,
    final RagLoadOptions options,
    final LlmListener io
  ) {
    requireNonNull(folderOrFile, "folderOrFile");
    requireNonNull(options, "options");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Path path = folderOrFile.toAbsolutePath().normalize();
    CorpusLoader.Builder corpus = CorpusLoader.builder().apply(options).listen(streams);
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
      if (!LlmListeners.isSilent(streams)) {
        LlmListeners.infof(streams, null, "RAG skipped: no indexable text at %s%n", path);
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
    return seal(corpus.build(), null, options, LlmListeners.silent());
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
    final LlmListener io
  ) {
    PreparedRag prepared = PreparedRag.fromChunks(chunks, sourceRoot, options);
    if (!LlmListeners.isSilent(io)) {
      LlmListeners.infof(io, null, "RAG ready: %d chunk(s)%s%n",
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
    private LlmListener io = LlmListeners.silent();

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
     * Progress sink for per-file load lines. {@code null} → {@link LlmListeners#silent()}.
     */
    public Builder listen(final LlmListener io) {
      this.io = io == null ? LlmListeners.silent() : io;
      this.corpus.listen(this.io);
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
