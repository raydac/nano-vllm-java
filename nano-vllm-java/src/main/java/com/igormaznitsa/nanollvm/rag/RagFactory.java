package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.models.LlmModel;
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
 * Optional embedding models produce a {@link HybridRagIndex} via {@link #withEmbeddings}.
 * Classpath documents use {@link Builder#addResource(String)} / {@link #makeResource(String)}.
 *
 * <p>Empty corpora throw {@link ModelLoadException}. Lexical indexes are immutable and safe to
 * share across threads; hybrid indexes additionally need a live embedding {@link LlmModel}.
 */
public final class RagFactory {

  private RagFactory() {
  }

  /**
   * Indexes a file or folder with {@link RagLoadOptions#defaults()}.
   *
   * @throws ModelLoadException if the path yields no indexable chunks
   */
  public static PreparedRag make(final Path folderOrFile) {
    return make(folderOrFile, RagLoadOptions.defaults(), LlmListeners.silent());
  }

  /**
   * Indexes a file or folder with explicit chunk/preprocess options.
   *
   * @throws ModelLoadException if the path yields no indexable chunks
   */
  public static PreparedRag make(final Path folderOrFile, final RagLoadOptions options) {
    return make(folderOrFile, options, LlmListeners.silent());
  }

  /**
   * Indexes a file or folder, emitting per-file load lines to {@code io}.
   *
   * @throws ModelLoadException if the path yields no indexable chunks
   */
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

  /**
   * Like {@link #tryMake(Path, RagLoadOptions, LlmListener)} with default options and a silent
   * listener.
   */
  public static Optional<PreparedRag> tryMake(final Path folderOrFile,
                                              final RagLoadOptions options) {
    return tryMake(folderOrFile, options, LlmListeners.silent());
  }

  /**
   * Like {@link #make(Path, RagLoadOptions, LlmListener)} but returns empty when the path exists
   * yet yields no indexable text (empty folder, only README, blank files).
   *
   * @throws IllegalArgumentException if {@code folderOrFile} is not a file or directory
   */
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

  /**
   * Inline documents with {@link RagLoadOptions#defaults()}.
   *
   * @throws ModelLoadException if every text is blank
   */
  public static PreparedRag of(String... texts) {
    return of(RagLoadOptions.defaults(), texts);
  }

  /**
   * Inline documents with explicit chunk/preprocess options.
   *
   * @throws ModelLoadException if every text is blank
   */
  public static PreparedRag of(final RagLoadOptions options, String... texts) {
    requireNonNull(options, "options");
    requireNonNull(texts, "texts");
    CorpusLoader.Builder corpus = CorpusLoader.builder().apply(options);
    for (String text : texts) {
      corpus.add(text);
    }
    return seal(corpus.build(), null, options, LlmListeners.silent());
  }

  /**
   * {@link #of(String...)} from a list.
   */
  public static PreparedRag of(final List<String> texts) {
    return of(RagLoadOptions.defaults(), texts.toArray(String[]::new));
  }

  /**
   * Fluent corpus builder (inline text, files, folders, classpath resources).
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Loads one absolute classpath resource into a BM25 index.
   *
   * @since 1.1.0
   */
  public static PreparedRag makeResource(final String resourcePath) {
    return builder().addResource(resourcePath).build();
  }

  /**
   * Loads one classpath resource resolved with {@code loader}.
   *
   * @since 1.1.0
   */
  public static PreparedRag makeResource(final ClassLoader loader, final String resourcePath) {
    return builder().addResource(loader, resourcePath).build();
  }

  /**
   * Loads a resource via {@link Class#getResourceAsStream(String)} (leading {@code /} = absolute).
   *
   * @since 1.1.0
   */
  public static PreparedRag makeResource(final Class<?> anchor, final String resourcePath) {
    return builder().addResource(anchor, resourcePath).build();
  }

  /**
   * Hybrid BM25 + dense retrieval over an existing lexical index.
   *
   * @since 1.1.0
   */
  public static HybridRagIndex withEmbeddings(
    final PreparedRag lexical,
    final LlmModel embeddingModel
  ) {
    requireNonNull(lexical, "lexical");
    requireNonNull(embeddingModel, "embeddingModel");
    return HybridRagIndex.of(lexical, embeddingModel);
  }

  /**
   * Lexical BM25 corpus plus dense embeddings from {@code embeddingModel}.
   *
   * @since 1.1.0
   */
  public static HybridRagIndex make(final Path folderOrFile, final LlmModel embeddingModel) {
    return withEmbeddings(make(folderOrFile), embeddingModel);
  }

  /**
   * Lexical BM25 plus dense embeddings, with corpus load options.
   *
   * @since 1.1.0
   */
  public static HybridRagIndex make(
    final Path folderOrFile,
    final RagLoadOptions options,
    final LlmModel embeddingModel
  ) {
    return withEmbeddings(make(folderOrFile, options), embeddingModel);
  }

  /**
   * Lexical BM25 plus dense embeddings, with load options and a progress listener.
   *
   * @since 1.1.0
   */
  public static HybridRagIndex make(
    final Path folderOrFile,
    final RagLoadOptions options,
    final LlmListener io,
    final LlmModel embeddingModel
  ) {
    return withEmbeddings(make(folderOrFile, options, io), embeddingModel);
  }

  /**
   * Like {@link #tryMake(Path, RagLoadOptions, LlmListener)} then {@link #withEmbeddings} when
   * the corpus is non-empty.
   *
   * @since 1.1.0
   */
  public static Optional<HybridRagIndex> tryMake(
    final Path folderOrFile,
    final RagLoadOptions options,
    final LlmListener io,
    final LlmModel embeddingModel
  ) {
    requireNonNull(embeddingModel, "embeddingModel");
    return tryMake(folderOrFile, options, io)
      .map(lexical -> withEmbeddings(lexical, embeddingModel));
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

    private Builder() {
    }

    /**
     * Chunk/preprocess knobs. Must be set before adding documents.
     */
    public Builder options(final RagLoadOptions options) {
      if (this.hasContent) {
        throw new IllegalStateException("options must be set before adding documents");
      }
      this.options = requireNonNull(options, "options");
      this.corpus.apply(options);
      return this;
    }

    /**
     * {@link RagLoadOptions#forTinyModels()} (shorter chunks for small context windows).
     */
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

    /**
     * Folder used as the corpus root label in logs and {@link PreparedRag#sourceRoot()}.
     */
    public Builder sourceRoot(final Path sourceRoot) {
      this.sourceRoot = requireNonNull(sourceRoot, "sourceRoot").toAbsolutePath().normalize();
      return this;
    }

    /**
     * Filename suffixes visited by {@link #addFolder(Path)} (include the dot, e.g. {@code .md}).
     */
    public Builder folderExtensions(final Set<String> extensions) {
      this.corpus.folderExtensions(extensions);
      return this;
    }

    /**
     * Adds one inline document (id/source assigned automatically).
     */
    public Builder add(final String text) {
      this.hasContent = true;
      this.corpus.add(text);
      return this;
    }

    /**
     * Adds one inline document with an explicit id (source matches id).
     */
    public Builder add(final String id, final String text) {
      this.hasContent = true;
      this.corpus.add(id, text);
      return this;
    }

    /**
     * Adds one file (text or PDF) from disk.
     */
    public Builder addFile(final Path file) {
      this.hasContent = true;
      this.corpus.addFile(file);
      return this;
    }

    /**
     * Absolute classpath resource (no leading {@code /}), e.g. {@code rag/facts.md}.
     *
     * @since 1.1.0
     */
    public Builder addResource(final String resourcePath) {
      this.hasContent = true;
      this.corpus.addResource(resourcePath);
      return this;
    }

    /**
     * Absolute classpath resource resolved with {@code loader}.
     *
     * @since 1.1.0
     */
    public Builder addResource(final ClassLoader loader, final String resourcePath) {
      this.hasContent = true;
      this.corpus.addResource(loader, resourcePath);
      return this;
    }

    /**
     * Classpath resource via {@link Class#getResourceAsStream(String)}.
     *
     * @since 1.1.0
     */
    public Builder addResource(final Class<?> anchor, final String resourcePath) {
      this.hasContent = true;
      this.corpus.addResource(anchor, resourcePath);
      return this;
    }

    /**
     * Adds several absolute classpath resources (no leading {@code /}).
     *
     * @since 1.1.0
     */
    public Builder addResources(final String... resourcePaths) {
      this.hasContent = true;
      this.corpus.addResources(resourcePaths);
      return this;
    }

    /**
     * Adds several classpath resources resolved with {@code loader}.
     *
     * @since 1.1.0
     */
    public Builder addResources(final ClassLoader loader, final String... resourcePaths) {
      this.hasContent = true;
      this.corpus.addResources(loader, resourcePaths);
      return this;
    }

    /**
     * Walks {@code folder} for {@link #folderExtensions(Set)} (sets {@link #sourceRoot(Path)} if
     * unset).
     */
    public Builder addFolder(final Path folder) {
      this.hasContent = true;
      this.corpus.addFolder(folder);
      if (this.sourceRoot == null) {
        this.sourceRoot = folder.toAbsolutePath().normalize();
      }
      return this;
    }

    /**
     * Adds several files from disk.
     */
    public Builder addFiles(Path... files) {
      this.hasContent = true;
      this.corpus.addFiles(files);
      return this;
    }

    /**
     * Builds an immutable lexical {@link PreparedRag}.
     *
     * @throws ModelLoadException if no non-blank chunks were added
     */
    public PreparedRag build() {
      return seal(this.corpus.build(), this.sourceRoot, this.options, this.io);
    }

    /**
     * Lexical index plus dense embeddings.
     *
     * @since 1.1.0
     */
    public HybridRagIndex build(final LlmModel embeddingModel) {
      return withEmbeddings(this.build(), embeddingModel);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
      return "RagFactory.Builder{options=%s, source=%s}".formatted(
        this.options,
        this.sourceRoot == null ? "-" : this.sourceRoot);
    }
  }
}
