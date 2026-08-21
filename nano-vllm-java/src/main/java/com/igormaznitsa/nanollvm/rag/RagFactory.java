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
import java.util.concurrent.Executor;

/**
 * Loads and preprocesses documents into a shareable {@link PreparedRag}.
 * Analogous to {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} for weights.
 *
 * <p>{@link #make(java.nio.file.Path)} indexes a file or folder with
 * {@link RagLoadOptions#defaults()}. {@link #builder()} mixes several folders, files, inline
 * text, and classpath resources into one BM25 index (do not fuse separate {@link PreparedRag}
 * instances with {@link HybridRagIndex} for that). Optional {@link #withEmbeddings} adds vector
 * search on top of BM25.
 *
 * <p>Chunk size and sentence packing live on {@link RagLoadOptions}: pass them to
 * {@link #make(Path, RagLoadOptions)} or {@link Builder#options(RagLoadOptions)} before adding
 * documents. Path-only {@link #make(Path)} uses {@link RagLoadOptions#defaults()} (500-char
 * ceiling). Preprocessing is document-side only: section titles, sentence passages, load-time
 * preparsing (model vs search text, term frequencies), inverted BM25 — not user-reply rules.
 * Pass {@link LlmListeners#toSystem()} to print per-file extraction stats while loading.
 * Optional embedding models produce a {@link HybridRagIndex} via {@link #withEmbeddings}.
 * Classpath documents use {@link Builder#addResource(String)} / {@link #makeResource(String)}.
 * Optional {@link Builder#addProcessor(RagTuner...)} hooks filter files, override text extraction,
 * and rewrite loaded text before chunking.
 *
 * <p>Empty corpora throw {@link ModelLoadException}. Lexical indexes are immutable and safe to
 * share across threads; hybrid indexes additionally need a live embedding {@link LlmModel}.
 */
public final class RagFactory {

  private RagFactory() {
  }

  /**
   * Indexes a file or folder with {@link RagLoadOptions#defaults()} (500-char packed sentences).
   * Override with {@link #make(Path, RagLoadOptions)}.
   *
   * @param folderOrFile file or directory to index; must exist
   * @return immutable BM25 index
   * @throws ModelLoadException if the path yields no indexable chunks
   */
  public static PreparedRag make(final Path folderOrFile) {
    return make(folderOrFile, RagLoadOptions.defaults(), LlmListeners.silent());
  }

  /**
   * Indexes a file or folder with explicit {@link RagLoadOptions} (chunk ceiling, overlap,
   * sentence packing).
   *
   * @param folderOrFile file or directory to index; must exist
   * @param options      load-time chunk/preprocess knobs; must not be {@code null}
   * @return immutable BM25 index
   * @throws ModelLoadException if the path yields no indexable chunks
   */
  public static PreparedRag make(final Path folderOrFile, final RagLoadOptions options) {
    return make(folderOrFile, options, LlmListeners.silent());
  }

  /**
   * Indexes a file or folder, emitting per-file load lines to {@code io}.
   *
   * @param folderOrFile file or directory to index; must exist
   * @param options      load-time chunk/preprocess knobs; must not be {@code null}
   * @param io           progress sink; {@code null} is treated as silent
   * @return immutable BM25 index
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
   * yet yields no indexable text (empty folder, only README, blank files). Uses
   * {@link RagLoadOptions#defaults()}.
   *
   * @param folderOrFile file or directory to index; must exist
   * @return the index, or empty when nothing was indexable
   */
  public static Optional<PreparedRag> tryMake(final Path folderOrFile) {
    return tryMake(folderOrFile, RagLoadOptions.defaults(), LlmListeners.silent());
  }

  /**
   * Like {@link #tryMake(Path, RagLoadOptions, LlmListener)} with a silent listener.
   *
   * @param folderOrFile file or directory to index; must exist
   * @param options      load-time chunk/preprocess knobs; must not be {@code null}
   * @return the index, or empty when nothing was indexable
   */
  public static Optional<PreparedRag> tryMake(final Path folderOrFile,
                                              final RagLoadOptions options) {
    return tryMake(folderOrFile, options, LlmListeners.silent());
  }

  /**
   * Like {@link #make(Path, RagLoadOptions, LlmListener)} but returns empty when the path exists
   * yet yields no indexable text (empty folder, only README, blank files).
   *
   * @param folderOrFile file or directory to index; must exist
   * @param options      load-time chunk/preprocess knobs; must not be {@code null}
   * @param io           progress sink; {@code null} is treated as silent
   * @return the index, or empty when nothing was indexable
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
   * @param texts document bodies; each is chunked independently
   * @return immutable BM25 index
   * @throws ModelLoadException if every text is blank
   */
  public static PreparedRag of(String... texts) {
    return of(RagLoadOptions.defaults(), texts);
  }

  /**
   * Inline documents with explicit {@link RagLoadOptions} (chunk ceiling, overlap, sentence
   * packing).
   *
   * @param options load-time chunk/preprocess knobs; must not be {@code null}
   * @param texts   document bodies; each is chunked independently
   * @return immutable BM25 index
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
   * {@link #of(String...)} from a list ({@link RagLoadOptions#defaults()}).
   *
   * @param texts document bodies; each is chunked independently
   * @return immutable BM25 index
   * @throws ModelLoadException if every text is blank
   */
  public static PreparedRag of(final List<String> texts) {
    return of(RagLoadOptions.defaults(), texts.toArray(String[]::new));
  }

  /**
   * Fluent corpus builder (inline text, files, one or more folders, classpath resources). Set
   * {@link Builder#options(RagLoadOptions)} before adding documents to change chunk size.
   * Optional {@link Builder#addProcessor(RagTuner...)} registers load-time tuners.
   *
   * @return a new builder using {@link RagLoadOptions#defaults()}
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Loads one absolute classpath resource into a BM25 index
   * ({@link RagLoadOptions#defaults()}). Use {@link #builder()} to pass other options.
   *
   * @param resourcePath absolute classpath path, no leading {@code /}
   * @return immutable BM25 index
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
   * Hybrid BM25 + dense retrieval over an existing lexical index. Dense passage embeds run on
   * the calling thread.
   *
   * @param lexical         BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel  embedding encoder kept open for query-time embed; must not be {@code null}
   * @return hybrid index over the same passages
   * @throws NullPointerException     if either argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed
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
   * {@link #withEmbeddings(PreparedRag, LlmModel)} with dense passage embeds submitted on
   * {@code executor}. The caller owns the executor; it is not shut down here.
   *
   * @param lexical        BM25 corpus whose chunks are embedded; must not be {@code null}
   * @param embeddingModel embedding encoder kept open for query-time embed; must not be {@code null}
   * @param executor       runs each dense passage embed; must not be {@code null}
   * @return hybrid index over the same passages
   * @throws NullPointerException     if any argument is {@code null}
   * @throws IllegalArgumentException if {@code lexical} has no chunks or {@code embeddingModel} is
   *                                  not an embedding encoder
   * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
   * @since 1.2.0
   */
  public static HybridRagIndex withEmbeddings(
    final PreparedRag lexical,
    final LlmModel embeddingModel,
    final Executor executor
  ) {
    requireNonNull(lexical, "lexical");
    requireNonNull(embeddingModel, "embeddingModel");
    return HybridRagIndex.of(lexical, embeddingModel, requireNonNull(executor, "executor"));
  }

  /**
   * Lexical BM25 corpus plus dense embeddings from {@code embeddingModel}
   * ({@link RagLoadOptions#defaults()}).
   *
   * @param folderOrFile   file or directory to index; must exist
   * @param embeddingModel encoder kept open for query-time embed; must not be {@code null}
   * @return hybrid index over the same passages
   * @since 1.1.0
   */
  public static HybridRagIndex make(final Path folderOrFile, final LlmModel embeddingModel) {
    return withEmbeddings(make(folderOrFile), embeddingModel);
  }

  /**
   * Lexical BM25 plus dense embeddings, with corpus {@link RagLoadOptions}.
   *
   * @param folderOrFile    file or directory to index; must exist
   * @param options         load-time chunk/preprocess knobs; must not be {@code null}
   * @param embeddingModel  encoder kept open for query-time embed; must not be {@code null}
   * @return hybrid index over the same passages
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
   * @param folderOrFile    file or directory to index; must exist
   * @param options         load-time chunk/preprocess knobs; must not be {@code null}
   * @param io              progress sink; {@code null} is treated as silent
   * @param embeddingModel  encoder kept open for query-time embed; must not be {@code null}
   * @return hybrid index over the same passages
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
   * @param folderOrFile    file or directory to index; must exist
   * @param options         load-time chunk/preprocess knobs; must not be {@code null}
   * @param io              progress sink; {@code null} is treated as silent
   * @param embeddingModel  encoder kept open for query-time embed; must not be {@code null}
   * @return the hybrid index, or empty when nothing was indexable
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

  /**
   * Fluent corpus assembler. {@link RagLoadOptions#defaults()} until {@link #options(RagLoadOptions)}
   * or {@link #forTinyModels()}; those must run before adding documents. Optional
   * {@link #addProcessor(RagTuner...)} may be called at any time and applies to documents added
   * afterwards. Mix {@link #addFolder(Path)} / {@link #addFolders(Path...)},
   * {@link #addResource(String)}, and {@link #add(String)} on one builder.
   */
  public static final class Builder {

    private RagLoadOptions options = RagLoadOptions.defaults();
    private final CorpusLoader.Builder corpus = CorpusLoader.builder().apply(this.options);
    private Path sourceRoot;
    private boolean hasContent;
    private LlmListener io = LlmListeners.silent();

    /**
     * Defaults: {@link RagLoadOptions#defaults()}, silent listener, no documents.
     */
    private Builder() {
    }

    /**
     * Replaces {@link RagLoadOptions#defaults()} for this builder. Call before adding documents.
     * Use {@link RagLoadOptions#withMaxChunkChars(int)} to change the character ceiling.
     *
     * @param options must not be {@code null}
     * @return {@code this}
     * @throws IllegalStateException if documents were already added
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
     * {@link RagLoadOptions#forTinyModels()} (220-char one-sentence chunks). Must run before
     * adding documents.
     *
     * @return {@code this}
     * @throws IllegalStateException if documents were already added
     */
    public Builder forTinyModels() {
      return this.options(RagLoadOptions.forTinyModels());
    }

    /**
     * Progress sink for per-file load lines. {@code null} → {@link LlmListeners#silent()}.
     *
     * @param io listener, or {@code null} for silent
     * @return {@code this}
     */
    public Builder listen(final LlmListener io) {
      this.io = io == null ? LlmListeners.silent() : io;
      this.corpus.listen(this.io);
      return this;
    }

    /**
     * Adds load-time {@link RagTuner}s (filter, extract, preprocess). Applied in order to
     * documents added after this call. Folder walks still honor {@link #folderExtensions(Set)};
     * include extra suffixes for custom extractors.
     *
     * @param tuners must not contain {@code null}; empty is a no-op
     * @return {@code this}
     * @throws NullPointerException if {@code tuners} or an element is {@code null}
     * @since 1.2.0
     */
    public Builder addProcessor(final RagTuner... tuners) {
      this.corpus.addProcessor(tuners);
      return this;
    }

    /**
     * Folder used as the corpus root label in logs and {@link PreparedRag#sourceRoot()}.
     *
     * @param sourceRoot directory or file label; must not be {@code null}
     * @return {@code this}
     * @throws NullPointerException if {@code sourceRoot} is {@code null}
     */
    public Builder sourceRoot(final Path sourceRoot) {
      this.sourceRoot = requireNonNull(sourceRoot, "sourceRoot").toAbsolutePath().normalize();
      return this;
    }

    /**
     * Filename suffixes visited by {@link #addFolder(Path)} (include the dot, e.g. {@code .md}).
     *
     * @param extensions suffixes with or without a leading dot; must not be {@code null}
     * @return {@code this}
     * @throws NullPointerException if {@code extensions} is {@code null}
     */
    public Builder folderExtensions(final Set<String> extensions) {
      this.corpus.folderExtensions(extensions);
      return this;
    }

    /**
     * Adds one inline document (id/source assigned automatically).
     *
     * @param text document body
     * @return {@code this}
     */
    public Builder add(final String text) {
      this.hasContent = true;
      this.corpus.add(text);
      return this;
    }

    /**
     * Adds one inline document with an explicit id (source matches id).
     *
     * @param id   non-blank document id
     * @param text document body
     * @return {@code this}
     */
    public Builder add(final String id, final String text) {
      this.hasContent = true;
      this.corpus.add(id, text);
      return this;
    }

    /**
     * Adds one file from disk (UTF-8 unless a {@link RagTuner} extractor supplies text).
     * {@link RagTuner#isRagResourceAllowed} may skip it.
     *
     * @param file regular file; must not be {@code null}
     * @return {@code this}
     */
    public Builder addFile(final Path file) {
      this.hasContent = true;
      this.corpus.addFile(file);
      return this;
    }

    /**
     * Absolute classpath resource (no leading {@code /}), e.g. {@code rag/facts.md}.
     *
     * @param resourcePath classpath path; must not be blank
     * @return {@code this}
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
     * @param loader       class loader; must not be {@code null}
     * @param resourcePath classpath path; must not be blank
     * @return {@code this}
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
     * @param anchor       class used to resolve the path; must not be {@code null}
     * @param resourcePath leading {@code /} = absolute, otherwise package-relative
     * @return {@code this}
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
     * @param resourcePaths classpath paths; must not be {@code null}
     * @return {@code this}
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
     * @param loader         class loader; must not be {@code null}
     * @param resourcePaths  classpath paths; must not be {@code null}
     * @return {@code this}
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
     *
     * @param folder directory to scan; must not be {@code null}
     * @return {@code this}
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
     * Walks each directory like {@link #addFolder(Path)}. The first folder becomes
     * {@link #sourceRoot(Path)} when none is set.
     *
     * @param folders directories to scan; must not be {@code null}
     * @return {@code this}
     * @throws NullPointerException     if {@code folders} or an element is {@code null}
     * @throws IllegalArgumentException if an element is not a directory
     * @since 1.2.0
     */
    public Builder addFolders(final Path... folders) {
      requireNonNull(folders, "folders");
      this.hasContent = true;
      for (Path folder : folders) {
        this.addFolder(folder);
      }
      return this;
    }

    /**
     * Adds several files from disk.
     *
     * @param files regular files; must not be {@code null}
     * @return {@code this}
     */
    public Builder addFiles(Path... files) {
      this.hasContent = true;
      this.corpus.addFiles(files);
      return this;
    }

    /**
     * Builds an immutable lexical {@link PreparedRag}.
     *
     * @return BM25 index
     * @throws ModelLoadException if no non-blank chunks were added
     */
    public PreparedRag build() {
      return seal(this.corpus.build(), this.sourceRoot, this.options, this.io);
    }

    /**
     * Lexical index plus dense embeddings.
     *
     * @param embeddingModel encoder kept open for query-time embed; must not be {@code null}
     * @return hybrid index over the same passages
     * @since 1.1.0
     */
    public HybridRagIndex build(final LlmModel embeddingModel) {
      return withEmbeddings(this.build(), embeddingModel);
    }

    /**
     * {@link #build(LlmModel)} with dense passage embeds submitted on {@code executor}.
     * The caller owns the executor; it is not shut down here.
     *
     * @param embeddingModel encoder kept open for query-time embed; must not be {@code null}
     * @param executor       runs each dense passage embed; must not be {@code null}
     * @return hybrid index over the same passages
     * @throws NullPointerException     if either argument is {@code null}
     * @throws IllegalArgumentException if the corpus is empty or {@code embeddingModel} is not an
     *                                  embedding encoder
     * @throws IllegalStateException    if {@code embeddingModel} is closed, or embedding is interrupted
     * @throws ModelLoadException       if no non-blank chunks were added
     * @since 1.2.0
     */
    public HybridRagIndex build(final LlmModel embeddingModel, final Executor executor) {
      return withEmbeddings(this.build(), embeddingModel, executor);
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
