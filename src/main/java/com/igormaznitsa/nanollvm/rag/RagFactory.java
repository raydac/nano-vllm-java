package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

/**
 * Loads and preprocesses documents into a shareable {@link PreparedRag}.
 * Analogous to {@link com.igormaznitsa.nanollvm.ModelFactory} for weights.
 *
 * <p>Preprocessing is document-side only: section titles, sentence passages, load-time
 * preparsing (model vs search text, term frequencies), inverted BM25 — not user-reply rules.
 */
public final class RagFactory {

  private RagFactory() {
  }

  public static PreparedRag make(Path folderOrFile) {
    return make(folderOrFile, RagLoadOptions.defaults());
  }

  public static PreparedRag make(Path folderOrFile, RagLoadOptions options) {
    requireNonNull(folderOrFile, "folderOrFile");
    requireNonNull(options, "options");
    Path path = folderOrFile.toAbsolutePath().normalize();
    TextCorpus.Builder corpus = TextCorpus.builder().apply(options);
    if (Files.isDirectory(path)) {
      corpus.addFolder(path);
    } else if (Files.isRegularFile(path)) {
      corpus.addFile(path);
    } else {
      throw new IllegalArgumentException("path is not a file or directory: " + path);
    }
    return seal(corpus.build(), path, options);
  }

  public static PreparedRag of(String... texts) {
    return of(RagLoadOptions.defaults(), texts);
  }

  public static PreparedRag of(RagLoadOptions options, String... texts) {
    requireNonNull(options, "options");
    requireNonNull(texts, "texts");
    TextCorpus.Builder corpus = TextCorpus.builder().apply(options);
    for (String text : texts) {
      corpus.add(text);
    }
    return seal(corpus.build(), null, options);
  }

  public static PreparedRag of(List<String> texts) {
    return of(RagLoadOptions.defaults(), texts.toArray(String[]::new));
  }

  public static Builder builder() {
    return new Builder();
  }

  private static PreparedRag seal(TextCorpus corpus, Path sourceRoot, RagLoadOptions options) {
    List<PreparedPassage> passages = PassagePreparser.prepare(corpus.chunks());
    return new PreparedRag(passages, Bm25Index.buildPrepared(passages), sourceRoot, options);
  }

  public static final class Builder {

    private RagLoadOptions options = RagLoadOptions.defaults();
    private final TextCorpus.Builder corpus = TextCorpus.builder().apply(this.options);
    private Path sourceRoot;
    private boolean hasContent;

    public Builder options(RagLoadOptions options) {
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

    public Builder sourceRoot(Path sourceRoot) {
      this.sourceRoot = requireNonNull(sourceRoot, "sourceRoot").toAbsolutePath().normalize();
      return this;
    }

    public Builder folderExtensions(Set<String> extensions) {
      this.corpus.folderExtensions(extensions);
      return this;
    }

    public Builder add(String text) {
      this.hasContent = true;
      this.corpus.add(text);
      return this;
    }

    public Builder add(String id, String text) {
      this.hasContent = true;
      this.corpus.add(id, text);
      return this;
    }

    public Builder addFile(Path file) {
      this.hasContent = true;
      this.corpus.addFile(file);
      return this;
    }

    public Builder addFolder(Path folder) {
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
      return seal(this.corpus.build(), this.sourceRoot, this.options);
    }

    @Override
    public String toString() {
      return "RagFactory.Builder{options=%s, source=%s}".formatted(
          this.options,
          this.sourceRoot == null ? "-" : this.sourceRoot);
    }
  }
}
