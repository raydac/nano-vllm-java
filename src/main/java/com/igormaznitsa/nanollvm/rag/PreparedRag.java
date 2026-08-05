package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Immutable, shareable RAG corpus: preparsed passages + inverted BM25 index.
 * Load once via {@link RagFactory}; reuse across many LLM sessions.
 */
public final class PreparedRag implements RagIndex {

  private final List<PreparedPassage> passages;
  private final TextCorpus corpus;
  private final Bm25Index index;
  private final Path sourceRoot;
  private final RagLoadOptions options;

  PreparedRag(
      List<PreparedPassage> passages,
      Bm25Index index,
      Path sourceRoot,
      RagLoadOptions options
  ) {
    this.passages = List.copyOf(requireNonNull(passages, "passages"));
    this.index = requireNonNull(index, "index");
    this.sourceRoot = sourceRoot;
    this.options = requireNonNull(options, "options");
    this.corpus = TextCorpus.ofChunks(this.passages.stream().map(PreparedPassage::chunk).toList());
  }

  /**
   * Fraction of distinct query terms that appear in the passage (0..1).
   * Favors fact cards that mention {@code fairy}/{@code fable}/… over long narration
   * that only matches the story title tokens.
   */
  static double termCoverage(String passageText, List<String> queryTerms) {
    if (queryTerms.isEmpty()) {
      return 0.0;
    }
    Set<String> passageTerms = new LinkedHashSet<>(Bm25Index.tokenize(passageText));
    long hit = queryTerms.stream().filter(passageTerms::contains).count();
    return hit / (double) queryTerms.size();
  }

  public List<PreparedPassage> passages() {
    return this.passages;
  }

  public TextCorpus corpus() {
    return this.corpus;
  }

  public Bm25Index bm25() {
    return this.index;
  }

  public Optional<Path> sourceRoot() {
    return Optional.ofNullable(this.sourceRoot);
  }

  public RagLoadOptions options() {
    return this.options;
  }

  public int size() {
    return this.index.size();
  }

  @Override
  public List<RagHit> retrieve(String query, int topK) {
    List<String> terms = this.index.selectedQueryTerms(query);
    List<RagHit> hits = this.index.retrieve(query, Math.max(topK * 4, topK));
    return hits.stream()
        .map(hit -> new RagHit(hit.chunk(),
            hit.score() * (1.0 + termCoverage(hit.chunk().text(), terms))))
        .sorted(Comparator
            .comparingDouble(RagHit::score).reversed()
            .thenComparingInt(hit -> hit.chunk().text().length()))
        .limit(topK)
        .toList();
  }

  @Override
  public String toString() {
    return "PreparedRag{passages=%d, source=%s}".formatted(
        this.size(),
        this.sourceRoot == null ? "inline" : this.sourceRoot);
  }
}
