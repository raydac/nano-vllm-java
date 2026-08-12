package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.utils.ResourceLimits;

/**
 * Options for {@link RagFactory} when loading and chunking a shared corpus.
 * Tuning is about document shape for the model, not about classifying user replies.
 *
 * <p>{@link #resourceLimits()} caps file size, corpus totals, and PDF inflate budgets
 * (defaults from {@link ResourceLimits#current()}).
 */
public record RagLoadOptions(
  int maxChunkChars,
  int chunkOverlap,
  boolean preprocess,
  boolean atomicSentences,
  boolean dedupe,
  ResourceLimits resourceLimits
) {

  public RagLoadOptions {
    if (maxChunkChars < 64) {
      throw new IllegalArgumentException("maxChunkChars must be >= 64");
    }
    if (chunkOverlap < 0) {
      throw new IllegalArgumentException("chunkOverlap must be >= 0");
    }
    requireNonNull(resourceLimits, "resourceLimits");
  }

  public static RagLoadOptions defaults() {
    return new RagLoadOptions(500, 40, true, false, true, ResourceLimits.current());
  }

  /**
   * One sentence per chunk, short windows — better for small generators.
   */
  public static RagLoadOptions forTinyModels() {
    return new RagLoadOptions(220, 0, true, true, true, ResourceLimits.current());
  }

  public RagLoadOptions withMaxChunkChars(final int maxChunkChars) {
    return new RagLoadOptions(
      maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  public RagLoadOptions withChunkOverlap(final int chunkOverlap) {
    return new RagLoadOptions(
      this.maxChunkChars, chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  public RagLoadOptions withPreprocess(final boolean preprocess) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  public RagLoadOptions withAtomicSentences(final boolean atomicSentences) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  public RagLoadOptions withDedupe(final boolean dedupe) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, dedupe,
      this.resourceLimits);
  }

  public RagLoadOptions withResourceLimits(final ResourceLimits resourceLimits) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      requireNonNull(resourceLimits, "resourceLimits"));
  }
}
