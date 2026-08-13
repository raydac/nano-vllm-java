package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.utils.ResourceLimits;

/**
 * Options for {@link RagFactory} when loading and chunking a shared corpus.
 *
 * <p>Tuning is about document shape for the model, not about classifying user replies.
 * {@link #defaults()} is the usual starting point; {@link #forTinyModels()} uses shorter
 * sentence-sized chunks. {@link #resourceLimits()} caps file size, corpus totals, and PDF inflate
 * budgets (defaults from {@link ResourceLimits#current()}). {@code with*} methods return a copy;
 * this type is immutable.
 *
 * @param maxChunkChars    maximum characters per chunk; must be {@code >= 64}
 * @param chunkOverlap     characters reused at the start of the next window; must be {@code >= 0}
 *                         (ignored when {@code atomicSentences} is {@code true})
 * @param preprocess       when {@code true}, normalize whitespace / split into sentence-like units
 *                         before packing; {@code false} uses raw sliding windows on the file text
 * @param atomicSentences  when {@code true}, one sentence per chunk (better for tiny generators);
 *                         long sentences still window-split
 * @param dedupe           when {@code true}, drop duplicate chunk bodies after load
 * @param resourceLimits   parser / corpus / PDF budgets; never {@code null}
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

  /**
   * Default corpus shape: 500-char chunks, 40-char overlap, preprocess and dedupe on, process
   * {@link ResourceLimits#current()}.
   */
  public static RagLoadOptions defaults() {
    return new RagLoadOptions(500, 40, true, false, true, ResourceLimits.current());
  }

  /**
   * One sentence per chunk, short windows — better for small generators (SmolLM2 / Tiny-LLM).
   */
  public static RagLoadOptions forTinyModels() {
    return new RagLoadOptions(220, 0, true, true, true, ResourceLimits.current());
  }

  /**
   * Copy with a new chunk size ({@code >= 64}).
   */
  public RagLoadOptions withMaxChunkChars(final int maxChunkChars) {
    return new RagLoadOptions(
      maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy with a new window overlap ({@code >= 0}).
   */
  public RagLoadOptions withChunkOverlap(final int chunkOverlap) {
    return new RagLoadOptions(
      this.maxChunkChars, chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy that enables or disables sentence-unit preprocessing.
   */
  public RagLoadOptions withPreprocess(final boolean preprocess) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy that enables or disables one-sentence chunks.
   */
  public RagLoadOptions withAtomicSentences(final boolean atomicSentences) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy that enables or disables duplicate-body removal.
   */
  public RagLoadOptions withDedupe(final boolean dedupe) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, dedupe,
      this.resourceLimits);
  }

  /**
   * Copy with different parser / corpus budgets.
   *
   * @param resourceLimits must not be {@code null}
   */
  public RagLoadOptions withResourceLimits(final ResourceLimits resourceLimits) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      requireNonNull(resourceLimits, "resourceLimits"));
  }
}
