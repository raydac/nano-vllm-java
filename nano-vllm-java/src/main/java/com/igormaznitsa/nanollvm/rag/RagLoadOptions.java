package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.utils.ResourceLimits;

/**
 * Load-time knobs for how {@link RagFactory} cuts documents into {@link TextChunk}s.
 *
 * <p>{@link #maxChunkChars()} is a <em>character</em> ceiling (Java {@code char}s, not tokens, not
 * a target length). With {@link #preprocess()} on, the loader splits on sentences and packs them
 * until the next sentence would exceed the ceiling — so most chunks are shorter than
 * {@code maxChunkChars}. {@link #chunkOverlap()} applies only to sliding windows on leftover long
 * text (and when preprocess is off). Packed sentence chunks do not overlap;
 * {@link #atomicSentences()} ignores overlap.
 *
 * <p>Start from a preset, then copy: {@link #defaults()} (500 / 40, packed sentences) or
 * {@link #forTinyModels()} (220 / 0, one sentence per chunk). Change the ceiling with
 * {@link #withMaxChunkChars(int)}. Pass the result into
 * {@link RagFactory#make(java.nio.file.Path, RagLoadOptions)} or
 * {@link RagFactory.Builder#options(RagLoadOptions)} <em>before</em> adding documents. Prompt
 * concatenation is a separate cap: {@link RagSession#maxContextChars(int)}.
 *
 * <pre>{@code
 * PreparedRag rag = RagFactory.make(
 *     Path.of("docs"),
 *     RagLoadOptions.defaults().withMaxChunkChars(800).withChunkOverlap(80));
 * }</pre>
 *
 * <p>{@code with*} methods return a copy; this type is immutable.
 * {@link #resourceLimits()} caps file size, corpus totals, and PDF inflate
 * (defaults from {@link ResourceLimits#current()}).
 *
 * @param maxChunkChars   maximum characters per chunk ({@code >= 64}); packing may emit shorter
 * @param chunkOverlap    characters reused at the start of the next sliding window ({@code >= 0});
 *                        unused for packed sentences; ignored when {@code atomicSentences} is
 *                        {@code true}
 * @param preprocess      when {@code true}, normalize whitespace / split into sentence-like units
 *                        before packing; {@code false} uses raw sliding windows on the file text
 * @param atomicSentences when {@code true}, one sentence per chunk (better for tiny generators);
 *                        long sentences still window-split
 * @param dedupe          when {@code true}, drop duplicate chunk bodies after load
 * @param resourceLimits  parser / corpus / PDF budgets; never {@code null}
 */
public record RagLoadOptions(
  int maxChunkChars,
  int chunkOverlap,
  boolean preprocess,
  boolean atomicSentences,
  boolean dedupe,
  ResourceLimits resourceLimits
) {

  /**
   * @throws IllegalArgumentException if {@code maxChunkChars < 64} or {@code chunkOverlap < 0}
   * @throws NullPointerException     if {@code resourceLimits} is {@code null}
   */
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
   * Packed-sentence corpus: {@code maxChunkChars = 500}, {@code chunkOverlap = 40}, preprocess and
   * dedupe on, {@code atomicSentences} off, process {@link ResourceLimits#current()}.
   *
   * @return immutable options; copy with {@link #withMaxChunkChars(int)} to change the ceiling
   */
  public static RagLoadOptions defaults() {
    return new RagLoadOptions(500, 40, true, false, true, ResourceLimits.current());
  }

  /**
   * One sentence per chunk for small generators (SmolLM2 / Tiny-LLM / Gemma 3 270M-class):
   * {@code maxChunkChars = 220}, {@code chunkOverlap = 0}, preprocess, atomic sentences, and
   * dedupe on, process {@link ResourceLimits#current()}. Sentences longer than 220 still
   * window-split.
   *
   * @return immutable options
   */
  public static RagLoadOptions forTinyModels() {
    return new RagLoadOptions(220, 0, true, true, true, ResourceLimits.current());
  }

  /**
   * Copy with a new character ceiling. Units are Java {@code char}s, not tokens. Existing indexes
   * are unchanged — rebuild with {@link RagFactory} to apply.
   *
   * @param maxChunkChars must be {@code >= 64}
   * @return a new instance; {@code this} is unchanged
   * @throws IllegalArgumentException if {@code maxChunkChars < 64}
   */
  public RagLoadOptions withMaxChunkChars(final int maxChunkChars) {
    return new RagLoadOptions(
      maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy with a new sliding-window overlap. Unused for packed sentence chunks; ignored when
   * {@link #atomicSentences()} is {@code true}.
   *
   * @param chunkOverlap must be {@code >= 0}
   * @return a new instance; {@code this} is unchanged
   * @throws IllegalArgumentException if {@code chunkOverlap < 0}
   */
  public RagLoadOptions withChunkOverlap(final int chunkOverlap) {
    return new RagLoadOptions(
      this.maxChunkChars, chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy that enables or disables sentence-unit preprocessing before packing.
   *
   * @param preprocess {@code true} to split/pack sentences; {@code false} for raw sliding windows
   * @return a new instance; {@code this} is unchanged
   */
  public RagLoadOptions withPreprocess(final boolean preprocess) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, preprocess, this.atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy that enables or disables one-sentence chunks (long sentences still window-split).
   *
   * @param atomicSentences {@code true} for one sentence per chunk
   * @return a new instance; {@code this} is unchanged
   */
  public RagLoadOptions withAtomicSentences(final boolean atomicSentences) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, atomicSentences, this.dedupe,
      this.resourceLimits);
  }

  /**
   * Copy that enables or disables duplicate-body removal after chunking.
   *
   * @param dedupe {@code true} to keep the first copy of each normalized body
   * @return a new instance; {@code this} is unchanged
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
   * @return a new instance; {@code this} is unchanged
   * @throws NullPointerException if {@code resourceLimits} is {@code null}
   */
  public RagLoadOptions withResourceLimits(final ResourceLimits resourceLimits) {
    return new RagLoadOptions(
      this.maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe,
      requireNonNull(resourceLimits, "resourceLimits"));
  }
}
