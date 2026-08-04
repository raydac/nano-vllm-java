package com.igormaznitsa.nanollvm.rag;

/**
 * Options for {@link RagFactory} when loading and chunking a shared corpus.
 * Tuning is about document shape for the model, not about classifying user replies.
 */
public record RagLoadOptions(
    int maxChunkChars,
    int chunkOverlap,
    boolean preprocess,
    boolean atomicSentences,
    boolean dedupe
) {

  public RagLoadOptions {
    if (maxChunkChars < 64) {
      throw new IllegalArgumentException("maxChunkChars must be >= 64");
    }
    if (chunkOverlap < 0) {
      throw new IllegalArgumentException("chunkOverlap must be >= 0");
    }
  }

  public static RagLoadOptions defaults() {
    return new RagLoadOptions(500, 40, true, false, true);
  }

  /**
   * One sentence per chunk, short windows — better for small generators.
   */
  public static RagLoadOptions forTinyModels() {
    return new RagLoadOptions(220, 0, true, true, true);
  }

  public RagLoadOptions withMaxChunkChars(int maxChunkChars) {
    return new RagLoadOptions(
        maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe);
  }

  public RagLoadOptions withChunkOverlap(int chunkOverlap) {
    return new RagLoadOptions(
        this.maxChunkChars, chunkOverlap, this.preprocess, this.atomicSentences, this.dedupe);
  }

  public RagLoadOptions withPreprocess(boolean preprocess) {
    return new RagLoadOptions(
        this.maxChunkChars, this.chunkOverlap, preprocess, this.atomicSentences, this.dedupe);
  }

  public RagLoadOptions withAtomicSentences(boolean atomicSentences) {
    return new RagLoadOptions(
        this.maxChunkChars, this.chunkOverlap, this.preprocess, atomicSentences, this.dedupe);
  }

  public RagLoadOptions withDedupe(boolean dedupe) {
    return new RagLoadOptions(
        this.maxChunkChars, this.chunkOverlap, this.preprocess, this.atomicSentences, dedupe);
  }
}
