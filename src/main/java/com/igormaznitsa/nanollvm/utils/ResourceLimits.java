package com.igormaznitsa.nanollvm.utils;

import static java.util.Objects.requireNonNull;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Process-wide and per-call budgets for parsers and corpus loaders.
 *
 * <p>Defaults protect against accidental or hostile oversized inputs (OOM / hang). Replace the
 * process default with {@link #setCurrent(ResourceLimits)} or pass a custom instance into RAG /
 * load APIs that accept one. Values of {@code 0} or negative mean “use the field’s built-in
 * absolute ceiling” only where noted; prefer positive limits.
 */
public record ResourceLimits(
  long maxFileBytes,
  long maxTotalCorpusBytes,
  int maxCorpusFiles,
  long maxPdfInflateBytes,
  int maxCmapRangeSpan,
  int maxCmapEntries,
  long maxSafetensorsHeaderBytes,
  int maxJsonDepth,
  long maxJsonChars,
  long maxGgufStringBytes,
  int maxGgufDims,
  int maxHistoryMessages
) {

  public static final long DEFAULT_MAX_FILE_BYTES = 32L * 1024 * 1024;
  public static final long DEFAULT_MAX_TOTAL_CORPUS_BYTES = 256L * 1024 * 1024;
  public static final int DEFAULT_MAX_CORPUS_FILES = 10_000;
  public static final long DEFAULT_MAX_PDF_INFLATE_BYTES = 32L * 1024 * 1024;
  public static final int DEFAULT_MAX_CMAP_RANGE_SPAN = 65_536;
  public static final int DEFAULT_MAX_CMAP_ENTRIES = 1_000_000;
  public static final long DEFAULT_MAX_SAFETENSORS_HEADER_BYTES = 100L * 1024 * 1024;
  public static final int DEFAULT_MAX_JSON_DEPTH = 64;
  public static final long DEFAULT_MAX_JSON_CHARS = 64L * 1024 * 1024;
  public static final long DEFAULT_MAX_GGUF_STRING_BYTES = 16L * 1024 * 1024;
  public static final int DEFAULT_MAX_GGUF_DIMS = 8;
  public static final int DEFAULT_MAX_HISTORY_MESSAGES = 200;

  private static final AtomicReference<ResourceLimits> CURRENT =
    new AtomicReference<>(defaults());

  public ResourceLimits {
    if (maxFileBytes < 1) {
      throw new IllegalArgumentException("maxFileBytes must be >= 1");
    }
    if (maxTotalCorpusBytes < 1) {
      throw new IllegalArgumentException("maxTotalCorpusBytes must be >= 1");
    }
    if (maxCorpusFiles < 1) {
      throw new IllegalArgumentException("maxCorpusFiles must be >= 1");
    }
    if (maxPdfInflateBytes < 1) {
      throw new IllegalArgumentException("maxPdfInflateBytes must be >= 1");
    }
    if (maxCmapRangeSpan < 1) {
      throw new IllegalArgumentException("maxCmapRangeSpan must be >= 1");
    }
    if (maxCmapEntries < 1) {
      throw new IllegalArgumentException("maxCmapEntries must be >= 1");
    }
    if (maxSafetensorsHeaderBytes < 1) {
      throw new IllegalArgumentException("maxSafetensorsHeaderBytes must be >= 1");
    }
    if (maxJsonDepth < 1) {
      throw new IllegalArgumentException("maxJsonDepth must be >= 1");
    }
    if (maxJsonChars < 1) {
      throw new IllegalArgumentException("maxJsonChars must be >= 1");
    }
    if (maxGgufStringBytes < 1) {
      throw new IllegalArgumentException("maxGgufStringBytes must be >= 1");
    }
    if (maxGgufDims < 1) {
      throw new IllegalArgumentException("maxGgufDims must be >= 1");
    }
    if (maxHistoryMessages < 1) {
      throw new IllegalArgumentException("maxHistoryMessages must be >= 1");
    }
  }

  public static ResourceLimits defaults() {
    return new ResourceLimits(
      DEFAULT_MAX_FILE_BYTES,
      DEFAULT_MAX_TOTAL_CORPUS_BYTES,
      DEFAULT_MAX_CORPUS_FILES,
      DEFAULT_MAX_PDF_INFLATE_BYTES,
      DEFAULT_MAX_CMAP_RANGE_SPAN,
      DEFAULT_MAX_CMAP_ENTRIES,
      DEFAULT_MAX_SAFETENSORS_HEADER_BYTES,
      DEFAULT_MAX_JSON_DEPTH,
      DEFAULT_MAX_JSON_CHARS,
      DEFAULT_MAX_GGUF_STRING_BYTES,
      DEFAULT_MAX_GGUF_DIMS,
      DEFAULT_MAX_HISTORY_MESSAGES);
  }

  public static ResourceLimits current() {
    return CURRENT.get();
  }

  public static void setCurrent(final ResourceLimits limits) {
    CURRENT.set(requireNonNull(limits, "limits"));
  }

  public static void resetCurrent() {
    CURRENT.set(defaults());
  }

  public static Builder builder() {
    return new Builder(current());
  }

  public ResourceLimits withMaxFileBytes(final long maxFileBytes) {
    return new ResourceLimits(
      maxFileBytes, this.maxTotalCorpusBytes, this.maxCorpusFiles, this.maxPdfInflateBytes,
      this.maxCmapRangeSpan, this.maxCmapEntries, this.maxSafetensorsHeaderBytes,
      this.maxJsonDepth, this.maxJsonChars, this.maxGgufStringBytes, this.maxGgufDims,
      this.maxHistoryMessages);
  }

  public ResourceLimits withMaxTotalCorpusBytes(final long maxTotalCorpusBytes) {
    return new ResourceLimits(
      this.maxFileBytes, maxTotalCorpusBytes, this.maxCorpusFiles, this.maxPdfInflateBytes,
      this.maxCmapRangeSpan, this.maxCmapEntries, this.maxSafetensorsHeaderBytes,
      this.maxJsonDepth, this.maxJsonChars, this.maxGgufStringBytes, this.maxGgufDims,
      this.maxHistoryMessages);
  }

  public ResourceLimits withMaxCorpusFiles(final int maxCorpusFiles) {
    return new ResourceLimits(
      this.maxFileBytes, this.maxTotalCorpusBytes, maxCorpusFiles, this.maxPdfInflateBytes,
      this.maxCmapRangeSpan, this.maxCmapEntries, this.maxSafetensorsHeaderBytes,
      this.maxJsonDepth, this.maxJsonChars, this.maxGgufStringBytes, this.maxGgufDims,
      this.maxHistoryMessages);
  }

  public ResourceLimits withMaxPdfInflateBytes(final long maxPdfInflateBytes) {
    return new ResourceLimits(
      this.maxFileBytes, this.maxTotalCorpusBytes, this.maxCorpusFiles, maxPdfInflateBytes,
      this.maxCmapRangeSpan, this.maxCmapEntries, this.maxSafetensorsHeaderBytes,
      this.maxJsonDepth, this.maxJsonChars, this.maxGgufStringBytes, this.maxGgufDims,
      this.maxHistoryMessages);
  }

  public ResourceLimits withMaxHistoryMessages(final int maxHistoryMessages) {
    return new ResourceLimits(
      this.maxFileBytes, this.maxTotalCorpusBytes, this.maxCorpusFiles, this.maxPdfInflateBytes,
      this.maxCmapRangeSpan, this.maxCmapEntries, this.maxSafetensorsHeaderBytes,
      this.maxJsonDepth, this.maxJsonChars, this.maxGgufStringBytes, this.maxGgufDims,
      maxHistoryMessages);
  }

  public static final class Builder {
    private long maxFileBytes;
    private long maxTotalCorpusBytes;
    private int maxCorpusFiles;
    private long maxPdfInflateBytes;
    private int maxCmapRangeSpan;
    private int maxCmapEntries;
    private long maxSafetensorsHeaderBytes;
    private int maxJsonDepth;
    private long maxJsonChars;
    private long maxGgufStringBytes;
    private int maxGgufDims;
    private int maxHistoryMessages;

    private Builder(final ResourceLimits base) {
      this.maxFileBytes = base.maxFileBytes;
      this.maxTotalCorpusBytes = base.maxTotalCorpusBytes;
      this.maxCorpusFiles = base.maxCorpusFiles;
      this.maxPdfInflateBytes = base.maxPdfInflateBytes;
      this.maxCmapRangeSpan = base.maxCmapRangeSpan;
      this.maxCmapEntries = base.maxCmapEntries;
      this.maxSafetensorsHeaderBytes = base.maxSafetensorsHeaderBytes;
      this.maxJsonDepth = base.maxJsonDepth;
      this.maxJsonChars = base.maxJsonChars;
      this.maxGgufStringBytes = base.maxGgufStringBytes;
      this.maxGgufDims = base.maxGgufDims;
      this.maxHistoryMessages = base.maxHistoryMessages;
    }

    public Builder maxFileBytes(final long maxFileBytes) {
      this.maxFileBytes = maxFileBytes;
      return this;
    }

    public Builder maxTotalCorpusBytes(final long maxTotalCorpusBytes) {
      this.maxTotalCorpusBytes = maxTotalCorpusBytes;
      return this;
    }

    public Builder maxCorpusFiles(final int maxCorpusFiles) {
      this.maxCorpusFiles = maxCorpusFiles;
      return this;
    }

    public Builder maxPdfInflateBytes(final long maxPdfInflateBytes) {
      this.maxPdfInflateBytes = maxPdfInflateBytes;
      return this;
    }

    public Builder maxCmapRangeSpan(final int maxCmapRangeSpan) {
      this.maxCmapRangeSpan = maxCmapRangeSpan;
      return this;
    }

    public Builder maxCmapEntries(final int maxCmapEntries) {
      this.maxCmapEntries = maxCmapEntries;
      return this;
    }

    public Builder maxSafetensorsHeaderBytes(final long maxSafetensorsHeaderBytes) {
      this.maxSafetensorsHeaderBytes = maxSafetensorsHeaderBytes;
      return this;
    }

    public Builder maxJsonDepth(final int maxJsonDepth) {
      this.maxJsonDepth = maxJsonDepth;
      return this;
    }

    public Builder maxJsonChars(final long maxJsonChars) {
      this.maxJsonChars = maxJsonChars;
      return this;
    }

    public Builder maxGgufStringBytes(final long maxGgufStringBytes) {
      this.maxGgufStringBytes = maxGgufStringBytes;
      return this;
    }

    public Builder maxGgufDims(final int maxGgufDims) {
      this.maxGgufDims = maxGgufDims;
      return this;
    }

    public Builder maxHistoryMessages(final int maxHistoryMessages) {
      this.maxHistoryMessages = maxHistoryMessages;
      return this;
    }

    public ResourceLimits build() {
      return new ResourceLimits(
        this.maxFileBytes,
        this.maxTotalCorpusBytes,
        this.maxCorpusFiles,
        this.maxPdfInflateBytes,
        this.maxCmapRangeSpan,
        this.maxCmapEntries,
        this.maxSafetensorsHeaderBytes,
        this.maxJsonDepth,
        this.maxJsonChars,
        this.maxGgufStringBytes,
        this.maxGgufDims,
        this.maxHistoryMessages);
    }
  }
}
