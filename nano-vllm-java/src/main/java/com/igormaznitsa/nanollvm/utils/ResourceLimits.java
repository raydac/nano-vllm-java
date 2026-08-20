package com.igormaznitsa.nanollvm.utils;

import static java.util.Objects.requireNonNull;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Process-wide and per-call budgets for parsers and corpus loaders.
 *
 * <p>Defaults protect against accidental or hostile oversized inputs (OOM / hang). Replace the
 * process default with {@link #setCurrent(ResourceLimits)} or pass a custom instance into RAG /
 * load APIs that accept one. {@link #setCurrent(ResourceLimits)} is <em>JVM-global</em> — every
 * thread and tenant in this process sees it. In a multi-tenant server prefer
 * {@link com.igormaznitsa.nanollvm.rag.RagLoadOptions#withResourceLimits(ResourceLimits)} (and
 * {@link com.igormaznitsa.nanollvm.chat.ChatSession#maxHistoryMessages(int)}) over mutating the
 * process default. Every field must be {@code >= 1}. Prefer {@link #builder()} (starts
 * from {@link #current()}), {@link #defaults()}, or the {@code with*} copies.
 *
 * <pre>{@code
 * ResourceLimits.setCurrent(
 *     ResourceLimits.builder()
 *         .maxFileBytes(64L * 1024 * 1024)
 *         .maxHistoryMessages(50)
 *         .build());
 * }</pre>
 *
 * @param maxFileBytes              cap on one document / weight sidecar read
 * @param maxTotalCorpusBytes       cap on summed RAG corpus bytes
 * @param maxCorpusFiles            cap on files accepted into one corpus load
 * @param maxSafetensorsHeaderBytes cap on the JSON header of a {@code .safetensors} file
 * @param maxJsonDepth              cap on JSON object/array nesting
 * @param maxJsonChars              cap on JSON document size (config / tokenizer)
 * @param maxGgufStringBytes        cap on one GGUF metadata string
 * @param maxGgufDims               cap on GGUF tensor rank
 * @param maxHistoryMessages        cap on {@link com.igormaznitsa.nanollvm.chat.ChatSession}
 *                                  history length
 */
public record ResourceLimits(
  long maxFileBytes,
  long maxTotalCorpusBytes,
  int maxCorpusFiles,
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

  /**
   * Built-in library defaults (32 MiB file, 256 MiB corpus, 200 history messages, …).
   */
  public static ResourceLimits defaults() {
    return new ResourceLimits(
      DEFAULT_MAX_FILE_BYTES,
      DEFAULT_MAX_TOTAL_CORPUS_BYTES,
      DEFAULT_MAX_CORPUS_FILES,
      DEFAULT_MAX_SAFETENSORS_HEADER_BYTES,
      DEFAULT_MAX_JSON_DEPTH,
      DEFAULT_MAX_JSON_CHARS,
      DEFAULT_MAX_GGUF_STRING_BYTES,
      DEFAULT_MAX_GGUF_DIMS,
      DEFAULT_MAX_HISTORY_MESSAGES);
  }

  /**
   * Process-wide limits used when a call does not pass its own instance.
   */
  public static ResourceLimits current() {
    return CURRENT.get();
  }

  /**
   * Replaces {@link #current()} for this JVM. Subsequent loads / RAG / sessions pick this up.
   * Visible to every thread in the process — not a request-scoped setting.
   *
   * @param limits must not be {@code null}
   */
  public static void setCurrent(final ResourceLimits limits) {
    CURRENT.set(requireNonNull(limits, "limits"));
  }

  /**
   * Restores {@link #current()} to {@link #defaults()}.
   */
  public static void resetCurrent() {
    CURRENT.set(defaults());
  }

  /**
   * Builder seeded from {@link #current()}.
   */
  public static Builder builder() {
    return new Builder(current());
  }

  /**
   * Copy with a new per-file byte cap ({@code >= 1}).
   */
  public ResourceLimits withMaxFileBytes(final long maxFileBytes) {
    return new ResourceLimits(
      maxFileBytes, this.maxTotalCorpusBytes, this.maxCorpusFiles,
      this.maxSafetensorsHeaderBytes, this.maxJsonDepth, this.maxJsonChars,
      this.maxGgufStringBytes, this.maxGgufDims, this.maxHistoryMessages);
  }

  /**
   * Copy with a new summed-corpus byte cap ({@code >= 1}).
   */
  public ResourceLimits withMaxTotalCorpusBytes(final long maxTotalCorpusBytes) {
    return new ResourceLimits(
      this.maxFileBytes, maxTotalCorpusBytes, this.maxCorpusFiles,
      this.maxSafetensorsHeaderBytes, this.maxJsonDepth, this.maxJsonChars,
      this.maxGgufStringBytes, this.maxGgufDims, this.maxHistoryMessages);
  }

  /**
   * Copy with a new corpus file-count cap ({@code >= 1}).
   */
  public ResourceLimits withMaxCorpusFiles(final int maxCorpusFiles) {
    return new ResourceLimits(
      this.maxFileBytes, this.maxTotalCorpusBytes, maxCorpusFiles,
      this.maxSafetensorsHeaderBytes, this.maxJsonDepth, this.maxJsonChars,
      this.maxGgufStringBytes, this.maxGgufDims, this.maxHistoryMessages);
  }

  /**
   * Copy with a new chat-history message cap ({@code >= 1}).
   */
  public ResourceLimits withMaxHistoryMessages(final int maxHistoryMessages) {
    return new ResourceLimits(
      this.maxFileBytes, this.maxTotalCorpusBytes, this.maxCorpusFiles,
      this.maxSafetensorsHeaderBytes, this.maxJsonDepth, this.maxJsonChars,
      this.maxGgufStringBytes, this.maxGgufDims, maxHistoryMessages);
  }

  /**
   * Mutable builder for {@link ResourceLimits}. Starts from {@link #current()} via
   * {@link ResourceLimits#builder()}. {@link #build()} validates every field ({@code >= 1}).
   */
  public static final class Builder {
    private long maxFileBytes;
    private long maxTotalCorpusBytes;
    private int maxCorpusFiles;
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
      this.maxSafetensorsHeaderBytes = base.maxSafetensorsHeaderBytes;
      this.maxJsonDepth = base.maxJsonDepth;
      this.maxJsonChars = base.maxJsonChars;
      this.maxGgufStringBytes = base.maxGgufStringBytes;
      this.maxGgufDims = base.maxGgufDims;
      this.maxHistoryMessages = base.maxHistoryMessages;
    }

    /**
     * Per-file read cap ({@code >= 1}).
     */
    public Builder maxFileBytes(final long maxFileBytes) {
      this.maxFileBytes = maxFileBytes;
      return this;
    }

    /** Summed RAG corpus cap ({@code >= 1}). */
    public Builder maxTotalCorpusBytes(final long maxTotalCorpusBytes) {
      this.maxTotalCorpusBytes = maxTotalCorpusBytes;
      return this;
    }

    /** Max files in one corpus load ({@code >= 1}). */
    public Builder maxCorpusFiles(final int maxCorpusFiles) {
      this.maxCorpusFiles = maxCorpusFiles;
      return this;
    }

    /** {@code .safetensors} JSON header cap ({@code >= 1}). */
    public Builder maxSafetensorsHeaderBytes(final long maxSafetensorsHeaderBytes) {
      this.maxSafetensorsHeaderBytes = maxSafetensorsHeaderBytes;
      return this;
    }

    /** JSON nesting cap ({@code >= 1}). */
    public Builder maxJsonDepth(final int maxJsonDepth) {
      this.maxJsonDepth = maxJsonDepth;
      return this;
    }

    /** JSON document size cap ({@code >= 1}). */
    public Builder maxJsonChars(final long maxJsonChars) {
      this.maxJsonChars = maxJsonChars;
      return this;
    }

    /** Max GGUF metadata string ({@code >= 1}). */
    public Builder maxGgufStringBytes(final long maxGgufStringBytes) {
      this.maxGgufStringBytes = maxGgufStringBytes;
      return this;
    }

    /** Max GGUF tensor rank ({@code >= 1}). */
    public Builder maxGgufDims(final int maxGgufDims) {
      this.maxGgufDims = maxGgufDims;
      return this;
    }

    /** Chat history length cap ({@code >= 1}). */
    public Builder maxHistoryMessages(final int maxHistoryMessages) {
      this.maxHistoryMessages = maxHistoryMessages;
      return this;
    }

    /**
     * Builds a validated {@link ResourceLimits}.
     *
     * @throws IllegalArgumentException if any field is {@code < 1}
     */
    public ResourceLimits build() {
      return new ResourceLimits(
        this.maxFileBytes,
        this.maxTotalCorpusBytes,
        this.maxCorpusFiles,
        this.maxSafetensorsHeaderBytes,
        this.maxJsonDepth,
        this.maxJsonChars,
        this.maxGgufStringBytes,
        this.maxGgufDims,
        this.maxHistoryMessages);
    }
  }
}
