package com.igormaznitsa.nanollvm.engine;

import com.igormaznitsa.nanollvm.llm.SamplingDefaults;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * One generate request: prompt + completion tokens, paged KV block table, and sampling knobs.
 *
 * <p>{@link Scheduler} moves a sequence {@link Status#WAITING} → {@link Status#RUNNING} →
 * {@link Status#FINISHED}. Prefill consumes the prompt (possibly in chunks); decode
 * {@link #appendToken(int) appends} one sampled id per step until stop / max tokens / max model
 * length / {@link #hasDegenerateRepetition()}.
 *
 * <p>{@link #tokenIds()} and {@link #blockTable()} are live lists owned by this object. The
 * scheduler mutates them; callers must not replace or reorder entries. {@link #seqId()} is unique
 * in the process (monotonic {@link AtomicInteger}), used as the conv-arena key.
 *
 * <p><strong>Thread safety:</strong> not concurrent-safe; used on the generate thread only.
 *
 * @see Scheduler
 * @see BlockManager
 */
public final class Sequence {

  private static final AtomicInteger NEXT_SEQ_ID = new AtomicInteger();

  private final int seqId;
  private final int blockSize;
  private final List<Integer> tokenIds;
  private final int numPromptTokens;
  private final List<Integer> blockTable;
  private final float temperature;
  private final int maxTokens;
  private final boolean ignoreEos;
  private final int topK;
  private final float topP;
  private Status status;
  private int lastToken;
  private int numTokens;
  private int numCachedTokens;
  private int numScheduledTokens;
  private boolean prefill;

  /**
   * Copies {@code tokenIds} as the prompt and snapshots sampling knobs. Starts {@link Status#WAITING}
   * with an empty block table and {@code prefill == true}.
   *
   * @param tokenIds       non-empty prompt token ids (copied)
   * @param samplingParams per-sequence knobs; {@code null} → {@link SamplingDefaults#neutral()}
   * @param blockSize      KV page width; must match the scheduler's {@code kvcacheBlockSize}
   * @throws IllegalArgumentException if {@code tokenIds} is null/empty or {@code blockSize < 1}
   */
  public Sequence(
    final List<Integer> tokenIds,
    final SamplingParams samplingParams,
    final int blockSize) {
    if (tokenIds == null || tokenIds.isEmpty()) {
      throw new IllegalArgumentException("tokenIds must not be empty");
    }
    if (blockSize < 1) {
      throw new IllegalArgumentException("kvcacheBlockSize must be >= 1, got " + blockSize);
    }

    SamplingParams sp = samplingParams == null ? SamplingDefaults.neutral() : samplingParams;
    this.seqId = NEXT_SEQ_ID.getAndIncrement();
    this.blockSize = blockSize;
    this.status = Status.WAITING;
    this.tokenIds = new ArrayList<>(tokenIds);
    this.lastToken = tokenIds.getLast();
    this.numTokens = tokenIds.size();
    this.numPromptTokens = tokenIds.size();
    this.numCachedTokens = 0;
    this.numScheduledTokens = 0;
    this.prefill = true;
    this.blockTable = new ArrayList<>();
    this.temperature = sp.temperature();
    this.maxTokens = sp.maxTokens();
    this.ignoreEos = sp.ignoreEos();
    this.topK = sp.topK();
    this.topP = sp.topP();
  }

  /**
   * Process-wide id assigned at construction. Stable for the lifetime of this object; reused as the
   * {@link ConvStateArena} key.
   *
   * @return unique sequence id
   */
  public int seqId() {
    return this.seqId;
  }

  /**
   * KV page width this sequence was built with. Must equal the scheduler's block size.
   *
   * @return page size in tokens
   */
  public int blockSize() {
    return this.blockSize;
  }

  /**
   * Scheduler lifecycle flag.
   *
   * @return current {@link Status}
   */
  public Status status() {
    return this.status;
  }

  /**
   * Updates the lifecycle flag. The scheduler owns transitions; do not set {@link Status#FINISHED}
   * without deallocating KV.
   *
   * @param status new status
   */
  public void setStatus(final Status status) {
    this.status = status;
  }

  /**
   * Live prompt + completion ids. Mutated only through {@link #appendToken(int)}.
   *
   * @return this sequence's token list (not a copy)
   */
  public List<Integer> tokenIds() {
    return this.tokenIds;
  }

  /**
   * Token at absolute index {@code [0, numTokens)}.
   *
   * @param index position in {@link #tokenIds()}
   * @return token id
   */
  public int tokenAt(final int index) {
    return this.tokenIds.get(index);
  }

  /**
   * Last id in {@link #tokenIds()} — decode input for the next step.
   *
   * @return current tail token
   */
  public int lastToken() {
    return this.lastToken;
  }

  /**
   * Prompt length plus completion tokens appended so far.
   *
   * @return {@link #tokenIds()} size
   */
  public int numTokens() {
    return this.numTokens;
  }

  /**
   * Frozen prompt length from construction. Completion is {@link #numTokens()} minus this.
   *
   * @return prompt token count
   */
  public int numPromptTokens() {
    return this.numPromptTokens;
  }

  /**
   * Tokens whose K/V is already accounted in the arena (cached prefix + previous steps).
   *
   * @return cached token count
   */
  public int numCachedTokens() {
    return this.numCachedTokens;
  }

  /**
   * Overwrites the cached-token cursor (used when allocating a prefix-cache hit).
   *
   * @param n new cached count
   */
  public void setNumCachedTokens(final int n) {
    this.numCachedTokens = n;
  }

  /**
   * Advances the cached-token cursor after a step hashes its scheduled window.
   *
   * @param n tokens just written / attended
   */
  public void addCachedTokens(final int n) {
    this.numCachedTokens += n;
  }

  /**
   * Tokens packed into the current engine batch, not yet hashed. Prefill may be a chunk of the
   * remaining prompt; decode is always {@code 1}.
   *
   * @return scheduled token count for this tick
   */
  public int numScheduledTokens() {
    return this.numScheduledTokens;
  }

  /**
   * Sets the scheduled window size for the batch about to run.
   *
   * @param n token count in this tick
   */
  public void setNumScheduledTokens(final int n) {
    this.numScheduledTokens = n;
  }

  /**
   * {@code true} until the scheduler has committed a decode slot ({@code setPrefill(false)}).
   *
   * @return whether this sequence is still in prefill
   */
  public boolean isPrefill() {
    return this.prefill;
  }

  /**
   * Marks prefill vs decode. Preempted sequences are set back to {@code true}.
   *
   * @param prefill {@code true} for prompt ingestion
   */
  public void setPrefill(final boolean prefill) {
    this.prefill = prefill;
  }

  /**
   * Live physical page ids into {@link KvCacheArena} ({@code blockId * blockSize + offset} for a
   * token slot). Empty until {@link BlockManager#allocate} runs.
   *
   * @return this sequence's block table (not a copy)
   */
  public List<Integer> blockTable() {
    return this.blockTable;
  }

  /**
   * Softmax temperature snapped from {@link SamplingParams} at construction.
   *
   * @return temperature {@code > 1e-10}
   */
  public float temperature() {
    return this.temperature;
  }

  /**
   * Cap on <em>new</em> completion tokens (not including the prompt).
   *
   * @return {@code maxTokens} from sampling knobs
   */
  public int maxTokens() {
    return this.maxTokens;
  }

  /**
   * When {@code true}, stop / EOS ids do not finish this sequence.
   *
   * @return whether EOS is ignored
   */
  public boolean ignoreEos() {
    return this.ignoreEos;
  }

  /**
   * Top-k snapped at construction. {@code 0} disables top-k.
   *
   * @return top-k, or {@code 0}
   */
  public int topK() {
    return this.topK;
  }

  /**
   * Nucleus {@code p} snapped at construction. {@code 1} disables top-p.
   *
   * @return top-p in {@code (0, 1]}
   */
  public float topP() {
    return this.topP;
  }

  /**
   * {@code true} after the scheduler has marked this sequence {@link Status#FINISHED}.
   *
   * @return whether generation is done
   */
  public boolean isFinished() {
    return this.status == Status.FINISHED;
  }

  /**
   * Tokens appended after the prompt.
   *
   * @return {@code numTokens - numPromptTokens}
   */
  public int numCompletionTokens() {
    return this.numTokens - this.numPromptTokens;
  }

  /**
   * Copy of completion ids ({@code tokenIds[numPromptTokens .. numTokens)}).
   *
   * @return immutable completion list
   */
  public List<Integer> completionTokenIds() {
    return List.copyOf(this.tokenIds.subList(this.numPromptTokens, this.numTokens));
  }

  /**
   * KV pages needed for the current {@link #numTokens()} ({@code ceil(numTokens / blockSize)}).
   *
   * @return block count
   */
  public int numBlocks() {
    return (this.numTokens + this.blockSize - 1) / this.blockSize;
  }

  /**
   * Occupied slots in the last page ({@code 1 .. blockSize}).
   *
   * @return tokens in the last allocated block
   */
  public int lastBlockNumTokens() {
    return this.numTokens - (this.numBlocks() - 1) * this.blockSize;
  }

  /**
   * Copy of token ids that belong to KV page {@code i}.
   *
   * @param i block index in {@code [0, numBlocks)}
   * @return immutable page contents
   * @throws IndexOutOfBoundsException if {@code i} is out of range
   */
  public List<Integer> block(final int i) {
    if (i < 0 || i >= this.numBlocks()) {
      throw new IndexOutOfBoundsException("block " + i);
    }
    int from = i * this.blockSize;
    int to = Math.min(from + this.blockSize, this.numTokens);
    return List.copyOf(this.tokenIds.subList(from, to));
  }

  /**
   * Appends a sampled completion token and updates {@link #lastToken()} / {@link #numTokens()}.
   *
   * @param tokenId newly sampled id
   */
  public void appendToken(final int tokenId) {
    this.tokenIds.add(tokenId);
    this.lastToken = tokenId;
    this.numTokens++;
  }

  /**
   * Same as {@link #numTokens()} (vLLM-style alias used by {@link BlockManager#canAppend}).
   *
   * @return current length
   */
  public int length() {
    return this.numTokens;
  }

  /**
   * Detects stuck decode loops: same-token streaks, exact repeated blocks, or an n-gram
   * reused enough times that the model is paraphrasing the same sentence forever.
   *
   * @return {@code true} if the scheduler should finish this sequence
   * @since 1.1.0
   */
  public boolean hasDegenerateRepetition() {
    int completion = this.numCompletionTokens();
    if (completion < 32) {
      return false;
    }
    if (this.sameTokenStreak() >= 32) {
      return true;
    }
    int maxPeriod = Math.min(64, completion / 2);
    for (int period = 16; period <= maxPeriod; period += 8) {
      if (this.endsWithTwoIdenticalBlocks(period)) {
        return true;
      }
    }
    return completion >= 48 && this.hasOverusedNgram();
  }

  /**
   * Count of {@link #lastToken()} repeats at the tail of the completion (not the prompt).
   */
  private int sameTokenStreak() {
    int streak = 0;
    for (int i = this.numTokens - 1; i >= this.numPromptTokens; i--) {
      if (this.tokenIds.get(i) != this.lastToken) {
        break;
      }
      streak++;
    }
    return streak;
  }

  /**
   * {@code true} when the last {@code 2 * period} completion tokens are two identical halves.
   */
  private boolean endsWithTwoIdenticalBlocks(final int period) {
    int start = this.numTokens - 2 * period;
    if (start < this.numPromptTokens) {
      return false;
    }
    for (int i = 0; i < period; i++) {
      if (!this.tokenIds.get(start + i).equals(this.tokenIds.get(start + period + i))) {
        return false;
      }
    }
    return true;
  }

  /**
   * {@code true} when some 12-gram in the last 128 completion tokens appears at least 3 times.
   */
  private boolean hasOverusedNgram() {
    final int ngram = 12;
    final int minHits = 3;
    int window = Math.min(this.numCompletionTokens(), 128);
    int from = this.numTokens - window;
    if (window < ngram * minHits) {
      return false;
    }
    Map<Long, Integer> counts = new HashMap<>();
    for (int start = from; start <= this.numTokens - ngram; start++) {
      long hash = 0x9E3779B97F4A7C15L;
      for (int i = 0; i < ngram; i++) {
        hash = 31L * hash + this.tokenIds.get(start + i);
      }
      int hits = counts.merge(hash, 1, Integer::sum);
      if (hits >= minHits) {
        return true;
      }
    }
    return false;
  }

  /**
   * Scheduler lifecycle. {@link #WAITING} is the waiting queue (new or preempted).
   * {@link #RUNNING} is decoding. {@link #FINISHED} has released KV pages.
   */
  public enum Status {
    /**
     * Prompt not fully cached, or preempted back to waiting.
     */
    WAITING,
    /**
     * In the running queue; decode steps append completion tokens.
     */
    RUNNING,
    /**
     * Stopped; block table emptied by {@link BlockManager#deallocate}.
     */
    FINISHED
  }
}
