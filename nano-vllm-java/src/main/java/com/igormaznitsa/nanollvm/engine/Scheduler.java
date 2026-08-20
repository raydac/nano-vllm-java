package com.igormaznitsa.nanollvm.engine;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.Config;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * Continuous batching scheduler for one {@link com.igormaznitsa.nanollvm.llm.LLM} engine.
 *
 * <h2>Role in the engine</h2>
 * Each generate step asks the scheduler for the next work batch, runs the model on it, then feeds
 * sampled tokens back through {@link #postprocess}. The scheduler owns:
 * <ul>
 *   <li>two queues — {@code waiting} (new / preempted prompts) and {@code running} (decoding)</li>
 *   <li>a {@link BlockManager} that pages KV slots into the per-LLM {@link KvCacheArena}</li>
 *   <li>stop / max-token / max-model-len / degenerate-loop finish rules from {@link Config}</li>
 * </ul>
 * Prefill is preferred over decode: as long as waiting sequences can be admitted, {@link #schedule()}
 * returns a prefill batch. Only when waiting cannot proceed does it schedule one decode token per
 * running sequence.
 *
 * <h2>Prefill vs decode</h2>
 * <dl>
 *   <dt>Prefill</dt>
 *   <dd>Consume prompt tokens (possibly chunked by {@code maxNumBatchedTokens}). Allocate KV blocks
 *   (with prefix-cache reuse when {@link BlockManager#canAllocate} reports cached blocks). A sequence
 *   moves from waiting → running only when all of its prompt tokens are cached.</dd>
 *   <dt>Decode</dt>
 *   <dd>Each scheduled sequence contributes exactly one new token slot. If the block pool cannot
 *   {@linkplain BlockManager#canAppend append}, the scheduler {@linkplain #preempt(Sequence) preempts}
 *   other running sequences (LIFO from the back of {@code running}) to free pages, or preempts the
 *   current sequence itself when it is alone and still cannot append.</dd>
 * </dl>
 *
 * <h2>Batch limits</h2>
 * Admission respects {@code maxNumSeqs} (sequences in one batch) and {@code maxNumBatchedTokens}
 * (token count in a prefill batch). Decode batches are sized by how many running sequences fit under
 * {@code maxNumSeqs} after append capacity is ensured.
 *
 * <h2>Lifecycle</h2>
 * Typical loop (driven by {@code LLM}): {@link #add} → while not {@link #isFinished}:
 * {@link #schedule} → model forward → {@link #postprocess}. {@link #clear()} finishes and frees every
 * sequence (used on cancel / timeout).
 *
 * <p><strong>Thread safety:</strong> not concurrent-safe; one scheduler per {@code LLM}, used on the
 * generate thread only.
 *
 * @see Sequence
 * @see BlockManager
 * @see Transformer
 * @see ScheduleResult
 */
public final class Scheduler {

  private final int maxNumSeqs;
  private final int maxNumBatchedTokens;
  private final int maxModelLen;
  private final List<Integer> stopTokenIds;
  private final int blockSize;
  private final BlockManager blockManager;
  private final IntConsumer onSequenceReleased;
  private final Deque<Sequence> waiting = new ArrayDeque<>();
  private final Deque<Sequence> running = new ArrayDeque<>();

  /**
   * Builds a scheduler from engine limits and stop tokens in {@code config}.
   * Instantiates a fresh {@link BlockManager} sized to {@code config.numKvcacheBlocks()}.
   *
   * @param config engine layout (batch limits, KV pages, stop ids)
   * @throws NullPointerException if {@code config} is {@code null}
   */
  public Scheduler(final Config config) {
    this(config, seqId -> {
    });
  }

  /**
   * Same as {@link #Scheduler(Config)}, plus {@code onSequenceReleased} when a sequence's KV is
   * deallocated (finish, cancel/{@link #clear()}, or preempt) so callers can drop short-conv state.
   *
   * @param config              engine layout
   * @param onSequenceReleased  {@link Sequence#seqId()} consumer; {@link Transformer#clearConvState(int)}
   * @throws NullPointerException if {@code config} or {@code onSequenceReleased} is {@code null}
   */
  public Scheduler(final Config config, final IntConsumer onSequenceReleased) {
    requireNonNull(config, "config");
    this.maxNumSeqs = config.maxNumSeqs();
    this.maxNumBatchedTokens = config.maxNumBatchedTokens();
    this.maxModelLen = config.maxModelLen();
    this.stopTokenIds = config.stopTokenIds().isEmpty()
      ? List.of(config.eos())
      : List.copyOf(config.stopTokenIds());
    this.blockSize = config.kvcacheBlockSize();
    this.blockManager = new BlockManager(config.numKvcacheBlocks(), config.kvcacheBlockSize());
    this.onSequenceReleased = requireNonNull(onSequenceReleased, "onSequenceReleased");
  }

  /**
   * {@code true} when both waiting and running queues are empty (no work left for this generate).
   *
   * @return whether {@link #schedule()} has nothing left to do
   */
  public boolean isFinished() {
    return this.waiting.isEmpty() && this.running.isEmpty();
  }

  /**
   * Aborts every in-flight sequence: deallocates KV pages, marks {@link Sequence.Status#FINISHED},
   * and clears both queues. Used when generation is cancelled or times out.
   */
  public void clear() {
    this.finishAndDeallocateAll(this.running);
    this.running.clear();
    this.finishAndDeallocateWaiting();
    this.waiting.clear();
  }

  /**
   * Deallocates KV and marks every sequence {@link Sequence.Status#FINISHED} (running queue).
   */
  private void finishAndDeallocateAll(final Iterable<Sequence> sequences) {
    for (Sequence seq : sequences) {
      this.releaseSequence(seq);
      seq.setStatus(Sequence.Status.FINISHED);
    }
  }

  /**
   * Finishes waiting sequences, deallocating only those that already hold a block table.
   */
  private void finishAndDeallocateWaiting() {
    for (Sequence seq : this.waiting) {
      if (!seq.blockTable().isEmpty()) {
        this.blockManager.deallocate(seq);
      }
      this.onSequenceReleased.accept(seq.seqId());
      seq.setStatus(Sequence.Status.FINISHED);
    }
  }

  /**
   * Frees KV pages and notifies {@code onSequenceReleased} (conv-arena cleanup).
   */
  private void releaseSequence(final Sequence seq) {
    this.blockManager.deallocate(seq);
    this.onSequenceReleased.accept(seq.seqId());
  }

  /**
   * Enqueues {@code seq} at the end of the waiting queue (status should already be WAITING).
   *
   * @param seq new or not-yet-scheduled sequence
   * @throws IllegalArgumentException if {@code seq.blockSize()} does not match this scheduler
   */
  public void add(final Sequence seq) {
    if (seq.blockSize() != this.blockSize) {
      throw new IllegalArgumentException(
        "sequence blockSize " + seq.blockSize()
          + " does not match scheduler blockSize " + this.blockSize);
    }
    this.waiting.addLast(seq);
  }

  /**
   * Picks the next engine batch: a non-empty prefill batch if any waiting sequence can be admitted,
   * otherwise a decode batch from {@code running}.
   *
   * @return sequences to run plus whether the batch is prefill ({@code true}) or decode ({@code false})
   * @throws IllegalStateException if decode is required but {@code running} is empty / cannot schedule
   */
  public ScheduleResult schedule() {
    List<Sequence> prefillBatch = this.trySchedulePrefill();
    if (!prefillBatch.isEmpty()) {
      return new ScheduleResult(prefillBatch, true);
    }
    return new ScheduleResult(this.scheduleDecodeBatch(), false);
  }

  /**
   * Admits waiting sequences into a prefill batch under {@code maxNumSeqs} /
   * {@code maxNumBatchedTokens}. Stops when the next prompt cannot fit or {@link #planPrefillTokens}
   * cannot allocate. Empty list means fall through to decode.
   */
  private List<Sequence> trySchedulePrefill() {
    List<Sequence> scheduled = new ArrayList<>();
    int batchedTokens = 0;

    while (!this.waiting.isEmpty() && scheduled.size() < this.maxNumSeqs) {
      Sequence seq = this.waiting.peekFirst();
      int remainingCapacity = this.maxNumBatchedTokens - batchedTokens;
      if (remainingCapacity == 0) {
        break;
      }

      PrefillPlan plan = this.planPrefillTokens(seq);
      if (plan == null) {
        break;
      }
      if (remainingCapacity < plan.tokenCount() && !scheduled.isEmpty()) {
        break;
      }

      this.commitPrefillSlot(seq, plan, remainingCapacity);
      batchedTokens += seq.numScheduledTokens();
      this.promoteToRunningIfPrefillComplete(seq);
      scheduled.add(seq);
    }

    return scheduled;
  }

  /**
   * Remaining uncached tokens and whether {@link BlockManager#allocate} is still needed.
   * {@code null} when the free list cannot admit this prompt.
   */
  private PrefillPlan planPrefillTokens(final Sequence seq) {
    if (seq.blockTable().isEmpty()) {
      int cachedBlocks = this.blockManager.canAllocate(seq);
      if (cachedBlocks == -1) {
        return null;
      }
      return new PrefillPlan(seq.numTokens() - cachedBlocks * this.blockSize, cachedBlocks, true);
    }
    return new PrefillPlan(seq.numTokens() - seq.numCachedTokens(), 0, false);
  }

  /**
   * Allocates pages if needed and sets {@link Sequence#numScheduledTokens()} to the chunk that
   * fits {@code remainingCapacity}.
   */
  private void commitPrefillSlot(final Sequence seq, final PrefillPlan plan,
                                 final int remainingCapacity) {
    if (plan.needsAllocate()) {
      this.blockManager.allocate(seq, plan.cachedBlocks());
    }
    seq.setNumScheduledTokens(Math.min(plan.tokenCount(), remainingCapacity));
  }

  /**
   * Moves {@code seq} waiting → running when this chunk covers the rest of the prompt.
   */
  private void promoteToRunningIfPrefillComplete(final Sequence seq) {
    if (seq.numCachedTokens() + seq.numScheduledTokens() != seq.numTokens()) {
      return;
    }
    seq.setStatus(Sequence.Status.RUNNING);
    this.waiting.removeFirst();
    this.running.addLast(seq);
  }

  /**
   * One decode token per admitted running sequence, after {@link #ensureAppendCapacity}. Restores
   * running-queue order so the same sequences stay at the front for the next tick.
   */
  private List<Sequence> scheduleDecodeBatch() {
    List<Sequence> scheduled = new ArrayList<>();

    while (!this.running.isEmpty() && scheduled.size() < this.maxNumSeqs) {
      Sequence seq = this.running.removeFirst();
      if (!this.ensureAppendCapacity(seq)) {
        break;
      }
      this.commitDecodeSlot(seq);
      scheduled.add(seq);
    }

    if (scheduled.isEmpty()) {
      throw new IllegalStateException("scheduler produced empty decode batch");
    }

    this.restoreRunningOrder(scheduled);
    return scheduled;
  }

  /**
   * Preempts other running sequences (LIFO) until {@link BlockManager#canAppend} succeeds, or
   * preempts {@code seq} itself when it is alone. {@code false} means {@code seq} left the batch.
   */
  private boolean ensureAppendCapacity(final Sequence seq) {
    while (!this.blockManager.canAppend(seq)) {
      if (this.running.isEmpty()) {
        this.preempt(seq);
        return false;
      }
      this.preempt(this.running.removeLast());
    }
    return true;
  }

  /**
   * Marks one decode token and grows the block table when the last token starts a new page.
   */
  private void commitDecodeSlot(final Sequence seq) {
    seq.setNumScheduledTokens(1);
    seq.setPrefill(false);
    this.blockManager.mayAppend(seq);
  }

  /**
   * Puts scheduled sequences back at the front of {@code running} in original order (they were
   * {@code removeFirst}'d while building the batch).
   */
  private void restoreRunningOrder(final List<Sequence> scheduled) {
    for (int i = scheduled.size() - 1; i >= 0; i--) {
      this.running.addFirst(scheduled.get(i));
    }
  }

  /**
   * Evicts {@code seq} from decode: frees its KV pages, marks it WAITING / prefill-again, and
   * requeues it at the <em>front</em> of waiting so it is retried before newer arrivals.
   *
   * @param seq running sequence that cannot append, or a victim chosen to free pages
   */
  public void preempt(final Sequence seq) {
    seq.setStatus(Sequence.Status.WAITING);
    seq.setPrefill(true);
    this.releaseSequence(seq);
    this.waiting.addFirst(seq);
  }

  /**
   * Applies one forward’s sampled token ids to the scheduled batch (no appended-token side channel).
   *
   * @param seqs      same order as {@link #schedule()}
   * @param tokenIds  one sampled id per sequence
   * @param isPrefill {@code true} if this batch was a prefill step
   * @see #postprocess(List, List, boolean, List)
   */
  public void postprocess(final List<Sequence> seqs, final List<Integer> tokenIds,
                          final boolean isPrefill) {
    this.postprocess(seqs, tokenIds, isPrefill, null);
  }

  /**
   * Advances each scheduled sequence after a model step.
   *
   * <p>Always hashes / accounts scheduled KV tokens via {@link BlockManager#hashBlocks}. Mid-prefill
   * chunks that have not yet covered the full prompt return early (no completion token appended).
   * Otherwise appends {@code tokenIds.get(i)}, optionally records {@code [seqId, tokenId]} into
   * {@code appendedOut}, and finishes the sequence on stop-id, {@code maxTokens},
   * {@code maxModelLen}, or a detected degenerate token-loop.
   *
   * @param seqs        same order as the batch returned by {@link #schedule()}
   * @param tokenIds    one sampled id per sequence
   * @param isPrefill   {@code true} if this batch was a prefill step
   * @param appendedOut nullable collector of newly appended completion tokens for streaming UIs
   */
  public void postprocess(
    final List<Sequence> seqs,
    final List<Integer> tokenIds,
    final boolean isPrefill,
    final List<int[]> appendedOut
  ) {
    IntStream.range(0, seqs.size())
      .forEach(i -> this.postprocessOne(seqs.get(i), tokenIds.get(i), isPrefill, appendedOut));
  }

  /**
   * Hashes the scheduled KV window, then either returns (mid-prefill) or appends {@code tokenId}
   * and maybe finishes the sequence.
   */
  private void postprocessOne(
    final Sequence seq,
    final int tokenId,
    final boolean isPrefill,
    final List<int[]> appendedOut
  ) {
    this.blockManager.hashBlocks(seq);
    seq.addCachedTokens(seq.numScheduledTokens());
    seq.setNumScheduledTokens(0);

    if (isPrefill && seq.numCachedTokens() < seq.numTokens()) {
      return;
    }

    seq.appendToken(tokenId);
    if (appendedOut != null) {
      appendedOut.add(new int[] {seq.seqId(), tokenId});
    }

    if (this.shouldFinish(seq, tokenId)) {
      this.finishSequence(seq);
    }
  }

  /**
   * Stop id (unless {@link Sequence#ignoreEos()}), {@link Sequence#maxTokens()},
   * {@code maxModelLen}, or {@link Sequence#hasDegenerateRepetition()}.
   */
  private boolean shouldFinish(final Sequence seq, final int tokenId) {
    return (!seq.ignoreEos() && this.stopTokenIds.contains(tokenId))
      || seq.numCompletionTokens() == seq.maxTokens()
      || seq.length() >= this.maxModelLen
      || seq.hasDegenerateRepetition();
  }

  /**
   * Marks FINISHED, releases KV / conv state, and removes {@code seq} from {@code running}.
   */
  private void finishSequence(final Sequence seq) {
    seq.setStatus(Sequence.Status.FINISHED);
    this.releaseSequence(seq);
    this.running.remove(seq);
  }

  /**
   * Prefill admission: remaining uncached tokens, prefix-cache hit count, and whether
   * {@link BlockManager#allocate} still needs to run.
   */
  private record PrefillPlan(int tokenCount, int cachedBlocks, boolean needsAllocate) {
  }

  /**
   * One scheduling decision: the sequences to run together and whether the step is prefill.
   *
   * @param sequences batch in engine order (paired with sampled ids in {@link #postprocess})
   * @param prefill   {@code true} for prompt ingestion; {@code false} for one-token decode
   */
  public record ScheduleResult(List<Sequence> sequences, boolean prefill) {
  }
}
