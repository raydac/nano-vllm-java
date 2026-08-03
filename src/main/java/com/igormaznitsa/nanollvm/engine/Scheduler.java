package com.igormaznitsa.nanollvm.engine;

import com.igormaznitsa.nanollvm.Config;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.stream.IntStream;

public final class Scheduler {

  private final int maxNumSeqs;
  private final int maxNumBatchedTokens;
  private final List<Integer> stopTokenIds;
  private final int blockSize;
  private final BlockManager blockManager;
  private final Deque<Sequence> waiting = new ArrayDeque<>();
  private final Deque<Sequence> running = new ArrayDeque<>();

  public Scheduler(Config config) {
    this.maxNumSeqs = config.maxNumSeqs();
    this.maxNumBatchedTokens = config.maxNumBatchedTokens();
    this.stopTokenIds = config.stopTokenIds().isEmpty()
        ? List.of(config.eos())
        : List.copyOf(config.stopTokenIds());
    this.blockSize = config.kvcacheBlockSize();
    this.blockManager = new BlockManager(config.numKvcacheBlocks(), config.kvcacheBlockSize());
  }

  public boolean isFinished() {
    return this.waiting.isEmpty() && this.running.isEmpty();
  }

  public void clear() {
    this.finishAndDeallocateAll(this.running);
    this.running.clear();
    this.finishAndDeallocateWaiting();
    this.waiting.clear();
  }

  private void finishAndDeallocateAll(Iterable<Sequence> sequences) {
    for (Sequence seq : sequences) {
      this.blockManager.deallocate(seq);
      seq.setStatus(Sequence.Status.FINISHED);
    }
  }

  private void finishAndDeallocateWaiting() {
    for (Sequence seq : this.waiting) {
      if (!seq.blockTable().isEmpty()) {
        this.blockManager.deallocate(seq);
      }
      seq.setStatus(Sequence.Status.FINISHED);
    }
  }

  public void add(Sequence seq) {
    this.waiting.addLast(seq);
  }

  public ScheduleResult schedule() {
    List<Sequence> prefillBatch = this.trySchedulePrefill();
    if (!prefillBatch.isEmpty()) {
      return new ScheduleResult(prefillBatch, true);
    }
    return new ScheduleResult(this.scheduleDecodeBatch(), false);
  }

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

  private PrefillPlan planPrefillTokens(Sequence seq) {
    if (seq.blockTable().isEmpty()) {
      int cachedBlocks = this.blockManager.canAllocate(seq);
      if (cachedBlocks == -1) {
        return null;
      }
      return new PrefillPlan(seq.numTokens() - cachedBlocks * this.blockSize, cachedBlocks, true);
    }
    return new PrefillPlan(seq.numTokens() - seq.numCachedTokens(), 0, false);
  }

  private void commitPrefillSlot(Sequence seq, PrefillPlan plan, int remainingCapacity) {
    if (plan.needsAllocate()) {
      this.blockManager.allocate(seq, plan.cachedBlocks());
    }
    seq.setNumScheduledTokens(Math.min(plan.tokenCount(), remainingCapacity));
  }

  private void promoteToRunningIfPrefillComplete(Sequence seq) {
    if (seq.numCachedTokens() + seq.numScheduledTokens() != seq.numTokens()) {
      return;
    }
    seq.setStatus(Sequence.Status.RUNNING);
    this.waiting.removeFirst();
    this.running.addLast(seq);
  }

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

  private boolean ensureAppendCapacity(Sequence seq) {
    while (!this.blockManager.canAppend(seq)) {
      if (this.running.isEmpty()) {
        this.preempt(seq);
        return false;
      }
      this.preempt(this.running.removeLast());
    }
    return true;
  }

  private void commitDecodeSlot(Sequence seq) {
    seq.setNumScheduledTokens(1);
    seq.setPrefill(false);
    this.blockManager.mayAppend(seq);
  }

  private void restoreRunningOrder(List<Sequence> scheduled) {
    for (int i = scheduled.size() - 1; i >= 0; i--) {
      this.running.addFirst(scheduled.get(i));
    }
  }

  public void preempt(Sequence seq) {
    seq.setStatus(Sequence.Status.WAITING);
    seq.setPrefill(true);
    this.blockManager.deallocate(seq);
    this.waiting.addFirst(seq);
  }

  public void postprocess(List<Sequence> seqs, List<Integer> tokenIds, boolean isPrefill) {
    this.postprocess(seqs, tokenIds, isPrefill, null);
  }

  public void postprocess(
      List<Sequence> seqs,
      List<Integer> tokenIds,
      boolean isPrefill,
      List<int[]> appendedOut
  ) {
    IntStream.range(0, seqs.size())
        .forEach(i -> this.postprocessOne(seqs.get(i), tokenIds.get(i), isPrefill, appendedOut));
  }

  private void postprocessOne(
      Sequence seq,
      int tokenId,
      boolean isPrefill,
      List<int[]> appendedOut
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

  private boolean shouldFinish(Sequence seq, int tokenId) {
    return (!seq.ignoreEos() && this.stopTokenIds.contains(tokenId))
        || seq.numCompletionTokens() == seq.maxTokens();
  }

  private void finishSequence(Sequence seq) {
    seq.setStatus(Sequence.Status.FINISHED);
    this.blockManager.deallocate(seq);
    this.running.remove(seq);
  }

  private record PrefillPlan(int tokenCount, int cachedBlocks, boolean needsAllocate) {
  }

  public record ScheduleResult(List<Sequence> sequences, boolean prefill) {
  }
}
