package io.nanovllm.engine;

import io.nanovllm.Config;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

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
    for (Sequence seq : this.running) {
      this.blockManager.deallocate(seq);
      seq.setStatus(Sequence.Status.FINISHED);
    }
    this.running.clear();
    for (Sequence seq : this.waiting) {
      if (!seq.blockTable().isEmpty()) {
        this.blockManager.deallocate(seq);
      }
      seq.setStatus(Sequence.Status.FINISHED);
    }
    this.waiting.clear();
  }

  public void add(Sequence seq) {
    this.waiting.addLast(seq);
  }

  public ScheduleResult schedule() {
    List<Sequence> scheduled = new ArrayList<>();
    int numBatchedTokens = 0;

    while (!this.waiting.isEmpty() && scheduled.size() < this.maxNumSeqs) {
      Sequence seq = this.waiting.peekFirst();
      int remaining = this.maxNumBatchedTokens - numBatchedTokens;
      if (remaining == 0) {
        break;
      }
      int numTokens;
      int numCachedBlocks = 0;
      if (seq.blockTable().isEmpty()) {
        numCachedBlocks = this.blockManager.canAllocate(seq);
        if (numCachedBlocks == -1) {
          break;
        }
        numTokens = seq.numTokens() - numCachedBlocks * this.blockSize;
      } else {
        numTokens = seq.numTokens() - seq.numCachedTokens();
      }
      if (remaining < numTokens && !scheduled.isEmpty()) {
        break;
      }
      if (seq.blockTable().isEmpty()) {
        this.blockManager.allocate(seq, numCachedBlocks);
      }
      seq.setNumScheduledTokens(Math.min(numTokens, remaining));
      numBatchedTokens += seq.numScheduledTokens();
      if (seq.numCachedTokens() + seq.numScheduledTokens() == seq.numTokens()) {
        seq.setStatus(Sequence.Status.RUNNING);
        this.waiting.removeFirst();
        this.running.addLast(seq);
      }
      scheduled.add(seq);
    }

    if (!scheduled.isEmpty()) {
      return new ScheduleResult(scheduled, true);
    }

    while (!this.running.isEmpty() && scheduled.size() < this.maxNumSeqs) {
      Sequence seq = this.running.removeFirst();
      while (!this.blockManager.canAppend(seq)) {
        if (!this.running.isEmpty()) {
          this.preempt(this.running.removeLast());
        } else {
          this.preempt(seq);
          seq = null;
          break;
        }
      }
      if (seq == null) {
        break;
      }
      seq.setNumScheduledTokens(1);
      seq.setPrefill(false);
      this.blockManager.mayAppend(seq);
      scheduled.add(seq);
    }
    if (scheduled.isEmpty()) {
      throw new IllegalStateException("scheduler produced empty decode batch");
    }
    for (int i = scheduled.size() - 1; i >= 0; i--) {
      this.running.addFirst(scheduled.get(i));
    }
    return new ScheduleResult(scheduled, false);
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
    for (int i = 0; i < seqs.size(); i++) {
      Sequence seq = seqs.get(i);
      int tokenId = tokenIds.get(i);
      this.blockManager.hashBlocks(seq);
      seq.addCachedTokens(seq.numScheduledTokens());
      seq.setNumScheduledTokens(0);
      if (isPrefill && seq.numCachedTokens() < seq.numTokens()) {
        continue;
      }
      seq.appendToken(tokenId);
      if (appendedOut != null) {
        appendedOut.add(new int[] {seq.seqId(), tokenId});
      }
      if ((!seq.ignoreEos() && this.stopTokenIds.contains(tokenId))
          || seq.numCompletionTokens() == seq.maxTokens()) {
        seq.setStatus(Sequence.Status.FINISHED);
        this.blockManager.deallocate(seq);
        this.running.remove(seq);
      }
    }
  }

  public record ScheduleResult(List<Sequence> sequences, boolean prefill) {
  }
}
