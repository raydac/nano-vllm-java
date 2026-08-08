package com.igormaznitsa.nanollvm.engine;

import com.igormaznitsa.nanollvm.llm.SamplingParams;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

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

    SamplingParams sp = samplingParams == null ? new SamplingParams() : samplingParams;
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

  public int seqId() {
    return this.seqId;
  }

  public int blockSize() {
    return this.blockSize;
  }

  public Status status() {
    return this.status;
  }

  public void setStatus(final Status status) {
    this.status = status;
  }

  public List<Integer> tokenIds() {
    return this.tokenIds;
  }

  public int tokenAt(final int index) {
    return this.tokenIds.get(index);
  }

  public int lastToken() {
    return this.lastToken;
  }

  public int numTokens() {
    return this.numTokens;
  }

  public int numPromptTokens() {
    return this.numPromptTokens;
  }

  public int numCachedTokens() {
    return this.numCachedTokens;
  }

  public void setNumCachedTokens(final int n) {
    this.numCachedTokens = n;
  }

  public void addCachedTokens(final int n) {
    this.numCachedTokens += n;
  }

  public int numScheduledTokens() {
    return this.numScheduledTokens;
  }

  public void setNumScheduledTokens(final int n) {
    this.numScheduledTokens = n;
  }

  public boolean isPrefill() {
    return this.prefill;
  }

  public void setPrefill(final boolean prefill) {
    this.prefill = prefill;
  }

  public List<Integer> blockTable() {
    return this.blockTable;
  }

  public float temperature() {
    return this.temperature;
  }

  public int maxTokens() {
    return this.maxTokens;
  }

  public boolean ignoreEos() {
    return this.ignoreEos;
  }

  public int topK() {
    return this.topK;
  }

  public float topP() {
    return this.topP;
  }

  public boolean isFinished() {
    return this.status == Status.FINISHED;
  }

  public int numCompletionTokens() {
    return this.numTokens - this.numPromptTokens;
  }

  public List<Integer> completionTokenIds() {
    return List.copyOf(this.tokenIds.subList(this.numPromptTokens, this.numTokens));
  }

  public int numBlocks() {
    return (this.numTokens + this.blockSize - 1) / this.blockSize;
  }

  public int lastBlockNumTokens() {
    return this.numTokens - (this.numBlocks() - 1) * this.blockSize;
  }

  public List<Integer> block(final int i) {
    if (i < 0 || i >= this.numBlocks()) {
      throw new IndexOutOfBoundsException("block " + i);
    }
    int from = i * this.blockSize;
    int to = Math.min(from + this.blockSize, this.numTokens);
    return List.copyOf(this.tokenIds.subList(from, to));
  }

  public void appendToken(final int tokenId) {
    this.tokenIds.add(tokenId);
    this.lastToken = tokenId;
    this.numTokens++;
  }

  public int length() {
    return this.numTokens;
  }

  public enum Status {
    WAITING, RUNNING, FINISHED
  }
}
