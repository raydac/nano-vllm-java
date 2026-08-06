package com.igormaznitsa.nanollvm.engine;

import com.igormaznitsa.nanollvm.llm.SamplingParams;

import java.util.ArrayList;
import java.util.List;

public final class Sequence {

  private static int nextSeqId;
  private static int blockSize = 256;
  private final int seqId;
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

  public Sequence(final List<Integer> tokenIds, final SamplingParams samplingParams) {
    if (tokenIds == null || tokenIds.isEmpty()) {
      throw new IllegalArgumentException("tokenIds must not be empty");
    }
    SamplingParams sp = samplingParams == null ? new SamplingParams() : samplingParams;
    this.seqId = nextSeqId++;
    this.status = Status.WAITING;
    this.tokenIds = new ArrayList<>(tokenIds);
    this.lastToken = tokenIds.get(tokenIds.size() - 1);
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

  public static void setBlockSize(final int size) {
    blockSize = size;
  }

  public static int blockSize() {
    return blockSize;
  }

  public int seqId() {
    return this.seqId;
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
    return (this.numTokens + blockSize - 1) / blockSize;
  }

  public int lastBlockNumTokens() {
    return this.numTokens - (this.numBlocks() - 1) * blockSize;
  }

  public List<Integer> block(final int i) {
    if (i < 0 || i >= this.numBlocks()) {
      throw new IndexOutOfBoundsException("block " + i);
    }
    int from = i * blockSize;
    int to = Math.min(from + blockSize, this.numTokens);
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
