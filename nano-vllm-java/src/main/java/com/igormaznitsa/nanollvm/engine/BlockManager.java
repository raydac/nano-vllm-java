package com.igormaznitsa.nanollvm.engine;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.CRC32;

public final class BlockManager {

  private static final long HASH_POLY = 0xC96C5795D7870F42L;

  private final int blockSize;
  private final List<Block> blocks;
  private final Map<Long, Integer> hashToBlockId;
  private final Deque<Integer> freeBlockIds;
  private final Set<Integer> usedBlockIds;

  public BlockManager(final int numBlocks, final int blockSize) {
    if (numBlocks < 1) {
      throw new IllegalArgumentException("numBlocks must be >= 1");
    }
    this.blockSize = blockSize;
    this.blocks = new ArrayList<>(numBlocks);
    this.hashToBlockId = new HashMap<>();
    this.freeBlockIds = new ArrayDeque<>();
    this.usedBlockIds = new HashSet<>();
    for (int i = 0; i < numBlocks; i++) {
      this.blocks.add(new Block(i));
      this.freeBlockIds.add(i);
    }
  }

  public static long computeHash(final List<Integer> tokenIds, final long prefix) {
    long h = 0xCBF29CE484222325L;
    if (prefix != -1L) {
      h ^= prefix;
      h *= 0x100000001B3L;
    }
    ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    for (int token : tokenIds) {
      buf.clear();
      buf.putInt(token);
      buf.flip();
      while (buf.hasRemaining()) {
        h ^= Byte.toUnsignedLong(buf.get());
        h *= 0x100000001B3L;
      }
    }
    CRC32 crc = new CRC32();
    ByteBuffer all = ByteBuffer.allocate(8 + tokenIds.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
    all.putLong(prefix);
    for (int token : tokenIds) {
      all.putInt(token);
    }
    crc.update(all.array());
    return h ^ (crc.getValue() * HASH_POLY);
  }

  private int allocateBlock() {
    Integer blockId = this.freeBlockIds.pollFirst();
    if (blockId == null) {
      throw new IllegalStateException("no free KV blocks");
    }
    Block block = this.blocks.get(blockId);
    if (block.refCount() != 0) {
      throw new IllegalStateException("free block has refCount != 0");
    }
    if (block.hash() != -1L && this.hashToBlockId.getOrDefault(block.hash(), -1).equals(blockId)) {
      this.hashToBlockId.remove(block.hash());
    }
    block.reset();
    this.usedBlockIds.add(blockId);
    return blockId;
  }

  private void deallocateBlock(final int blockId) {
    if (this.blocks.get(blockId).refCount() != 0) {
      throw new IllegalStateException("cannot free block with refs");
    }
    this.usedBlockIds.remove(blockId);
    this.freeBlockIds.addLast(blockId);
  }

  public int canAllocate(final Sequence seq) {
    long h = -1L;
    int numCachedBlocks = 0;
    int numNewBlocks = seq.numBlocks();
    for (int i = 0; i < seq.numBlocks() - 1; i++) {
      List<Integer> tokenIds = seq.block(i);
      h = computeHash(tokenIds, h);
      int blockId = this.hashToBlockId.getOrDefault(h, -1);
      if (blockId == -1 || !this.blocks.get(blockId).tokenIds().equals(tokenIds)) {
        break;
      }
      numCachedBlocks++;
      if (this.usedBlockIds.contains(blockId)) {
        numNewBlocks--;
      }
    }
    if (this.freeBlockIds.size() < numNewBlocks) {
      return -1;
    }
    return numCachedBlocks;
  }

  public void allocate(final Sequence seq, final int numCachedBlocks) {
    if (!seq.blockTable().isEmpty()) {
      throw new IllegalStateException("sequence already has block table");
    }
    long h = -1L;
    for (int i = 0; i < numCachedBlocks; i++) {
      List<Integer> tokenIds = seq.block(i);
      h = computeHash(tokenIds, h);
      int blockId = this.hashToBlockId.get(h);
      Block block = this.blocks.get(blockId);
      if (this.usedBlockIds.contains(blockId)) {
        block.incRef();
      } else {
        block.setRefCount(1);
        this.freeBlockIds.remove(blockId);
        this.usedBlockIds.add(blockId);
      }
      seq.blockTable().add(blockId);
    }
    for (int i = numCachedBlocks; i < seq.numBlocks(); i++) {
      seq.blockTable().add(this.allocateBlock());
    }
    seq.setNumCachedTokens(numCachedBlocks * this.blockSize);
  }

  public void deallocate(final Sequence seq) {
    List<Integer> table = seq.blockTable();
    for (int i = table.size() - 1; i >= 0; i--) {
      int blockId = table.get(i);
      Block block = this.blocks.get(blockId);
      block.decRef();
      if (block.refCount() == 0) {
        this.deallocateBlock(blockId);
      }
    }
    seq.setNumCachedTokens(0);
    table.clear();
  }

  public boolean canAppend(final Sequence seq) {
    return this.freeBlockIds.size() >= (seq.length() % this.blockSize == 1 ? 1 : 0);
  }

  public void mayAppend(final Sequence seq) {
    if (seq.length() % this.blockSize == 1) {
      seq.blockTable().add(this.allocateBlock());
    }
  }

  public void hashBlocks(final Sequence seq) {
    int start = seq.numCachedTokens() / this.blockSize;
    int end = (seq.numCachedTokens() + seq.numScheduledTokens()) / this.blockSize;
    if (start == end) {
      return;
    }
    long h = start > 0 ? this.blocks.get(seq.blockTable().get(start - 1)).hash() : -1L;
    for (int i = start; i < end; i++) {
      Block block = this.blocks.get(seq.blockTable().get(i));
      List<Integer> tokenIds = seq.block(i);
      h = computeHash(tokenIds, h);
      block.update(h, tokenIds);
      this.hashToBlockId.put(h, block.blockId());
    }
  }

  public static final class Block {
    private final int blockId;
    private int refCount;
    private long hash;
    private List<Integer> tokenIds;

    Block(int blockId) {
      this.blockId = blockId;
      this.refCount = 0;
      this.hash = -1L;
      this.tokenIds = List.of();
    }

    void update(final long hash, final List<Integer> tokenIds) {
      this.hash = hash;
      this.tokenIds = List.copyOf(tokenIds);
    }

    void reset() {
      this.refCount = 1;
      this.hash = -1L;
      this.tokenIds = List.of();
    }

    int blockId() {
      return this.blockId;
    }

    int refCount() {
      return this.refCount;
    }

    void setRefCount(final int refCount) {
      this.refCount = refCount;
    }

    void incRef() {
      this.refCount++;
    }

    void decRef() {
      this.refCount--;
    }

    long hash() {
      return this.hash;
    }

    List<Integer> tokenIds() {
      return this.tokenIds;
    }
  }
}
