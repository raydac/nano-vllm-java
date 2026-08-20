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

/**
 * Paged KV allocator with prefix-cache reuse for one {@link Scheduler}.
 *
 * <p>The pool has a fixed number of pages ({@code numBlocks × blockSize} token slots). Each
 * {@link Sequence} holds a {@linkplain Sequence#blockTable() block table} of physical page ids.
 * Full pages are hashed ({@link #computeHash}) so a later prompt that shares a prefix can share
 * those pages (refcount++) instead of allocating new ones.
 *
 * <p>{@link #canAllocate} is the admission probe (how many leading pages hit cache, or {@code -1}
 * if the free list cannot cover the miss). {@link #allocate} attaches cached pages and new pages.
 * {@link #hashBlocks} publishes hashes after a step fills complete pages.
 * {@link #canAppend} / {@link #mayAppend} grow the table by one page when the current last token
 * is the first slot of a new page ({@code length % blockSize == 1}).
 *
 * <p><strong>Thread safety:</strong> not concurrent-safe; one manager per scheduler, generate
 * thread only.
 *
 * @see Sequence
 * @see KvCacheArena
 */
public final class BlockManager {

  private static final long HASH_POLY = 0xC96C5795D7870F42L;

  private final int blockSize;
  private final List<Block> blocks;
  private final Map<Long, Integer> hashToBlockId;
  private final Deque<Integer> freeBlockIds;
  private final Set<Integer> usedBlockIds;

  /**
   * Builds a pool of {@code numBlocks} empty pages.
   *
   * @param numBlocks page count ({@code >= 1})
   * @param blockSize tokens per page (must match {@link Sequence#blockSize()})
   * @throws IllegalArgumentException if {@code numBlocks < 1}
   */
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

  /**
   * Content hash of a full page, chained from the previous page's hash.
   *
   * <p>{@code prefix == -1} means this is the first page of a sequence. The mix is FNV-1a over
   * little-endian token bytes, XOR a CRC32 of {@code (prefix, tokens)}.
   *
   * @param tokenIds page contents
   * @param prefix   previous page hash, or {@code -1} for the first page
   * @return page hash used as the prefix-cache map key
   */
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

  /**
   * Takes one unused page, resets its hash/tokens, and marks it used with refcount 1.
   *
   * @return physical block id
   * @throws IllegalStateException if the free list is empty
   */
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

  /**
   * Returns {@code blockId} to the free list. Caller must have dropped the last ref.
   */
  private void deallocateBlock(final int blockId) {
    if (this.blocks.get(blockId).refCount() != 0) {
      throw new IllegalStateException("cannot free block with refs");
    }
    this.usedBlockIds.remove(blockId);
    this.freeBlockIds.addLast(blockId);
  }

  /**
   * Prefix-cache probe: how many leading <em>full</em> pages of {@code seq} already live in the
   * pool (hash + token equality), and whether the free list can cover the rest.
   *
   * <p>The last (possibly partial) page is never treated as a cache hit. Sharing a cached page
   * that is already in {@code usedBlockIds} does not consume an extra free page (refcount bump).
   *
   * @param seq sequence whose prompt pages are probed
   * @return cached full-page count, or {@code -1} if allocation cannot succeed
   */
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

  /**
   * Fills {@code seq}'s empty block table: {@code numCachedBlocks} shared pages (refcount++), then
   * new pages for the remainder. Sets {@link Sequence#numCachedTokens()} to the cached prefix
   * length ({@code numCachedBlocks * blockSize}).
   *
   * @param seq             sequence with an empty block table
   * @param numCachedBlocks value previously returned by {@link #canAllocate}
   * @throws IllegalStateException if the table is already non-empty or the pool is exhausted
   */
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

  /**
   * Drops this sequence's refs on its pages (freeing a page at refcount 0), clears the block
   * table, and zeros {@link Sequence#numCachedTokens()}.
   *
   * @param seq sequence to release
   */
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

  /**
   * {@code true} if a decode append can proceed without allocating, or the free list has a page
   * when {@code seq.length() % blockSize == 1} (last token is the first slot of a new page).
   *
   * @param seq running sequence
   * @return whether {@link #mayAppend} will succeed
   */
  public boolean canAppend(final Sequence seq) {
    return this.freeBlockIds.size() >= (seq.length() % this.blockSize == 1 ? 1 : 0);
  }

  /**
   * Allocates one new page onto {@code seq}'s table when the last token starts a new page.
   *
   * @param seq running sequence about to write one decode slot
   * @throws IllegalStateException if a page is required and the pool is empty
   */
  public void mayAppend(final Sequence seq) {
    if (seq.length() % this.blockSize == 1) {
      seq.blockTable().add(this.allocateBlock());
    }
  }

  /**
   * Hashes complete pages in {@code [numCachedTokens, numCachedTokens + numScheduledTokens)} so
   * later sequences can prefix-cache them. Partial last pages are left unhashed.
   *
   * @param seq sequence whose just-scheduled window should be published
   */
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

  /**
   * One physical KV page: refcount, content hash ({@code -1} if unhashed), and a snapshot of the
   * tokens that produced that hash. Mutation is package-private to {@link BlockManager}.
   */
  public static final class Block {
    private final int blockId;
    private int refCount;
    private long hash;
    private List<Integer> tokenIds;

    Block(final int blockId) {
      this.blockId = blockId;
      this.refCount = 0;
      this.hash = -1L;
      this.tokenIds = List.of();
    }

    /**
     * Records a full-page hash and an immutable token snapshot for prefix matching.
     */
    void update(final long hash, final List<Integer> tokenIds) {
      this.hash = hash;
      this.tokenIds = List.copyOf(tokenIds);
    }

    /**
     * Fresh allocation: refcount 1, no hash, empty token snapshot.
     */
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
