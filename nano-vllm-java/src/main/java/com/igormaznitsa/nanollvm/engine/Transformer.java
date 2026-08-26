package com.igormaznitsa.nanollvm.engine;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Sampler;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.LlmModelImpl;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Named home for one transformer tick: prepare batch tensors → {@link CausalLM#forward} →
 * logits → sample. Owns the per-{@code LLM} {@link KvCacheArena} (and optional
 * {@link ConvStateArena} for short-conv architectures); the immutable network graph lives on
 * {@link LlmModel}/{@link CausalLM}.
 *
 * <h2>Role in the engine</h2>
 * Driven by {@link com.igormaznitsa.nanollvm.llm.LLM#generate}: the {@link Scheduler} picks a prefill
 * or decode batch, then this class runs the model on that batch and returns one next-token id per
 * scheduled sequence. Attention and short-conv layers read the arenas through an explicit
 * {@link Context} owned by this transformer for the duration of {@link #step}.
 *
 * <h2>Prefill vs decode</h2>
 * <dl>
 *   <dt>Prefill</dt>
 *   <dd>{@link #preparePrefill} packs every newly scheduled prompt token into flat
 *   {@code inputIds}/{@code positions}, builds cumulative sequence lengths and KV slot mapping,
 *   and publishes them on {@link Context} so attention can write into the arena.</dd>
 *   <dt>Decode</dt>
 *   <dd>{@link #prepareDecode} contributes exactly one token per sequence (the last token),
 *   with context lengths and a single slot index each for the append write.</dd>
 * </dl>
 *
 * <h2>Call chain</h2>
 * {@link #step} → {@link #prepareInputs} → {@link #forwardHidden} → {@link #computeLogits} →
 * {@link #sampleTokens}. Sampling knobs (temperature / top-k / top-p) are collected once per batch
 * from the scheduled {@link Sequence}s.
 *
 * <h2>Lifecycle</h2>
 * Constructed once per {@code LLM} with a shared immutable {@link LlmModel}. {@link #close()}
 * clears the step {@link Context} and drops per-LLM arena / network references so KV and this
 * engine's hold on the shared graph can be GC'd; the {@link LlmModel} itself is not closed here.
 *
 * <p><strong>Thread safety:</strong> not concurrent-safe; one transformer per {@code LLM}, used on
 * the generate thread only (same contract as {@link Scheduler}).
 *
 * @see Scheduler
 * @see KvCacheArena
 * @see ConvStateArena
 * @see CausalLM
 */
public final class Transformer implements AutoCloseable {

  private final Config config;
  private final int blockSize;
  private final MatmulRuntime matmul;
  private final Context stepContext = new Context();
  private final Sampler sampler = new Sampler();
  private final Random sampleRandom;
  private final AtomicBoolean closed = new AtomicBoolean();
  private CausalLM network;
  private KvCacheArena kvCache;
  private ConvStateArena convCache;

  /**
   * Builds a transformer with silent engine I/O (no load-progress lines).
   *
   * @param model  immutable loaded graph + weights (shared across LLMs)
   * @param config engine limits used to size the KV arena
   * @param matmul per-LLM dense matmul runtime
   * @throws NullPointerException if {@code model}, {@code config}, or {@code matmul} is {@code null}
   */
  public Transformer(final LlmModel model, final Config config, final MatmulRuntime matmul) {
    this(model, config, matmul, LlmListeners.silent(), false, new Random());
  }

  /**
   * Builds a transformer, allocates the per-LLM {@link KvCacheArena}, and optionally a
   * {@link ConvStateArena} when the HF config advertises a short-conv cache length.
   *
   * @param model  immutable loaded graph + weights (shared across LLMs)
   * @param config engine limits used to size the KV arena
   * @param matmul per-LLM dense matmul runtime (bound on step {@link Context})
   * @param io     progress sink for KV / conv allocation messages; {@code null} → silent
   */
  public Transformer(
    final LlmModel model,
    final Config config,
    final MatmulRuntime matmul,
    final LlmListener io
  ) {
    this(model, config, matmul, io, false, new Random());
  }

  /**
   * Builds a transformer, allocates the per-LLM {@link KvCacheArena}, and optionally a
   * {@link ConvStateArena} when the HF config advertises a short-conv cache length.
   *
   * @param model                 immutable loaded graph + weights (shared across LLMs)
   * @param config                engine limits used to size the KV arena
   * @param matmul                per-LLM dense matmul runtime (bound on step {@link Context})
   * @param io                    progress sink for KV / conv allocation messages; {@code null} → silent
   * @param allowUnpackParameters when {@code true}, GGUF packed weights are expanded to float32
   */
  public Transformer(
    final LlmModel model,
    final Config config,
    final MatmulRuntime matmul,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) {
    this(model, config, matmul, io, allowUnpackParameters, new Random());
  }

  /**
   * Builds a transformer with an engine-owned sampling RNG.
   *
   * @param model                 immutable loaded graph + weights (shared across LLMs)
   * @param config                engine limits used to size the KV arena
   * @param matmul                per-LLM dense matmul runtime (bound on step {@link Context})
   * @param io                    progress sink for KV / conv allocation messages; {@code null} → silent
   * @param allowUnpackParameters when {@code true}, GGUF packed weights are expanded to float32
   * @param sampleRandom          Gumbel draw source for {@link Sampler}; must not be {@code null}
   */
  public Transformer(
    final LlmModel model,
    final Config config,
    final MatmulRuntime matmul,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Random sampleRandom
  ) {
    // Capture engine config and resolve progress sink (null → silent)
    this.config = config;
    this.sampleRandom = requireNonNull(sampleRandom, "sampleRandom");
    final LlmListener io1 = io == null ? LlmListeners.silent() : io;
    this.blockSize = config.kvcacheBlockSize();
    this.network = LlmModelImpl.peer(model).resolveNetwork(allowUnpackParameters, io1);
    this.matmul = requireNonNull(matmul, "matmul");

    // Allocate paged KV arena sized from config (or heap-aware estimate)
    LlmListeners.info(io1, null, "Allocating KV cache…");
    long tKv = System.nanoTime();
    this.kvCache = this.allocateKvCache();
    LlmListeners.infof(io1, null, "KV cache ready: %d blocks (%.1fs)%n",
      this.config.numKvcacheBlocks(),
      (System.nanoTime() - tKv) / 1e9);

    // Short-conv hybrid models need a parallel per-layer conv state arena
    Config.HfConfig hf = this.config.hfConfig();
    if (hf.convLCache() > 0) {
      this.convCache = new ConvStateArena(hf.numHiddenLayers(), hf.hiddenSize(), hf.convLCache());
      LlmListeners.info(io1, null, "Conv state arena ready (" + this.convCache + ")");
    } else {
      this.convCache = null;
    }
  }

  /**
   * Packs a list of integer token / position ids into a rank-1 float {@link Tensor}
   * (model embeddings expect float storage for gather indices).
   */
  private static Tensor toTensor1d(final List<Integer> values) {
    float[] data = new float[values.size()];
    for (int i = 0; i < values.size(); i++) {
      data[i] = values.get(i);
    }
    return Tensor.of(data, values.size());
  }

  /**
   * One forward+sample over a scheduled batch (prefill or decode).
   *
   * <p>Binds the KV (and optional conv) arenas on {@link Context}, prepares input tensors,
   * runs the causal LM, samples one token id per sequence, then clears the Context.
   *
   * @param seqs      sequences chosen by {@link Scheduler#schedule()} for this tick
   * @param isPrefill {@code true} for a prompt-token batch; {@code false} for one-token decode
   * @return sampled next-token id for each sequence, same order as {@code seqs}
   * @throws IllegalStateException if this transformer is {@linkplain #close() closed}
   */
  public List<Integer> step(final List<Sequence> seqs, final boolean isPrefill) {
    this.requireOpen();
    // Bind per-LLM arenas / matmul on the step Context passed through the forward graph
    this.stepContext.bindKvCache(this.kvCache);
    this.stepContext.bindMatmul(this.matmul);
    if (this.convCache != null) {
      this.stepContext.bindConvCache(this.convCache);
    }

    try {
      // 1) Flatten scheduled tokens → tensors + Context metadata (cuSeqlens, slots, …)
      PreparedInputs prepared = this.prepareInputs(seqs, isPrefill);

      // 2) Snapshot sampling knobs once per sequence for this batch
      SamplingControls sampling = this.collectSamplingControls(seqs);

      // 3) CausalLM forward → last-layer hidden states
      Tensor hidden = this.forwardHidden(prepared);

      // 4) Hidden → vocabulary logits
      Tensor logits = this.computeLogits(hidden);

      // 5) Temperature / top-k / top-p → one token id per sequence
      return this.sampleTokens(logits, sampling);
    } finally {
      this.stepContext.clear();
    }
  }

  /**
   * Runs {@link CausalLM#forward} on the prepared input-id and position tensors.
   */
  private Tensor forwardHidden(final PreparedInputs prepared) {
    return this.network.forward(prepared.inputIds(), prepared.positions(), this.stepContext);
  }

  /**
   * Projects last-layer hidden states to vocabulary logits via {@link CausalLM#computeLogits}.
   */
  private Tensor computeLogits(final Tensor hidden) {
    return this.network.computeLogits(hidden, this.stepContext);
  }

  /**
   * Samples one token id per batch row using the sequence-local temperature / top-k / top-p.
   */
  private List<Integer> sampleTokens(final Tensor logits, final SamplingControls sampling) {
    // Sampler.forward returns int[]; box to List for the scheduler postprocess API
    return this.toTokenIdList(this.sampler.forward(
      logits,
      sampling.temperatures(),
      sampling.topKs(),
      sampling.topPs(),
      this.sampleRandom));
  }

  /**
   * Dispatches to {@link #preparePrefill} or {@link #prepareDecode} based on the schedule flag.
   */
  private PreparedInputs prepareInputs(final List<Sequence> seqs, final boolean isPrefill) {
    // Prefill packs many tokens per seq; decode packs exactly one
    return isPrefill ? this.preparePrefill(seqs) : this.prepareDecode(seqs);
  }

  /**
   * Collects per-sequence temperature, top-k, and top-p into parallel arrays for {@link Sampler}.
   */
  private SamplingControls collectSamplingControls(final List<Sequence> seqs) {
    // Lockstep fill: three parallel primitive arrays indexed by sequence position
    float[] temperatures = new float[seqs.size()];
    int[] topKs = new int[seqs.size()];
    float[] topPs = new float[seqs.size()];
    for (int i = 0; i < seqs.size(); i++) {
      Sequence seq = seqs.get(i);
      temperatures[i] = seq.temperature();
      topKs[i] = seq.topK();
      topPs[i] = seq.topP();
    }
    return new SamplingControls(temperatures, topKs, topPs);
  }

  /**
   * Boxes a primitive token-id array into an immutable {@link List} for callers.
   */
  private List<Integer> toTokenIdList(final int[] tokenIds) {
    return Arrays.stream(tokenIds).boxed().toList();
  }

  /**
   * Clears the step {@link Context} and {@link AutoCloseable#close() closes} KV / conv arenas so
   * their heap can be reclaimed. Drops this engine's hold on the shared graph. Does not
   * {@link LlmModel#close()}.
   */
  @Override
  public void close() {
    if (this.closed.compareAndSet(false, true)) {
      this.stepContext.clear();
      if (this.kvCache != null) {
        this.kvCache.close();
        this.kvCache = null;
      }
      if (this.convCache != null) {
        this.convCache.close();
        this.convCache = null;
      }
      this.network = null;
    }
  }

  /**
   * Drops short-conv state for {@code seqId} (no-op when this transformer has no conv arena).
   *
   * @param seqId {@link Sequence#seqId()} whose KV was just released
   */
  public void clearConvState(final int seqId) {
    ConvStateArena arena = this.convCache;
    if (arena != null) {
      arena.clear(seqId);
    }
  }

  private void requireOpen() {
    if (this.closed.get()) {
      throw new IllegalStateException("Transformer is closed");
    }
  }

  /**
   * Allocates the paged {@link KvCacheArena} using {@link Config#numKvcacheBlocks()} already
   * resolved by {@link com.igormaznitsa.nanollvm.llm.LLM} construction.
   */
  private KvCacheArena allocateKvCache() {
    Config.HfConfig hf = this.config.hfConfig();
    if (this.config.numKvcacheBlocks() <= 0) {
      throw new IllegalStateException("numKvcacheBlocks must be > 0");
    }

    int layers = hf.numHiddenLayers();
    int[] headDims = new int[layers];
    boolean[] allocate = new boolean[layers];
    for (int i = 0; i < layers; i++) {
      headDims[i] = hf.layerHeadDim(i);
      allocate[i] = !hf.isKvSharedLayer(i);
    }
    return new KvCacheArena(
      this.config.numKvcacheBlocks(),
      this.blockSize,
      hf.numKeyValueHeads(),
      headDims,
      allocate);
  }

  /**
   * Pads each sequence's block-table to a rectangular {@code int[][]} (missing slots = {@code -1})
   * so attention can index physical KV pages uniformly across the batch.
   */
  private int[][] prepareBlockTables(final List<Sequence> seqs) {
    // Width = longest block table in the batch; shorter tables pad with -1
    int maxLen = seqs.stream().mapToInt(seq -> seq.blockTable().size()).max().orElse(0);
    int[][] tables = new int[seqs.size()][maxLen];
    for (int i = 0; i < seqs.size(); i++) {
      List<Integer> table = seqs.get(i).blockTable();
      for (int j = 0; j < maxLen; j++) {
        tables[i][j] = j < table.size() ? table.get(j) : -1;
      }
    }
    return tables;
  }

  /**
   * Builds flat prefill tensors and publishes varlen / slot metadata on {@link Context}.
   *
   * <p>For each sequence, only the uncached window
   * {@code [numCachedTokens, numCachedTokens + numScheduledTokens)} is packed. Cumulative
   * query/key lengths ({@code cuSeqlensQ}/{@code cuSeqlensK}), max sequence lengths, and the
   * physical slot mapping tell attention where to write new K/V. Block tables are attached only
   * when some tokens are already cached (prefix reuse) and every sequence has a block table.
   */
  private PreparedInputs preparePrefill(final List<Sequence> seqs) {
    // Accumulators for the packed batch (token ids, positions, cu-seqlens, slots, seq ids)
    List<Integer> inputIds = new ArrayList<>();
    List<Integer> positions = new ArrayList<>();
    List<Integer> cuSeqlensQ = new ArrayList<>();
    List<Integer> cuSeqlensK = new ArrayList<>();
    List<Integer> seqIds = new ArrayList<>();
    cuSeqlensQ.add(0);
    cuSeqlensK.add(0);
    int maxSeqlenQ = 0;
    int maxSeqlenK = 0;
    List<Integer> slotMapping = new ArrayList<>();
    boolean anyBlockTable = false;

    for (Sequence seq : seqs) {
      // Uncached window: start at already-cached prefix, end after this prefill chunk
      int start = seq.numCachedTokens();
      int seqlenQ = seq.numScheduledTokens();
      int end = start + seqlenQ;

      // Pack token ids + absolute positions + owning seqId (one entry per new token)
      for (int i = start; i < end; i++) {
        inputIds.add(seq.tokenAt(i));
        positions.add(i);
        seqIds.add(seq.seqId());
      }

      // Cumulative lengths: Q = new tokens only; K = full context through end
      cuSeqlensQ.add(cuSeqlensQ.getLast() + seqlenQ);
      cuSeqlensK.add(cuSeqlensK.getLast() + end);
      maxSeqlenQ = Math.max(maxSeqlenQ, seqlenQ);
      maxSeqlenK = Math.max(maxSeqlenK, end);

      // No block table yet → nothing to map into the KV arena for this sequence
      if (seq.blockTable().isEmpty()) {
        continue;
      }
      anyBlockTable = true;

      // Map each new token to its physical slot inside the sequence's allocated blocks
      int startBlock = start / this.blockSize;
      int endBlock = (end + this.blockSize - 1) / this.blockSize;
      for (int i = startBlock; i < endBlock; i++) {
        int slotStart = seq.blockTable().get(i) * this.blockSize;
        if (i == startBlock) {
          // First block may start mid-page when the cached prefix is not block-aligned
          slotStart += start % this.blockSize;
        }
        int slotEnd;
        if (i != endBlock - 1) {
          // Full interior block
          slotEnd = seq.blockTable().get(i) * this.blockSize + this.blockSize;
        } else {
          // Last block: only through the final token of this prefill window
          slotEnd = seq.blockTable().get(i) * this.blockSize + end - i * this.blockSize;
        }
        for (int s = slotStart; s < slotEnd; s++) {
          slotMapping.add(s);
        }
      }
    }

    // Block tables needed when K context exceeds Q (prefix cache hit) and tables exist
    int[][] blockTables = null;
    if (cuSeqlensK.getLast() > cuSeqlensQ.getLast() &&
      anyBlockTable) {
      blockTables = this.prepareBlockTables(seqs);
    }

    // Publish varlen attention metadata; layers read this via Context during forward
    this.stepContext.set(
      true,
      cuSeqlensQ.stream().mapToInt(Integer::intValue).toArray(),
      cuSeqlensK.stream().mapToInt(Integer::intValue).toArray(),
      maxSeqlenQ,
      maxSeqlenK,
      slotMapping.stream().mapToInt(Integer::intValue).toArray(),
      null,
      blockTables,
      seqIds.stream().mapToInt(Integer::intValue).toArray()
    );
    return new PreparedInputs(toTensor1d(inputIds), toTensor1d(positions));
  }

  /**
   * Builds one-token-per-sequence decode tensors and publishes append metadata on {@link Context}.
   *
   * <p>Each sequence contributes its last token id, position {@code length - 1}, full context
   * length, and the physical slot for the new K/V write (last block × blockSize + tokens in that
   * block − 1). Block tables are always attached so attention can read the growing cache.
   */
  private PreparedInputs prepareDecode(final List<Sequence> seqs) {
    int n = seqs.size();
    float[] inputIds = new float[n];
    float[] positions = new float[n];
    int[] slotMapping = new int[n];
    int[] contextLens = new int[n];
    int[] seqIds = new int[n];

    // One row per sequence: last token → next-token prediction
    for (int i = 0; i < n; i++) {
      Sequence seq = seqs.get(i);
      inputIds[i] = seq.lastToken();
      positions[i] = seq.length() - 1;
      contextLens[i] = seq.length();
      seqIds[i] = seq.seqId();
      // Physical slot for the append write inside the last allocated KV block
      slotMapping[i] = seq.blockTable().getLast() * this.blockSize
        + seq.lastBlockNumTokens() - 1;
    }

    // Decode Context: no cu-seqlens; contextLens + blockTables drive attention reads
    this.stepContext.set(
      false, null, null, 0, 0, slotMapping, contextLens, this.prepareBlockTables(seqs), seqIds);
    return new PreparedInputs(Tensor.of(inputIds, n), Tensor.of(positions, n));
  }

  /**
   * Input-id and position tensors ready for {@link CausalLM#forward}.
   */
  private record PreparedInputs(Tensor inputIds, Tensor positions) {
  }

  /**
   * Per-sequence sampling knobs aligned with the scheduled batch order.
   */
  @SuppressWarnings("ArrayRecordComponent")
  private record SamplingControls(float[] temperatures, int[] topKs, float[] topPs) {
  }
}
