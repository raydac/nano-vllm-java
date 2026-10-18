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
    this.config = config;
    this.sampleRandom = requireNonNull(sampleRandom, "sampleRandom");
    final LlmListener progress = io == null ? LlmListeners.silent() : io;
    this.blockSize = config.kvcacheBlockSize();
    this.network = LlmModelImpl.peer(model).resolveNetwork(allowUnpackParameters, progress);
    this.matmul = requireNonNull(matmul, "matmul");

    LlmListeners.info(progress, null, "Allocating KV cache…");
    long tKv = System.nanoTime();
    this.kvCache = this.allocateKvCache();
    LlmListeners.infof(progress, null, "KV cache ready: %d blocks (%.1fs)%n",
      this.config.numKvcacheBlocks(),
      (System.nanoTime() - tKv) / 1e9);

    Config.HfConfig hf = this.config.hfConfig();
    if (hf.convLCache() > 0) {
      this.convCache = new ConvStateArena(hf.numHiddenLayers(), hf.hiddenSize(), hf.convLCache());
      LlmListeners.info(progress, null, "Conv state arena ready (" + this.convCache + ")");
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

  private static int[] toInts(final List<Integer> values) {
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  private static List<Sequence> requirePrefill(final List<Sequence> seqs) {
    requireNonNull(seqs, "seqs");
    if (seqs.isEmpty()) {
      throw new IllegalArgumentException("seqs must not be empty");
    }
    return seqs;
  }

  /**
   * Prefills {@code seqs} and returns vocabulary logits at the last scheduled token of each
   * sequence. Does not sample and does not append a completion token.
   *
   * @param seqs sequences whose scheduled window is a prefill chunk; must not be empty
   * @return logits {@code [seqs.size(), vocab]}
   * @throws IllegalStateException    if this transformer is closed
   * @throws IllegalArgumentException if {@code seqs} is empty
   */
  public Tensor prefillLogits(final List<Sequence> seqs) {
    this.requireOpen();
    this.bindStep();
    try {
      return this.computeLogits(this.forwardHidden(this.preparePrefill(requirePrefill(seqs))));
    } finally {
      this.stepContext.clear();
    }
  }

  /**
   * Prefills {@code seqs} so attention writes KV for the scheduled window, and discards the
   * hidden states. Used for a prompt chunk that is not yet the answer position.
   *
   * @param seqs sequences whose scheduled window is a prefill chunk; must not be empty
   * @throws IllegalStateException    if this transformer is closed
   * @throws IllegalArgumentException if {@code seqs} is empty
   */
  public void prefillCache(final List<Sequence> seqs) {
    this.requireOpen();
    this.bindStep();
    try {
      this.forwardHidden(this.preparePrefill(requirePrefill(seqs)));
    } finally {
      this.stepContext.clear();
    }
  }

  private void bindStep() {
    this.stepContext.bindKvCache(this.kvCache);
    this.stepContext.bindMatmul(this.matmul);
    if (this.convCache != null) {
      this.stepContext.bindConvCache(this.convCache);
    }
  }

  private Tensor forwardHidden(final PreparedInputs prepared) {
    return this.network.forward(prepared.inputIds(), prepared.positions(), this.stepContext);
  }

  private Tensor computeLogits(final Tensor hidden) {
    return this.network.computeLogits(hidden, this.stepContext);
  }

  private List<Integer> sampleTokens(final Tensor logits, final SamplingControls sampling) {
    return this.toTokenIdList(this.sampler.forward(
      logits,
      sampling.temperatures(),
      sampling.topKs(),
      sampling.topPs(),
      this.sampleRandom));
  }

  private PreparedInputs prepareInputs(final List<Sequence> seqs, final boolean isPrefill) {
    return isPrefill ? this.preparePrefill(seqs) : this.prepareDecode(seqs);
  }

  private SamplingControls collectSamplingControls(final List<Sequence> seqs) {
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
    this.bindStep();
    try {
      PreparedInputs prepared = this.prepareInputs(seqs, isPrefill);
      SamplingControls sampling = this.collectSamplingControls(seqs);
      return this.sampleTokens(this.computeLogits(this.forwardHidden(prepared)), sampling);
    } finally {
      this.stepContext.clear();
    }
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
      int start = seq.numCachedTokens();
      int seqlenQ = seq.numScheduledTokens();
      int end = start + seqlenQ;

      for (int i = start; i < end; i++) {
        inputIds.add(seq.tokenAt(i));
        positions.add(i);
        seqIds.add(seq.seqId());
      }

      cuSeqlensQ.add(cuSeqlensQ.getLast() + seqlenQ);
      cuSeqlensK.add(cuSeqlensK.getLast() + end);
      maxSeqlenQ = Math.max(maxSeqlenQ, seqlenQ);
      maxSeqlenK = Math.max(maxSeqlenK, end);

      if (seq.blockTable().isEmpty()) {
        continue;
      }
      anyBlockTable = true;
      this.appendPrefillSlots(seq, start, end, slotMapping);
    }

    int[][] blockTables = cuSeqlensK.getLast() > cuSeqlensQ.getLast() && anyBlockTable
      ? this.prepareBlockTables(seqs)
      : null;
    this.stepContext.set(
      true,
      toInts(cuSeqlensQ),
      toInts(cuSeqlensK),
      maxSeqlenQ,
      maxSeqlenK,
      toInts(slotMapping),
      null,
      blockTables,
      toInts(seqIds));
    return new PreparedInputs(toTensor1d(inputIds), toTensor1d(positions));
  }

  private void appendPrefillSlots(
    final Sequence seq,
    final int start,
    final int end,
    final List<Integer> slotMapping
  ) {
    int startBlock = start / this.blockSize;
    int endBlock = (end + this.blockSize - 1) / this.blockSize;
    for (int block = startBlock; block < endBlock; block++) {
      int pageBase = seq.blockTable().get(block) * this.blockSize;
      int slotStart = block == startBlock ? pageBase + start % this.blockSize : pageBase;
      int slotEnd = block == endBlock - 1
        ? pageBase + end - block * this.blockSize
        : pageBase + this.blockSize;
      for (int slot = slotStart; slot < slotEnd; slot++) {
        slotMapping.add(slot);
      }
    }
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

    for (int i = 0; i < n; i++) {
      Sequence seq = seqs.get(i);
      inputIds[i] = seq.lastToken();
      positions[i] = seq.length() - 1;
      contextLens[i] = seq.length();
      seqIds[i] = seq.seqId();
      slotMapping[i] = seq.blockTable().getLast() * this.blockSize + seq.lastBlockNumTokens() - 1;
    }

    this.stepContext.set(
      false, null, null, 0, 0, slotMapping, contextLens, this.prepareBlockTables(seqs), seqIds);
    return new PreparedInputs(Tensor.of(inputIds, n), Tensor.of(positions, n));
  }

  private record PreparedInputs(Tensor inputIds, Tensor positions) {
  }

  @SuppressWarnings("ArrayRecordComponent")
  private record SamplingControls(float[] temperatures, int[] topKs, float[] topPs) {
  }
}
