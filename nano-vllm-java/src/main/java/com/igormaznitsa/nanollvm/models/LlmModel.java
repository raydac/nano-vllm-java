package com.igormaznitsa.nanollvm.models;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.CausalLMFactory;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoder;
import com.igormaznitsa.nanollvm.models.internal.LlmModelAccess;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Loaded model: architecture weights, HF/GGUF config, and tokenizer.
 *
 * <p>Safe to share across threads and across many {@link LLM} instances (causal models). Mutable
 * inference state (KV cache, scheduler, sampling) lives on each {@link LLM}, not here. Load-time
 * options ({@link #options()}, including {@link #OPTION_THINK_TAGS} and
 * {@link #OPTION_CHAT_SPECIALS}) are frozen at
 * {@link LlmModelFactory#make} and never change. Embedding models expose
 * {@link #embed(CharSequence)} instead of chat/generate.
 *
 * <p>GGUF models keep quantized weights packed by default. Prefer unpacking at load with
 * {@link LlmModelFactory#make(Path, LlmListener, boolean)} ({@code true}) so float32 is built
 * directly from the mmap with no packed heap copy. {@link LLM.Builder#allowUnpackParameters()}
 * late-unpacks an already-packed causal model by installing a dense graph; packed tensors already
 * bound by existing engines are left intact (peak RAM may briefly hold packed + dense).
 *
 * <p>Construct via {@link LlmModelFactory#make(Path)}. Closing an {@link LLM} does not unload
 * this model — call {@link #close()} after every bound {@link LLM} is closed.
 *
 * @see LlmModelFactory
 * @see LLM
 */
public final class LlmModel implements AutoCloseable {

  /**
   * Factory option: a {@link ThinkTags} pair for chat parse and ChatML skip-seed.
   *
   * @since 1.1.0
   */
  public static final String OPTION_THINK_TAGS = "thinkTags";

  /**
   * Factory option: {@link ChatSpecials} searched in decoded assistant text when stripping
   * leftover chat markup from the visible answer.
   *
   * @since 1.1.0
   */
  public static final String OPTION_CHAT_SPECIALS = "chatSpecials";

  private static final Set<String> KNOWN_OPTIONS = Set.of(OPTION_THINK_TAGS, OPTION_CHAT_SPECIALS);

  static {
    LlmModelAccess.setResolver(LlmModel::resolveNetwork);
  }

  private final Path path;
  private final Config.HfConfig hfConfig;
  private final Tokenizer tokenizer;
  private final Map<String, Object> options;
  private final boolean embeddingModel;
  private final AtomicReference<WeightBag> weights;
  private final AtomicReference<CausalLM> network;
  private final AtomicReference<EmbeddingEncoder> encoder;
  private final ReentrantLock unpackLock = new ReentrantLock();
  private final AtomicBoolean closed = new AtomicBoolean();

  LlmModel(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final CausalLM network,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    this(path, hfConfig, weights, network, null, tokenizer, options);
  }

  LlmModel(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final EmbeddingEncoder encoder,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    this(path, hfConfig, weights, null, encoder, tokenizer, options);
  }

  private LlmModel(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final CausalLM network,
    final EmbeddingEncoder encoder,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    if ((network == null) == (encoder == null)) {
      throw new IllegalArgumentException("exactly one of network or encoder must be set");
    }
    this.path = requireNonNull(path, "path").toAbsolutePath().normalize();
    this.hfConfig = requireNonNull(hfConfig, "hfConfig");
    this.weights = new AtomicReference<>(requireNonNull(weights, "weights"));
    this.network = new AtomicReference<>(network);
    this.encoder = new AtomicReference<>(encoder);
    this.embeddingModel = encoder != null;
    this.tokenizer = requireNonNull(tokenizer, "tokenizer");
    this.options = copyAndValidateOptions(options);
  }

  static Map<String, Object> copyAndValidateOptions(final Map<String, ?> options) {
    requireNonNull(options, "options");
    Map<String, Object> copy = new LinkedHashMap<>();
    options.forEach((key, value) -> {
      requireNonNull(key, "option key");
      if (key.isBlank()) {
        throw new IllegalArgumentException("option key must not be blank");
      }
      requireNonNull(value, "option value for " + key);
      if (!KNOWN_OPTIONS.contains(key)) {
        throw new IllegalArgumentException("unknown model option: " + key);
      }
      copy.put(key, value);
    });
    requireOptionType(copy, OPTION_THINK_TAGS, ThinkTags.class);
    requireOptionType(copy, OPTION_CHAT_SPECIALS, ChatSpecials.class);
    copy.putIfAbsent(OPTION_THINK_TAGS, ThinkTags.DEFAULT);
    copy.putIfAbsent(OPTION_CHAT_SPECIALS, ChatSpecials.DEFAULT);
    return Map.copyOf(copy);
  }

  private static void requireOptionType(
    final Map<String, Object> options,
    final String key,
    final Class<?> expected
  ) {
    Object value = options.get(key);
    if (value != null && !expected.isInstance(value)) {
      throw new IllegalArgumentException(
        key + " must be a " + expected.getSimpleName() + ", got " + value.getClass().getName());
    }
  }

  /**
   * Checkpoint folder or GGUF file path this model was loaded from.
   */
  public Path path() {
    this.assertNotClosed();
    return this.path;
  }

  /**
   * Hugging Face / GGUF-mapped architecture config.
   */
  public Config.HfConfig hfConfig() {
    this.assertNotClosed();
    return this.hfConfig;
  }

  /**
   * Tokenizer bound to this checkpoint.
   */
  public Tokenizer tokenizer() {
    this.assertNotClosed();
    return this.tokenizer;
  }

  /**
   * Load-time options frozen by {@link LlmModelFactory} ({@link Map#copyOf}). Always contains
   * {@link #OPTION_THINK_TAGS} and {@link #OPTION_CHAT_SPECIALS} (library defaults when the caller
   * omitted them).
   *
   * @since 1.1.0
   */
  public Map<String, Object> options() {
    this.assertNotClosed();
    return this.options;
  }

  /**
   * Scratchpad open/close markers for chat parse and ChatML skip-seed.
   * {@link ThinkTags#DEFAULT} ({@code <think>} / {@code </think>}) unless
   * {@link #OPTION_THINK_TAGS} was set at load.
   *
   * @since 1.1.0
   */
  public ThinkTags thinkTags() {
    this.assertNotClosed();
    return (ThinkTags) this.options.get(OPTION_THINK_TAGS);
  }

  /**
   * Special-token strings searched in decoded assistant text when stripping chat markup.
   * {@link ChatSpecials#DEFAULT} unless {@link #OPTION_CHAT_SPECIALS} was set at load.
   *
   * @since 1.1.0
   */
  public ChatSpecials chatSpecials() {
    this.assertNotClosed();
    return (ChatSpecials) this.options.get(OPTION_CHAT_SPECIALS);
  }

  /**
   * Architecture id (e.g. {@code qwen3}, {@code gemma3}, {@code bert}). Safe to call after
   * {@link #close()}.
   */
  public String architectureName() {
    if (this.isEmbeddingModel()) {
      return this.requireEncoder().architectureName();
    }
    return this.requireNetwork().architectureName();
  }

  /**
   * {@code true} when this checkpoint is a causal chat/completion graph.
   *
   * @since 1.1.0
   */
  public boolean isCausalModel() {
    this.assertNotClosed();
    return this.network.get() != null;
  }

  /**
   * {@code true} when this checkpoint is an embedding encoder ({@link #embed(CharSequence)}).
   *
   * @since 1.1.0
   */
  public boolean isEmbeddingModel() {
    this.assertNotClosed();
    return this.encoder.get() != null;
  }

  /**
   * {@code true} when GGUF/QAT weights are still packed (not widened to float32).
   */
  public boolean hasPackedWeights() {
    return this.requireWeights().hasPacked();
  }

  /**
   * {@code true} after {@link #close()} (further accessors throw).
   */
  public boolean isClosed() {
    return this.closed.get();
  }

  private static String weightsLabel(final WeightBag bag) {
    if (bag == null) {
      return "released";
    }
    if (bag.hasGemmaQat()) {
      return "qat";
    }
    return bag.hasPacked() ? "packed" : "dense";
  }

  /**
   * Kind, architecture, container, sizes, packed/dense/qat, and chat format (safe after close).
   *
   * @since 1.1.0
   */
  @Override
  public String toString() {
    Config.HfConfig cfg = this.hfConfig;
    WeightBag bag = this.weights.get();
    return ("LlmModel{kind=%s, architecture=%s, container=%s, path=%s, layers=%d, hidden=%d, "
      + "intermediate=%d, heads=%d/%d, headDim=%d, context=%d, vocab=%d, tensors=%s, weights=%s, "
      + "chatFormat=%s%s%s}").formatted(
      this.kindLabel(),
      this.architectureId(),
      this.containerLabel(),
      this.path,
      cfg.numHiddenLayers(),
      cfg.hiddenSize(),
      cfg.intermediateSize(),
      cfg.numAttentionHeads(),
      cfg.numKeyValueHeads(),
      cfg.headDim(),
      cfg.maxPositionEmbeddings(),
      cfg.vocabSize(),
      bag == null ? "released" : Integer.toString(bag.size()),
      weightsLabel(bag),
      this.tokenizer.chatFormat(),
      this.thinkSuffix() + this.chatSpecialsSuffix(),
      this.closed.get() ? ", closed" : "");
  }

  private String kindLabel() {
    return this.embeddingModel ? "embedding" : "chat";
  }

  private String architectureId() {
    EmbeddingEncoder encoder = this.encoder.get();
    if (encoder != null) {
      return encoder.architectureName();
    }
    CausalLM network = this.network.get();
    if (network != null) {
      return network.architectureName();
    }
    return this.hfConfig.modelType();
  }

  private String containerLabel() {
    Path fileName = this.path.getFileName();
    String name = (fileName == null ? this.path.toString() : fileName.toString()).toLowerCase(ROOT);
    if (name.endsWith(".gguf")) {
      return "gguf";
    }
    return this.path.toString().contains("nanollvm-memory") ? "memory" : "folder";
  }

  private String thinkSuffix() {
    Object tags = this.options.get(OPTION_THINK_TAGS);
    if (tags instanceof ThinkTags custom && !ThinkTags.DEFAULT.equals(custom)) {
      return ", thinkTags=%s/%s".formatted(custom.open(), custom.close());
    }
    return this.tokenizer.invitesThinking() ? ", think=true" : "";
  }

  private String chatSpecialsSuffix() {
    Object specials = this.options.get(OPTION_CHAT_SPECIALS);
    if (specials instanceof ChatSpecials custom && !ChatSpecials.DEFAULT.equals(custom)) {
      return ", chatSpecials=%d".formatted(custom.markers().size());
    }
    return "";
  }

  /**
   * Encodes {@code text} to a single L2-normalized embedding vector (embedding models only).
   * Tokenizes and wraps with {@code [CLS]} / {@code [SEP]} when present in the vocab.
   *
   * @since 1.1.0
   */
  public float[] embed(final CharSequence text) {
    requireNonNull(text, "text");
    return this.embedAll(List.of(text))[0];
  }

  /**
   * Encodes each text to an L2-normalized embedding vector (embedding models only).
   *
   * @since 1.1.0
   */
  public float[][] embed(final List<? extends CharSequence> texts) {
    requireNonNull(texts, "texts");
    return this.embedAll(texts);
  }

  /**
   * Encodes already-tokenized ids to a single L2-normalized embedding (embedding models only).
   * Ids are used as-is — include special tokens such as {@code [CLS]} / {@code [SEP]} when required.
   *
   * @since 1.1.0
   */
  public float[] embed(final int[] tokenIds) {
    requireNonNull(tokenIds, "tokenIds");
    if (tokenIds.length == 0) {
      throw new IllegalArgumentException("tokenIds must not be empty");
    }
    return this.requireEncoder().encode(tokenIds.clone(), MatmulRuntime.sequential());
  }

  private float[][] embedAll(final List<? extends CharSequence> texts) {
    EmbeddingEncoder active = this.requireEncoder();
    if (texts.isEmpty()) {
      return new float[0][];
    }
    MatmulRuntime runtime = MatmulRuntime.sequential();
    float[][] out = new float[texts.size()][];
    for (int i = 0; i < texts.size(); i++) {
      CharSequence text = requireNonNull(texts.get(i), "texts[" + i + "]");
      out[i] = active.encode(this.wrapClsSep(this.tokenizer.encode(text.toString())), runtime);
    }
    return out;
  }

  private int[] wrapClsSep(final List<Integer> pieces) {
    int cls = this.tokenizer.tokenId("[CLS]").orElseThrow(
      () -> new IllegalStateException("embedding tokenizer missing [CLS]"));
    int sep = this.tokenizer.tokenId("[SEP]").orElseThrow(
      () -> new IllegalStateException("embedding tokenizer missing [SEP]"));
    int[] ids = new int[pieces.size() + 2];
    ids[0] = cls;
    for (int i = 0; i < pieces.size(); i++) {
      ids[i + 1] = pieces.get(i);
    }
    ids[ids.length - 1] = sep;
    return ids;
  }

  /**
   * Releases packed payloads and drops weight/network/encoder refs. Close engines first.
   */
  @Override
  public void close() {
    if (!this.closed.compareAndSet(false, true)) {
      return;
    }

    this.unpackLock.lock();
    try {
      WeightBag bag = this.weights.getAndSet(null);
      this.network.set(null);
      this.encoder.set(null);
      if (bag != null) {
        bag.releaseResources();
      }
    } finally {
      this.unpackLock.unlock();
    }
  }

  private CausalLM resolveNetwork(final boolean allowUnpackParameters, final LlmListener io) {
    if (this.isEmbeddingModel()) {
      throw new IllegalStateException(ModelSupport.chatMisuseMessage(this.architectureName()));
    }
    CausalLM current = this.requireNetwork();
    if (!allowUnpackParameters) {
      return current;
    }
    if (!this.requireWeights().hasPacked()) {
      return current;
    }
    return this.unpackCausalToDense(io == null ? LlmListeners.silent() : io);
  }

  private CausalLM unpackCausalToDense(final LlmListener io) {
    this.unpackLock.lock();
    try {
      WeightBag currentWeights = this.requireWeights();
      if (!currentWeights.hasPacked()) {
        return this.requireNetwork();
      }

      LlmListeners.info(io, null, "Unpacking GGUF parameters to float32…");
      long startedAtNanos = System.nanoTime();
      WeightBag denseWeights = currentWeights.asDense();
      CausalLM built = CausalLMFactory.create(this.hfConfig, denseWeights);
      this.weights.set(denseWeights);
      this.network.set(built);
      LlmListeners.infof(io, null, "Unpacked float32 graph ready (%s) in %.1fs%n",
        built.architectureName(), (System.nanoTime() - startedAtNanos) / 1e9);
      return built;
    } finally {
      this.unpackLock.unlock();
    }
  }

  private CausalLM requireNetwork() {
    this.assertNotClosed();
    CausalLM current = this.network.get();
    if (current == null) {
      throw new IllegalStateException(
        this.encoder.get() != null
          ? ModelSupport.chatMisuseMessage(this.architectureName())
          : "LlmModel is closed");
    }
    return current;
  }

  private EmbeddingEncoder requireEncoder() {
    this.assertNotClosed();
    EmbeddingEncoder current = this.encoder.get();
    if (current == null) {
      throw new IllegalStateException(
        this.network.get() != null
          ? ModelSupport.embedMisuseMessage(this.architectureName())
          : "LlmModel is closed");
    }
    return current;
  }

  private WeightBag requireWeights() {
    this.assertNotClosed();
    WeightBag bag = this.weights.get();
    if (bag == null) {
      throw new IllegalStateException("LlmModel is closed");
    }
    return bag;
  }

  private void assertNotClosed() {
    if (this.closed.get()) {
      throw new IllegalStateException("LlmModel is closed");
    }
  }
}
