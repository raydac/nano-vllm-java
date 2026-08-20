package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Library {@link LlmModel}: weights, causal graph or embedding encoder, unpack, and engine lease.
 */
public final class LlmModelImpl extends LlmModel {

  private static final Set<String> KNOWN_OPTIONS = Set.of(
    LlmModel.OPTION_THINK_TAGS,
    LlmModel.OPTION_CHAT_SPECIALS
  );

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
  private final AtomicInteger liveEngines = new AtomicInteger();

  public LlmModelImpl(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final CausalLM network,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    this(path, hfConfig, weights, network, null, tokenizer, options);
  }

  public LlmModelImpl(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final EmbeddingEncoder encoder,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    this(path, hfConfig, weights, null, encoder, tokenizer, options);
  }

  private LlmModelImpl(
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

  public static LlmModelImpl peer(final LlmModel model) {
    requireNonNull(model, "model");
    if (model instanceof LlmModelImpl impl) {
      return impl;
    }
    throw new IllegalArgumentException("LlmModel must be a library-loaded instance");
  }

  public static Map<String, Object> copyAndValidateOptions(final Map<String, ?> options) {
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
    requireOptionType(copy, LlmModel.OPTION_THINK_TAGS, ThinkTags.class);
    requireOptionType(copy, LlmModel.OPTION_CHAT_SPECIALS, ChatSpecials.class);
    copy.putIfAbsent(LlmModel.OPTION_THINK_TAGS, ThinkTags.DEFAULT);
    copy.putIfAbsent(LlmModel.OPTION_CHAT_SPECIALS, ChatSpecials.DEFAULT);
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

  private static String weightsLabel(final WeightBag bag) {
    if (bag == null) {
      return "released";
    }
    if (bag.hasGemmaQat()) {
      return "qat";
    }
    return bag.hasPacked() ? "packed" : "dense";
  }

  @Override
  public Path path() {
    this.assertNotClosed();
    return this.path;
  }

  @Override
  public Config.HfConfig hfConfig() {
    this.assertNotClosed();
    return this.hfConfig;
  }

  @Override
  public Tokenizer tokenizer() {
    this.assertNotClosed();
    return this.tokenizer;
  }

  @Override
  public Map<String, Object> options() {
    this.assertNotClosed();
    return this.options;
  }

  @Override
  public ThinkTags thinkTags() {
    this.assertNotClosed();
    return (ThinkTags) this.options.get(LlmModel.OPTION_THINK_TAGS);
  }

  @Override
  public ChatSpecials chatSpecials() {
    this.assertNotClosed();
    return (ChatSpecials) this.options.get(LlmModel.OPTION_CHAT_SPECIALS);
  }

  @Override
  public String architectureName() {
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

  @Override
  public boolean isCausalModel() {
    this.assertNotClosed();
    return this.network.get() != null;
  }

  @Override
  public boolean isEmbeddingModel() {
    this.assertNotClosed();
    return this.encoder.get() != null;
  }

  @Override
  public boolean hasPackedWeights() {
    return this.requireWeights().hasPacked();
  }

  @Override
  public boolean isClosed() {
    return this.closed.get();
  }

  @Override
  public String toString() {
    Config.HfConfig cfg = this.hfConfig;
    WeightBag bag = this.weights.get();
    return ("LlmModel{kind=%s, architecture=%s, container=%s, path=%s, layers=%d, hidden=%d, "
      + "intermediate=%d, heads=%d/%d, headDim=%d, context=%d, vocab=%d, tensors=%s, weights=%s, "
      + "chatFormat=%s%s%s}").formatted(
      this.kindLabel(),
      this.architectureName(),
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

  private String containerLabel() {
    Path fileName = this.path.getFileName();
    String name = (fileName == null ? this.path.toString() : fileName.toString()).toLowerCase(ROOT);
    if (name.endsWith(".gguf")) {
      return "gguf";
    }
    return this.path.toString().contains("nanollvm-memory") ? "memory" : "folder";
  }

  private String thinkSuffix() {
    Object tags = this.options.get(LlmModel.OPTION_THINK_TAGS);
    if (tags instanceof ThinkTags custom && !ThinkTags.DEFAULT.equals(custom)) {
      return ", thinkTags=%s/%s".formatted(custom.open(), custom.close());
    }
    return this.tokenizer.invitesThinking() ? ", think=true" : "";
  }

  private String chatSpecialsSuffix() {
    Object specials = this.options.get(LlmModel.OPTION_CHAT_SPECIALS);
    if (specials instanceof ChatSpecials custom && !ChatSpecials.DEFAULT.equals(custom)) {
      return ", chatSpecials=%d".formatted(custom.markers().size());
    }
    return "";
  }

  @Override
  public float[] embed(final CharSequence text) {
    requireNonNull(text, "text");
    return this.embedAll(List.of(text))[0];
  }

  @Override
  public float[][] embed(final List<? extends CharSequence> texts) {
    requireNonNull(texts, "texts");
    return this.embedAll(texts);
  }

  @Override
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
    int cls = this.embeddingBosId();
    int sep = this.embeddingEosId();
    int[] ids = new int[pieces.size() + 2];
    ids[0] = cls;
    for (int i = 0; i < pieces.size(); i++) {
      ids[i + 1] = pieces.get(i);
    }
    ids[ids.length - 1] = sep;
    return ids;
  }

  private int embeddingBosId() {
    return this.embeddingSpecialId("[CLS]", "<s>");
  }

  private int embeddingEosId() {
    return this.embeddingSpecialId("[SEP]", "</s>");
  }

  private int embeddingSpecialId(final String bertToken, final String xlmToken) {
    return this.tokenizer.tokenId(bertToken)
      .or(() -> this.tokenizer.tokenId(xlmToken))
      .orElseThrow(() -> new IllegalStateException(
        "embedding tokenizer missing %s or %s".formatted(bertToken, xlmToken)));
  }

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

  public CausalLM resolveNetwork(final boolean allowUnpackParameters, final LlmListener io) {
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

  public void acquireEngine() {
    this.liveEngines.incrementAndGet();
  }

  public void releaseEngine() {
    this.liveEngines.decrementAndGet();
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
      WeightBag denseWeights = this.liveEngines.get() > 0
        ? currentWeights.asDense()
        : currentWeights.asDenseReleasingPacked();
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
