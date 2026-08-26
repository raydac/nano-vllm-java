package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.LlmInSound;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmInTokenIds;
import com.igormaznitsa.nanollvm.models.LlmInput;
import com.igormaznitsa.nanollvm.models.LlmModalities;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.LlmOutEmbedding;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import com.igormaznitsa.nanollvm.models.LlmOutText;
import com.igormaznitsa.nanollvm.models.LlmOutput;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.audio.WavPcm;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Library {@link LlmModel}: weights, causal graph, embedding encoder, speech graph, or
 * synthesis graph, unpack, and engine lease.
 */
public final class LlmModelImpl extends LlmModel {

  private static final Set<String> KNOWN_OPTIONS = Set.of(
    LlmModel.OPTION_THINK_TAGS,
    LlmModel.OPTION_CHAT_SPECIALS,
    LlmModel.OPTION_OPTIONAL_DATA
  );

  private final Path path;
  private final Config.HfConfig hfConfig;
  private final Tokenizer tokenizer;
  private final Map<String, Object> options;
  private final boolean embeddingModel;
  private final boolean speechModel;
  private final boolean synthesisModel;
  private final LlmModalities modalities;
  private final LlmModalities usableModalities;
  private final AtomicReference<WeightBag> weights;
  private final AtomicReference<CausalLM> network;
  private final AtomicReference<EmbeddingEncoder> encoder;
  private final AtomicReference<SpeechToText> speech;
  private final AtomicReference<TextToSpeech> synthesis;
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
    this(path, hfConfig, weights, network, null, null, null, tokenizer, options);
  }

  public LlmModelImpl(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final EmbeddingEncoder encoder,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    this(path, hfConfig, weights, null, encoder, null, null, tokenizer, options);
  }

  public LlmModelImpl(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final SpeechToText speech,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    this(path, hfConfig, weights, null, null, speech, null, tokenizer, options);
  }

  public LlmModelImpl(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final TextToSpeech synthesis,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    this(path, hfConfig, weights, null, null, null, synthesis, tokenizer, options);
  }

  private LlmModelImpl(
    final Path path,
    final Config.HfConfig hfConfig,
    final WeightBag weights,
    final CausalLM network,
    final EmbeddingEncoder encoder,
    final SpeechToText speech,
    final TextToSpeech synthesis,
    final Tokenizer tokenizer,
    final Map<String, ?> options
  ) {
    int graphs = (network != null ? 1 : 0) + (encoder != null ? 1 : 0)
      + (speech != null ? 1 : 0) + (synthesis != null ? 1 : 0);
    if (graphs != 1) {
      throw new IllegalArgumentException(
        "exactly one of network, encoder, speech, or synthesis must be set");
    }
    this.path = requireNonNull(path, "path").toAbsolutePath().normalize();
    this.hfConfig = requireNonNull(hfConfig, "hfConfig");
    this.weights = new AtomicReference<>(requireNonNull(weights, "weights"));
    this.network = new AtomicReference<>(network);
    this.encoder = new AtomicReference<>(encoder);
    this.speech = new AtomicReference<>(speech);
    this.synthesis = new AtomicReference<>(synthesis);
    this.embeddingModel = encoder != null;
    this.speechModel = speech != null;
    this.synthesisModel = synthesis != null;
    this.tokenizer = requireNonNull(tokenizer, "tokenizer");
    this.options = copyAndValidateOptions(options);
    this.modalities = LlmModalities.ofCheckpoint(this.hfConfig, this.embeddingModel);
    this.usableModalities = LlmModalities.usable(
      this.embeddingModel, this.speechModel, this.synthesisModel);
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
      copy.put(key, LlmModel.OPTION_OPTIONAL_DATA.equals(key) ? freezeOptionalData(value) : value);
    });
    requireOptionType(copy, LlmModel.OPTION_THINK_TAGS, ThinkTags.class);
    requireOptionType(copy, LlmModel.OPTION_CHAT_SPECIALS, ChatSpecials.class);
    copy.putIfAbsent(LlmModel.OPTION_THINK_TAGS, ThinkTags.DEFAULT);
    copy.putIfAbsent(LlmModel.OPTION_CHAT_SPECIALS, ChatSpecials.DEFAULT);
    return Map.copyOf(copy);
  }

  private static Map<String, Object> freezeOptionalData(final Object value) {
    if (!(value instanceof Map<?, ?> raw)) {
      throw new IllegalArgumentException(
        LlmModel.OPTION_OPTIONAL_DATA + " must be a Map, got " + value.getClass().getName());
    }
    Map<String, Object> frozen = new LinkedHashMap<>();
    raw.forEach((dataKey, dataValue) -> {
      if (dataKey == null || dataKey.toString().isBlank()) {
        throw new IllegalArgumentException("optionalData key must not be blank");
      }
      requireNonNull(dataValue, "optionalData value for " + dataKey);
      String id = dataKey.toString();
      frozen.put(
        id,
        LlmOptionalData.ESPEAK_DATA.id().equals(id) ? LlmOptionalData.asPath(dataValue) :
          dataValue);
    });
    return Map.copyOf(frozen);
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
  public <T> Optional<T> optionalData(final LlmOptionalData.Key<T> key) {
    requireNonNull(key, "key");
    Object bag = this.options.get(LlmModel.OPTION_OPTIONAL_DATA);
    if (!(bag instanceof Map<?, ?> data) || !data.containsKey(key.id())) {
      return Optional.empty();
    }
    return Optional.of(LlmOptionalData.cast(key, data.get(key.id())));
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
    SpeechToText speech = this.speech.get();
    if (speech != null) {
      return speech.architectureName();
    }
    TextToSpeech synthesis = this.synthesis.get();
    if (synthesis != null) {
      return synthesis.architectureName();
    }
    return this.hfConfig.modelType();
  }

  @Override
  public LlmModalities modalities() {
    return this.modalities;
  }

  @Override
  public LlmModalities usableModalities() {
    return this.usableModalities;
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
  public boolean isSpeechModel() {
    this.assertNotClosed();
    return this.speech.get() != null;
  }

  @Override
  public boolean isSynthesisModel() {
    this.assertNotClosed();
    return this.synthesis.get() != null;
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
    return ("LlmModel{kind=%s, modalities=%s%s, architecture=%s, container=%s, path=%s, layers=%d, "
      + "hidden=%d, intermediate=%d, heads=%d/%d, headDim=%d, context=%d, vocab=%d, tensors=%s, "
      + "weights=%s, chatFormat=%s%s%s}").formatted(
      this.kindLabel(),
      this.modalities,
      this.usableSuffix(),
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

  private String usableSuffix() {
    return this.modalities.equals(this.usableModalities)
      ? ""
      : ", usable=" + this.usableModalities;
  }

  private String kindLabel() {
    if (this.synthesisModel) {
      return "synthesis";
    }
    if (this.speechModel) {
      return "speech";
    }
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

  public float[] embed(final int[] tokenIds, final MatmulRuntime runtime) {
    requireNonNull(tokenIds, "tokenIds");
    requireNonNull(runtime, "runtime");
    if (tokenIds.length == 0) {
      throw new IllegalArgumentException("tokenIds must not be empty");
    }
    return this.requireEncoder().encode(tokenIds.clone(), runtime);
  }

  public String transcribe(final byte[] wav, final Locale language, final MatmulRuntime runtime) {
    requireNonNull(wav, "wav");
    requireNonNull(runtime, "runtime");
    WavPcm.MonoPcm audio = WavPcm.read(wav);
    return this.transcribe(audio.samples(), audio.sampleRate(), language, runtime);
  }

  public String transcribe(
    final float[] pcm,
    final int sampleRate,
    final Locale language,
    final MatmulRuntime runtime
  ) {
    requireNonNull(pcm, "pcm");
    requireNonNull(runtime, "runtime");
    return this.requireSpeech().transcribe(pcm, sampleRate, language, this.tokenizer, runtime);
  }

  public LlmOutSoundData synthesizeSound(final CharSequence text, final MatmulRuntime runtime) {
    return this.synthesizeSound(text, runtime, new Random(1L));
  }

  public LlmOutSoundData synthesizeSound(
    final CharSequence text,
    final MatmulRuntime runtime,
    final Random random
  ) {
    requireNonNull(text, "text");
    requireNonNull(runtime, "runtime");
    requireNonNull(random, "random");
    TextToSpeech tts = this.requireSynthesis();
    Path espeakData = this.optionalData(LlmOptionalData.ESPEAK_DATA)
      .orElseGet(() -> this.path.resolve("espeak-ng-data"));
    float[] samples = tts.synthesize(text, espeakData, runtime, random);
    return new LlmOutSoundData(WavPcm.toWav16Le(samples, tts.sampleRate()), tts.sampleRate());
  }

  @Override
  public LlmOutput generate(final LlmInput input, final LlmModality outputModality) {
    return this.generate(input, outputModality, MatmulRuntime.sequential());
  }

  public LlmOutput generate(
    final LlmInput input,
    final LlmModality outputModality,
    final MatmulRuntime runtime
  ) {
    return this.generate(
      input,
      outputModality,
      runtime,
      outputModality == LlmModality.AUDIO ? new Random(1L) : new Random());
  }

  public LlmOutput generate(
    final LlmInput input,
    final LlmModality outputModality,
    final MatmulRuntime runtime,
    final Random random
  ) {
    requireNonNull(input, "input");
    requireNonNull(outputModality, "outputModality");
    requireNonNull(runtime, "runtime");
    requireNonNull(random, "random");
    this.assertNotClosed();

    return switch (input) {
      case LlmInText text when outputModality == LlmModality.EMBEDDING ->
        new LlmOutEmbedding(this.embedAll(List.of(text.text()), runtime)[0]);
      case LlmInTokenIds ids when outputModality == LlmModality.EMBEDDING ->
        new LlmOutEmbedding(this.embed(ids.tokenIds(), runtime));
      case LlmInText text when outputModality == LlmModality.AUDIO ->
        this.synthesizeSound(text.text(), runtime, random);
      case LlmInText ignored when outputModality == LlmModality.TEXT ->
        throw new IllegalStateException(
          "text completion requires LLM.builder(model).build() then "
            + "LLM.generate(LlmInput, LlmModality); LlmModel has no KV engine");
      case LlmInSound sound when outputModality == LlmModality.TEXT ->
        new LlmOutText(this.transcribeSound(sound, runtime));
      default -> throw new IllegalArgumentException(
        "unsupported generate pair: %s → %s (usable %s)".formatted(
          input.getClass().getSimpleName(),
          outputModality.wireName(),
          this.usableModalities()));
    };
  }

  private String transcribeSound(final LlmInSound sound, final MatmulRuntime runtime) {
    if (sound.isWav()) {
      return this.transcribe(sound.wav(), sound.language(), runtime);
    }
    return this.transcribe(sound.pcm(), sound.sampleRate(), sound.language(), runtime);
  }

  public float[][] embedAll(final List<? extends CharSequence> texts, final MatmulRuntime runtime) {
    requireNonNull(runtime, "runtime");
    EmbeddingEncoder active = this.requireEncoder();
    if (texts.isEmpty()) {
      return new float[0][];
    }
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
      this.speech.set(null);
      this.synthesis.set(null);
      if (bag != null) {
        bag.releaseResources();
      }
    } finally {
      this.unpackLock.unlock();
    }
  }

  public CausalLM resolveNetwork(final boolean allowUnpackParameters, final LlmListener io) {
    if (this.isEmbeddingModel() || this.isSpeechModel() || this.isSynthesisModel()) {
      throw new IllegalStateException(
        this.isSynthesisModel()
          ? ModelSupport.synthesisEngineMisuseMessage(this.architectureName())
          : this.isSpeechModel()
          ? ModelSupport.speechEngineMisuseMessage(this.architectureName())
          : ModelSupport.chatMisuseMessage(this.architectureName()));
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
        this.synthesis.get() != null
          ? ModelSupport.synthesisEngineMisuseMessage(this.architectureName())
          : this.speech.get() != null
          ? ModelSupport.speechEngineMisuseMessage(this.architectureName())
          : this.encoder.get() != null
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
        this.synthesis.get() != null
          ? ModelSupport.synthesisEmbedMisuseMessage(this.architectureName())
          : this.speech.get() != null
          ? ModelSupport.speechEmbedMisuseMessage(this.architectureName())
          : this.network.get() != null
          ? ModelSupport.embedMisuseMessage(this.architectureName())
          : "LlmModel is closed");
    }
    return current;
  }

  private SpeechToText requireSpeech() {
    this.assertNotClosed();
    SpeechToText current = this.speech.get();
    if (current == null) {
      throw new IllegalStateException(
        ModelSupport.transcribeMisuseMessage(this.architectureName()));
    }
    return current;
  }

  private TextToSpeech requireSynthesis() {
    this.assertNotClosed();
    TextToSpeech current = this.synthesis.get();
    if (current == null) {
      throw new IllegalStateException(
        ModelSupport.synthesizeMisuseMessage(this.architectureName()));
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
