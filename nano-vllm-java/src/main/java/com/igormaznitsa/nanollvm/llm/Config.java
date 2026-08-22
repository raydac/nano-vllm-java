package com.igormaznitsa.nanollvm.llm;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Immutable engine layout sealed when an {@link LLM} is built: context length, scheduler
 * batch limits, KV paging, stop tokens, and the Hugging Face / GGUF architecture blueprint.
 *
 * <p>Applications rarely construct this directly. {@link LLM.Builder} copies its knobs into a
 * {@link Builder} and calls {@link Builder#build()}. EOS / stop ids come from the tokenizer
 * unless overridden. KV block count is estimated from heap ({@link #kvHeapFraction()}) when
 * unset. The object is read-only after construction; changing context or stops requires a new
 * {@link LLM}.
 *
 * @see LLM.Builder
 * @see HfConfig
 */
public final class Config {

  private final Path model;
  private final int maxNumBatchedTokens;
  private final int maxNumSeqs;
  private final int maxModelLen;
  private final float kvHeapFraction;
  private final int cpuThreads;
  private final HfConfig hfConfig;
  private final int kvcacheBlockSize;
  private final int eos;
  private final List<Integer> stopTokenIds;
  private final int numKvcacheBlocks;

  private Config(final Builder b) {
    this.model = b.model;
    this.maxNumBatchedTokens = b.maxNumBatchedTokens;
    this.maxNumSeqs = b.maxNumSeqs;
    this.kvHeapFraction = b.kvHeapFraction;
    this.cpuThreads = b.cpuThreads;
    this.kvcacheBlockSize = b.kvcacheBlockSize;

    if (b.hfConfig == null && !Files.isDirectory(this.model)) {
      throw new IllegalArgumentException("model path is not a directory: " + this.model);
    }
    if (b.hfConfig != null
      && !Files.isDirectory(this.model)
      && !Files.isRegularFile(this.model)) {
      throw new IllegalArgumentException("model path not found: " + this.model);
    }
    if (this.kvcacheBlockSize % 256 != 0) {
      throw new IllegalArgumentException("kvcacheBlockSize must be multiple of 256");
    }
    if (this.cpuThreads < 1) {
      throw new IllegalArgumentException("cpuThreads must be >= 1, got " + this.cpuThreads);
    }
    if (!(this.kvHeapFraction > 0f && this.kvHeapFraction <= 1f)) {
      throw new IllegalArgumentException(
        "kvHeapFraction must be in (0, 1], got " + this.kvHeapFraction);
    }

    if (b.hfConfig != null) {
      this.hfConfig = b.hfConfig;
    } else {
      try {
        this.hfConfig = HfConfig.load(this.model.resolve(CONFIG_JSON));
      } catch (IOException e) {
        throw new IllegalArgumentException("failed to load " + CONFIG_JSON + " from " + this.model,
          e);
      }
    }
    this.maxModelLen = Math.min(b.maxModelLen, this.hfConfig.maxPositionEmbeddings());
    this.eos = b.eos;
    this.stopTokenIds = List.copyOf(b.stopTokenIds);
    this.numKvcacheBlocks = Config.resolveNumKvcacheBlocks(
      b.numKvcacheBlocks,
      this.maxNumSeqs,
      this.maxModelLen,
      this.kvcacheBlockSize,
      this.kvHeapFraction,
      this.hfConfig);
  }

  private static int resolveNumKvcacheBlocks(
    final int configured,
    final int maxNumSeqs,
    final int maxModelLen,
    final int blockSize,
    final float kvHeapFraction,
    final HfConfig hf) {
    if (configured > 0) {
      return configured;
    }

    int blocksPerSeq = (maxModelLen + blockSize - 1) / blockSize;
    int estimated = Math.max(maxNumSeqs * blocksPerSeq, 128);
    long free = Runtime.getRuntime().maxMemory();
    long bytesPerBlock = 0L;
    for (int layer = 0; layer < hf.numHiddenLayers(); layer++) {
      if (hf.isKvSharedLayer(layer)) {
        continue;
      }
      bytesPerBlock +=
        2L * blockSize * hf.numKeyValueHeads() * hf.layerHeadDim(layer) * Float.BYTES;
    }
    if (bytesPerBlock <= 0L) {
      bytesPerBlock = 2L * hf.numHiddenLayers() * blockSize
        * hf.numKeyValueHeads() * hf.headDim() * Float.BYTES;
    }
    int heapCap = (int) Math.max(32, (long) (free * kvHeapFraction) / Math.max(1, bytesPerBlock));
    int resolved = Math.min(estimated, heapCap);
    if (resolved <= 0) {
      throw new IllegalStateException("numKvcacheBlocks must be > 0");
    }
    return resolved;
  }

  /**
   * Starts a builder from a checkpoint folder or GGUF file. Architecture fields are read from
   * {@code config.json} (or the GGUF-mapped equivalent) at {@link Builder#build()} unless an
   * {@link LlmModel} is supplied via {@link #builder(LlmModel)}.
   *
   * @param model directory or GGUF path; must exist by build time
   * @return a new builder
   */
  public static Builder builder(final Path model) {
    return new Builder(model);
  }

  /**
   * Starts a builder from an already-loaded checkpoint. Path and {@link LlmModel#hfConfig()} are
   * taken from the model so JSON is not re-read. Prefer this from {@link LLM.Builder}; the engine
   * owns the resulting {@link Config}.
   *
   * @param model loaded weights; must not be {@code null}
   * @return a new builder
   * @throws NullPointerException if {@code model} is {@code null}
   */
  public static Builder builder(final LlmModel model) {
    requireNonNull(model, "model");
    return new Builder(model.path(), model.hfConfig());
  }

  /**
   * Normalized filesystem path of the checkpoint folder or GGUF file this engine was sized for.
   * Informational after load; weight bytes live on {@link LlmModel}, not here.
   *
   * @return absolute normalized path; never {@code null}
   */
  public Path model() {
    return this.model;
  }

  /**
   * Prefill batch cap: total prompt tokens the scheduler may pack into one forward step across
   * sequences. Larger values raise throughput and peak activation memory. Default on the builder
   * is {@code 16384}.
   *
   * @return token budget {@code >= 1} as sealed at build
   */
  public int maxNumBatchedTokens() {
    return this.maxNumBatchedTokens;
  }

  /**
   * Maximum concurrent sequences in the scheduler — how many prompts may be active in one
   * {@link LLM#generate} batch. Default {@code 512}.
   *
   * @return sequence cap as sealed at build
   */
  public int maxNumSeqs() {
    return this.maxNumSeqs;
  }

  /**
   * Context length in tokens used for KV pages and chat truncation. Capped by
   * {@link HfConfig#maxPositionEmbeddings()} at build. Prompt plus generated tokens must fit;
   * {@link LLM#generate} also shrinks {@code maxTokens} to remaining room.
   *
   * @return context cap as sealed at build
   */
  public int maxModelLen() {
    return this.maxModelLen;
  }

  /**
   * Fraction of {@link Runtime#maxMemory()} used when auto-sizing KV blocks
   * ({@link #numKvcacheBlocks()} was unset at build). Default {@code 0.25}. Raise only when the
   * JVM heap can spare more for KV; lower if the process also holds large weights.
   *
   * @return fraction in {@code (0, 1]}
   */
  public float kvHeapFraction() {
    return this.kvHeapFraction;
  }

  /**
   * CPU workers for dense matmul. {@code 1} means sequential (calling thread only, no pool).
   * This is the count sealed into the engine; {@link LLM.Builder#cpuThreads(int)} wins over
   * {@code -Dnanollvm.cpu.threads}.
   *
   * @return worker count {@code >= 1}
   */
  public int cpuThreads() {
    return this.cpuThreads;
  }

  /**
   * Hugging Face / GGUF-mapped architecture blueprint used to size the graph (heads, hidden size,
   * RoPE, layer types). Same object as {@link LlmModel#hfConfig()} when the engine was built from
   * a loaded model.
   *
   * @return immutable architecture config; never {@code null}
   */
  public HfConfig hfConfig() {
    return this.hfConfig;
  }

  /**
   * Primary end-of-sequence token id — the first element of {@link #stopTokenIds()}. Hitting any
   * stop id finishes a sequence unless sampling {@code ignoreEos} is true.
   *
   * @return tokenizer / override EOS id
   */
  public int eos() {
    return this.eos;
  }

  /**
   * Token ids that end a generate (immutable copy). Comes from the tokenizer stop list, or
   * {@link Tokenizer#eosTokenId()} when that list is empty, unless the builder overrode them.
   *
   * @return non-empty unmodifiable list
   */
  public List<Integer> stopTokenIds() {
    return this.stopTokenIds;
  }

  /**
   * Tokens stored in one paged-KV block. Must be a multiple of 256 (default {@code 256}). Larger
   * blocks waste less page-table overhead at long context; smaller blocks pack short sequences
   * more tightly.
   *
   * @return block size in tokens
   */
  public int kvcacheBlockSize() {
    return this.kvcacheBlockSize;
  }

  /**
   * Number of KV blocks allocated for this engine. Auto-sized from heap, {@link #maxNumSeqs()},
   * and {@link #maxModelLen()} when the builder left it at {@code -1}. Too few blocks fail long
   * or highly concurrent generates.
   *
   * @return block count {@code > 0}
   */
  public int numKvcacheBlocks() {
    return this.numKvcacheBlocks;
  }

  /**
   * Fluent configurator for {@link Config}. Applications usually go through {@link LLM.Builder}
   * instead; those knobs are copied here at engine build. {@link #applyTokenizer(Tokenizer)}
   * should run before {@link #build()} so EOS / stop ids are not left at {@code -1}.
   */
  public static final class Builder {
    private final Path model;
    private final HfConfig hfConfig;
    private int maxNumBatchedTokens = 16384;
    private int maxNumSeqs = 512;
    private int maxModelLen = 4096;
    private float kvHeapFraction = 0.25f;
    private int cpuThreads = 1;
    private int eos = -1;
    private List<Integer> stopTokenIds = List.of();
    private int kvcacheBlockSize = 256;
    private int numKvcacheBlocks = -1;

    private Builder(final Path model) {
      this(model, null);
    }

    private Builder(final Path model, final HfConfig hfConfig) {
      this.model = requireNonNull(model, "model").toAbsolutePath().normalize();
      this.hfConfig = hfConfig;
    }

    /**
     * Prefill batch cap: tokens packed across sequences in one forward step. Larger values raise
     * throughput and peak activation memory. Default {@code 16384}.
     *
     * @param v token budget
     * @return {@code this}
     */
    public Builder maxNumBatchedTokens(final int v) {
      this.maxNumBatchedTokens = v;
      return this;
    }

    /**
     * Maximum concurrent sequences in the scheduler — how many prompts may be active in one
     * {@link LLM#generate} batch. Default {@code 512}.
     *
     * @param v sequence cap
     * @return {@code this}
     */
    public Builder maxNumSeqs(final int v) {
      this.maxNumSeqs = v;
      return this;
    }

    /**
     * Context length cap used for KV pages and chat truncation. At {@link #build()} this is
     * {@code min(value, hfConfig.maxPositionEmbeddings())}. Prompt plus generated tokens must fit.
     *
     * @param v requested context length in tokens
     * @return {@code this}
     */
    public Builder maxModelLen(final int v) {
      this.maxModelLen = v;
      return this;
    }

    /**
     * Fraction of {@link Runtime#maxMemory()} used when auto-sizing KV blocks
     * ({@link #numKvcacheBlocks(int)} is {@code -1}). Default {@code 0.25}. Must be in
     * {@code (0, 1]} at {@link #build()}.
     *
     * @param v heap fraction for KV estimate
     * @return {@code this}
     */
    public Builder kvHeapFraction(final float v) {
      this.kvHeapFraction = v;
      return this;
    }

    /**
     * CPU workers for dense matmul. {@code 1} is sequential (calling thread only). Must be
     * {@code >= 1} at {@link #build()}.
     *
     * @param v worker count
     * @return {@code this}
     */
    public Builder cpuThreads(final int v) {
      this.cpuThreads = v;
      return this;
    }

    /**
     * Primary EOS id. When {@link #stopTokenIds(List)} is still empty, this also becomes the sole
     * stop id. Prefer {@link #applyTokenizer(Tokenizer)} so vocab stops win, then override with
     * {@link #stopTokenIds(List)} if needed.
     *
     * @param v token id
     * @return {@code this}
     */
    public Builder eos(final int v) {
      this.eos = v;
      if (this.stopTokenIds.isEmpty()) {
        this.stopTokenIds = List.of(v);
      }
      return this;
    }

    /**
     * Seals EOS / stop ids from the model tokenizer. Non-empty {@link Tokenizer#stopTokenIds()}
     * win; otherwise {@link Tokenizer#eosTokenId()} is used when EOS was not set explicitly.
     * Call {@link #stopTokenIds(List)} after this to override the tokenizer pair.
     *
     * @param tokenizer model tokenizer; must not be {@code null}
     * @return {@code this}
     * @throws NullPointerException if {@code tokenizer} is {@code null}
     */
    public Builder applyTokenizer(final Tokenizer tokenizer) {
      requireNonNull(tokenizer, "tokenizer");
      List<Integer> stops = tokenizer.stopTokenIds();
      if (!stops.isEmpty()) {
        this.stopTokenIds = List.copyOf(stops);
        this.eos = stops.getFirst();
        return this;
      }
      if (this.eos < 0) {
        this.eos = tokenizer.eosTokenId();
      }
      if (this.stopTokenIds.isEmpty()) {
        this.stopTokenIds = List.of(this.eos);
      }
      return this;
    }

    /**
     * Replaces EOS / stop ids. The first id is {@link Config#eos()}. Must be non-empty.
     * Apply after {@link #applyTokenizer(Tokenizer)} so the caller wins over vocab stops.
     *
     * @param ids non-empty stop-token ids; must not be {@code null}
     * @return {@code this}
     * @throws NullPointerException     if {@code ids} is {@code null}
     * @throws IllegalArgumentException if {@code ids} is empty
     * @since 1.1.0
     */
    public Builder stopTokenIds(final List<Integer> ids) {
      requireNonNull(ids, "stopTokenIds");
      if (ids.isEmpty()) {
        throw new IllegalArgumentException("stopTokenIds must not be empty");
      }
      this.stopTokenIds = List.copyOf(ids);
      this.eos = this.stopTokenIds.getFirst();
      return this;
    }

    /**
     * Tokens stored in one paged-KV block. Must be a multiple of 256 at {@link #build()}
     * (default {@code 256}).
     *
     * @param v block size in tokens
     * @return {@code this}
     */
    public Builder kvcacheBlockSize(final int v) {
      this.kvcacheBlockSize = v;
      return this;
    }

    /**
     * Explicit KV-block count. {@code -1} (default) auto-sizes from heap, sequence cap, and
     * context length at {@link #build()}. The resolved count must be {@code > 0}.
     *
     * @param v explicit block count, or {@code -1} to auto-size
     * @return {@code this}
     */
    public Builder numKvcacheBlocks(final int v) {
      this.numKvcacheBlocks = v;
      return this;
    }

    /**
     * Seals EOS / stop ids and KV-block count, then returns an immutable {@link Config}.
     * Fails if the model path is missing, KV block size is not a multiple of 256,
     * {@code cpuThreads < 1}, or the KV heap estimate resolves to zero blocks.
     *
     * @return a new {@link Config}
     * @throws IllegalArgumentException if path, block size, threads, or heap fraction is invalid
     * @throws IllegalStateException    if auto-sized {@code numKvcacheBlocks} is not {@code > 0}
     */
    public Config build() {
      return new Config(this);
    }
  }

  /**
   * Parsed Hugging Face {@code config.json} (or GGUF-mapped equivalent) used to size the graph.
   *
   * <p>Apps normally read this from {@link com.igormaznitsa.nanollvm.models.LlmModel#hfConfig()}
   * after load rather than constructing it. {@link #load(Path)} / {@link #parse(String)} fill
   * fields from JSON keys ({@code hidden_size}, {@code num_attention_heads}, …). Missing keys use
   * the defaults noted below. Maps and lists are unmodifiable copies. Immutable; safe to share.
   *
   * @param vocabSize             {@code vocab_size}; tokenizer / LM-head width
   * @param hiddenSize            {@code hidden_size}; residual-stream width
   * @param intermediateSize      {@code intermediate_size}; MLP expand width
   * @param numHiddenLayers       {@code num_hidden_layers}; stacked blocks
   * @param numAttentionHeads     {@code num_attention_heads} (or {@code num_heads})
   * @param numKeyValueHeads      {@code num_key_value_heads}; GQA groups (defaults to query heads)
   * @param headDim               {@code head_dim}, or {@code hidden_size / num_attention_heads}
   * @param maxPositionEmbeddings {@code max_position_embeddings}; context-length ceiling
   *                              (default {@code 32768} when absent)
   * @param rmsNormEps            {@code rms_norm_eps} or {@code norm_eps} (default {@code 1e-6})
   * @param hiddenAct             {@code hidden_act} (default {@code silu}); see
   *                              {@link #effectiveActivation()}
   * @param tieWordEmbeddings     {@code tie_word_embeddings}; input embed and LM head share a matrix
   * @param attentionBias         {@code attention_bias}
   * @param ropeTheta             {@code rope_theta} / {@code rope_parameters.rope_theta}
   *                              (default {@code 1e6})
   * @param ropeScaling           {@code rope_scaling} object; empty map when absent
   * @param torchDtype            {@code torch_dtype} (default {@code float16}); informational
   * @param modelType             {@code model_type} architecture id ({@code qwen3}, {@code llama}, …)
   * @param architectures         {@code architectures} class names; empty list when absent
   * @param hiddenActivation      {@code hidden_activation} when present; wins over {@code hiddenAct}
   * @param slidingWindow         {@code sliding_window}; {@code 0} means no window
   * @param layerTypes            {@code layer_types} (Gemma / LFM2 hybrid); empty when absent
   * @param ropeLocalBaseFreq     {@code rope_local_base_freq} (default {@code 10_000})
   * @param queryPreAttnScalar    {@code query_pre_attn_scalar}; {@code 0} → use {@code headDim}
   *                              in {@link #attentionScale()}
   * @param convLCache            {@code conv_L_cache} (LFM2 short-conv state length)
   * @param visionConfigPresent   {@code true} when vision/image/video keys exist (unsupported VLMs
   *                              except Gemma 4 text load, which skips those towers)
   * @param imageConfigPresent    {@code true} when {@code vision_config} or {@code image_token_id}
   *                              is present
   * @param audioConfigPresent    {@code true} when {@code audio_config} or {@code audio_token_id}
   *                              is present
   * @param videoConfigPresent    {@code true} when {@code video_config} or {@code video_token_id}
   *                              is present
   * @param nestedTextConfig      {@code true} when {@code text_config} is a nested object
   * @param gemma4                Gemma 4 text extras; {@code null} for other families
   */
  public record HfConfig(
    int vocabSize,
    int hiddenSize,
    int intermediateSize,
    int numHiddenLayers,
    int numAttentionHeads,
    int numKeyValueHeads,
    int headDim,
    int maxPositionEmbeddings,
    float rmsNormEps,
    String hiddenAct,
    boolean tieWordEmbeddings,
    boolean attentionBias,
    float ropeTheta,
    Map<String, Object> ropeScaling,
    String torchDtype,
    String modelType,
    List<String> architectures,
    String hiddenActivation,
    int slidingWindow,
    List<String> layerTypes,
    float ropeLocalBaseFreq,
    float queryPreAttnScalar,
    int convLCache,
    boolean visionConfigPresent,
    boolean imageConfigPresent,
    boolean audioConfigPresent,
    boolean videoConfigPresent,
    boolean nestedTextConfig,
    Gemma4Text gemma4
  ) {
    public HfConfig {
      ropeScaling = freezeStringKeyedMap(ropeScaling);
      architectures = architectures == null ? List.of() : List.copyOf(architectures);
      layerTypes = layerTypes == null ? List.of() : List.copyOf(layerTypes);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> freezeStringKeyedMap(final Map<String, Object> map) {
      if (map == null || map.isEmpty()) {
        return Map.of();
      }
      return (Map<String, Object>) freezeJsonValue(map);
    }

    private static Object freezeJsonValue(final Object value) {
      if (value instanceof Map<?, ?> map) {
        Map<String, Object> frozen = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
          frozen.put(String.valueOf(entry.getKey()), freezeJsonValue(entry.getValue()));
        }
        return Map.copyOf(frozen);
      }
      if (value instanceof List<?> list) {
        return list.stream().map(HfConfig::freezeJsonValue).toList();
      }
      return value;
    }

    /**
     * Reads Hugging Face {@code config.json} from disk and {@link #parse(String) parses} it.
     * Gemma 4 nested {@code text_config} is flattened. Vision/audio-only checkpoints still parse
     * but {@link com.igormaznitsa.nanollvm.models.ModelSupport} rejects them at load.
     *
     * @param configJson path to a Hugging Face {@code config.json}
     * @return parsed blueprint
     * @throws IOException          if the file cannot be read
     * @throws NullPointerException if {@code configJson} is {@code null}
     */
    public static HfConfig load(final Path configJson) throws IOException {
      return parse(Files.readString(configJson));
    }

    /**
     * Parses a Hugging Face {@code config.json} body (or GGUF-mapped equivalent JSON). Missing
     * keys use the defaults documented on this record. Prefer {@link LlmModel#hfConfig()} after
     * a successful load rather than calling this from application code.
     *
     * @param configJson JSON object text; must not be {@code null}
     * @return parsed blueprint
     * @throws NullPointerException if {@code configJson} is {@code null}
     * @since 1.1.0
     */
    public static HfConfig parse(final String configJson) {
      Map<String, Object> root = Json.parseObject(requireNonNull(configJson, "configJson"));
      String modelType = Json.asString(root.get("model_type"));
      boolean nestedText = root.get("text_config") instanceof Map<?, ?>;
      ModalityFlags flags = modalityFlags(root);
      Map<String, Object> m = isGemma4Family(modelType) && nestedText
        ? flattenGemma4Text(root)
        : root;
      if (isGemma4Family(modelType)) {
        modelType = Json.asString(root.get("model_type"));
      } else {
        modelType = Json.asString(m.get("model_type"));
      }
      int hiddenSize = Json.asInt(m.get("hidden_size"), 0);
      int numAttentionHeads = Json.asInt(m.get("num_attention_heads"), 0);
      if (numAttentionHeads == 0) {
        numAttentionHeads = Json.asInt(m.get("num_heads"), 0);
      }
      int headDim = m.containsKey("head_dim")
        ? Json.asInt(m.get("head_dim"), 0)
        : (numAttentionHeads == 0 ? 0 : hiddenSize / numAttentionHeads);
      String hiddenAct = Json.asString(m.getOrDefault("hidden_act", "silu"));
      String hiddenActivation = Json.asString(m.get("hidden_activation"));
      List<String> architectures = null;
      List<Object> archArr = Json.asArray(root.containsKey("architectures")
        ? root.get("architectures")
        : m.get("architectures"));
      if (archArr != null) {
        architectures = archArr.stream().map(Json::asString).toList();
      }
      List<String> layerTypes = null;
      List<Object> layerArr = Json.asArray(m.get("layer_types"));
      if (layerArr != null) {
        layerTypes = layerArr.stream().map(Json::asString).toList();
      }
      float queryPre = m.containsKey("query_pre_attn_scalar")
        ? Json.asFloat(m.get("query_pre_attn_scalar"), headDim)
        : 0f;
      float rmsEps = m.containsKey("rms_norm_eps")
        ? Json.asFloat(m.get("rms_norm_eps"), 1e-6f)
        : Json.asFloat(m.get("norm_eps"), 1e-6f);
      RopeBases rope = resolveRopeBases(m);
      return new HfConfig(
        Json.asInt(m.get("vocab_size"), 0),
        hiddenSize,
        Json.asInt(m.get("intermediate_size"), 0),
        Json.asInt(m.get("num_hidden_layers"), 0),
        numAttentionHeads,
        Json.asInt(m.get("num_key_value_heads"), numAttentionHeads),
        headDim,
        Json.asInt(m.get("max_position_embeddings"), 32768),
        rmsEps,
        hiddenAct,
        Json.asBoolean(m.get("tie_word_embeddings"),
          Json.asBoolean(root.get("tie_word_embeddings"), false)),
        Json.asBoolean(m.get("attention_bias"), false),
        rope.theta(),
        Json.asObject(m.get("rope_scaling")),
        Json.asString(m.getOrDefault("torch_dtype", "float16")),
        modelType,
        architectures,
        hiddenActivation,
        Json.asInt(m.get("sliding_window"), 0),
        layerTypes,
        rope.localBaseFreq(),
        queryPre,
        Json.asInt(m.get("conv_L_cache"), 0),
        flags.vision(),
        flags.image(),
        flags.audio(),
        flags.video(),
        nestedText,
        isGemma4Family(modelType) ? parseGemma4Text(m, rope.partialRotaryFactor()) : null
      );
    }

    private static ModalityFlags modalityFlags(final Map<String, Object> root) {
      boolean image = root.containsKey("vision_config") || root.containsKey("image_token_id");
      boolean audio = root.containsKey("audio_config") || root.containsKey("audio_token_id");
      boolean video = root.containsKey("video_config") || root.containsKey("video_token_id");
      return new ModalityFlags(image, audio, video);
    }

    private record ModalityFlags(boolean image, boolean audio, boolean video) {
      boolean vision() {
        return this.image || this.video;
      }
    }

    static boolean isGemma4Family(final String modelType) {
      return "gemma4".equals(modelType) || "gemma4_text".equals(modelType);
    }

    private static Map<String, Object> flattenGemma4Text(final Map<String, Object> root) {
      Map<String, Object> text = Json.asObject(root.get("text_config"));
      Map<String, Object> merged = new LinkedHashMap<>(text);
      merged.put("model_type", Json.asString(root.get("model_type")));
      return merged;
    }

    private static RopeBases resolveRopeBases(final Map<String, Object> m) {
      float theta = Json.asFloat(m.get("rope_theta"), 1_000_000f);
      float local = Json.asFloat(m.get("rope_local_base_freq"), 10_000f);
      float partial = 0.25f;
      Map<String, Object> ropeParams = Json.asObject(m.get("rope_parameters"));
      if (ropeParams == null) {
        return new RopeBases(theta, local, partial);
      }
      if (ropeParams.containsKey("rope_theta")) {
        theta = Json.asFloat(ropeParams.get("rope_theta"), theta);
      }
      Map<String, Object> sliding = Json.asObject(ropeParams.get("sliding_attention"));
      if (sliding != null) {
        local = Json.asFloat(sliding.get("rope_theta"), local);
      }
      Map<String, Object> full = Json.asObject(ropeParams.get("full_attention"));
      if (full != null) {
        theta = Json.asFloat(full.get("rope_theta"), theta);
        partial = Json.asFloat(full.get("partial_rotary_factor"), partial);
      }
      return new RopeBases(theta, local, partial);
    }

    private static Gemma4Text parseGemma4Text(final Map<String, Object> m,
                                              final float partialRotary) {
      return new Gemma4Text(
        Json.asInt(m.get("hidden_size_per_layer_input"), 256),
        Json.asInt(m.get("num_kv_shared_layers"), 0),
        Json.asBoolean(m.get("use_double_wide_mlp"), false),
        Json.asInt(m.get("global_head_dim"), 512),
        partialRotary,
        Json.asFloat(m.get("final_logit_softcapping"), 0f),
        Json.asBoolean(m.get("enable_moe_block"), false));
    }

    /**
     * Attention softmax scale: {@code (queryPreAttnScalar or headDim)^-0.5} for most families.
     * Gemma 4 uses {@code 1.0} because Q/K RMSNorm already unit-RMS those tensors.
     *
     * @return scale applied to QK scores before softmax
     */
    public float attentionScale() {
      if (this.gemma4 != null) {
        return 1.0f;
      }
      float denom = this.queryPreAttnScalar > 0f ? this.queryPreAttnScalar : this.headDim;
      return (float) Math.pow(denom, -0.5);
    }

    /**
     * {@code true} when {@link #layerTypes()} names a linear-attention layer. Those hybrids
     * (Qwen3.5 / Fara-style Gated DeltaNet) are not supported; load fails via
     * {@link com.igormaznitsa.nanollvm.models.ModelSupport} before weights bind.
     *
     * @return whether any layer type contains {@code linear_attention}
     * @since 1.1.0
     */
    public boolean hasLinearAttentionLayers() {
      return this.layerTypes.stream()
        .anyMatch(
          type -> type != null && type.toLowerCase(Locale.ROOT).contains("linear_attention"));
    }

    /**
     * {@code true} when layer {@code layerIndex} is a short-convolution block (LFM2), not
     * attention. Out-of-range indexes are treated as attention.
     *
     * @param layerIndex zero-based transformer layer
     * @return whether this layer uses conv state instead of KV attention
     */
    public boolean isConvLayer(final int layerIndex) {
      if (this.layerTypes != null && layerIndex >= 0 && layerIndex < this.layerTypes.size()) {
        String type = this.layerTypes.get(layerIndex);
        return type != null && type.toLowerCase(Locale.ROOT).contains("conv")
          && !type.toLowerCase(Locale.ROOT).contains("attention");
      }
      return false;
    }

    /**
     * Inverse of {@link #isConvLayer(int)}: the layer runs attention (full or sliding) rather
     * than LFM2 short-convolution.
     *
     * @param layerIndex zero-based transformer layer
     * @return {@code true} when the layer is not a conv block
     */
    public boolean isFullAttentionLayer(final int layerIndex) {
      return !this.isConvLayer(layerIndex);
    }

    /**
     * MLP activation name used by the graph: {@link #hiddenActivation()} when non-blank, else
     * {@link #hiddenAct()}, else {@code silu}. Gemma often stores GELU-tanh under
     * {@code hidden_activation}.
     *
     * @return non-blank activation id
     */
    public String effectiveActivation() {
      if (this.hiddenActivation != null && !this.hiddenActivation.isBlank()) {
        return this.hiddenActivation;
      }
      return this.hiddenAct == null ? "silu" : this.hiddenAct;
    }

    /**
     * {@code true} when this blueprint includes {@link Gemma4Text} extras (QAT mobile text, PLE,
     * KV sharing). Other families leave {@link #gemma4()} {@code null}.
     *
     * @return whether Gemma 4 text fields were parsed
     * @since 1.1.0
     */
    public boolean isGemma4() {
      return this.gemma4 != null;
    }

    /**
     * Attention head dim at {@code layerIndex}. Non-Gemma-4 models use {@link #headDim()} for
     * every layer. Gemma 4 global (non-sliding) layers may use {@link Gemma4Text#globalHeadDim()}.
     *
     * @param layerIndex zero-based transformer layer
     * @return head dimension for that layer
     * @since 1.1.0
     */
    public int layerHeadDim(final int layerIndex) {
      if (this.gemma4 == null) {
        return this.headDim;
      }
      return this.isSlidingLayer(layerIndex) ? this.headDim : this.gemma4.globalHeadDim();
    }

    /**
     * MLP intermediate size at {@code layerIndex}. Gemma 4 shared-KV layers may double this when
     * {@link Gemma4Text#useDoubleWideMlp()} is set.
     *
     * @param layerIndex zero-based transformer layer
     * @return MLP expand width for that layer
     * @since 1.1.0
     */
    public int mlpIntermediateSize(final int layerIndex) {
      if (this.gemma4 == null) {
        return this.intermediateSize;
      }
      return this.gemma4.useDoubleWideMlp() && this.isKvSharedLayer(layerIndex)
        ? this.intermediateSize * 2
        : this.intermediateSize;
    }

    /**
     * First layer index that reuses KV from an earlier producer. Returns {@link #numHiddenLayers()}
     * when this family does not share KV (every layer owns its cache).
     *
     * @return first shared-KV layer, or layer count when none
     * @since 1.1.0
     */
    public int firstKvSharedLayer() {
      if (this.gemma4 == null || this.gemma4.numKvSharedLayers() <= 0) {
        return this.numHiddenLayers;
      }
      return this.numHiddenLayers - this.gemma4.numKvSharedLayers();
    }

    /**
     * {@code true} when layer {@code layerIndex} reuses KV from {@link #kvProducerLayer(int)}
     * instead of writing its own cache. Gemma 4 shares the last {@code numKvSharedLayers}.
     *
     * @param layerIndex zero-based transformer layer
     * @return whether this layer is a KV consumer
     * @since 1.1.0
     */
    public boolean isKvSharedLayer(final int layerIndex) {
      int first = this.firstKvSharedLayer();
      return first > 0 && layerIndex >= first;
    }

    /**
     * Layer that owns the KV cache for {@code layerIndex}. Returns {@code layerIndex} when the
     * layer is not shared. Shared Gemma 4 layers pick the last earlier layer of the same
     * sliding/full type.
     *
     * @param layerIndex zero-based transformer layer
     * @return producer layer index
     * @throws IllegalStateException if a shared layer has no matching producer
     * @since 1.1.0
     */
    public int kvProducerLayer(final int layerIndex) {
      if (!this.isKvSharedLayer(layerIndex)) {
        return layerIndex;
      }
      boolean sliding = this.isSlidingLayer(layerIndex);
      for (int i = this.firstKvSharedLayer() - 1; i >= 0; i--) {
        if (this.isSlidingLayer(i) == sliding) {
          return i;
        }
      }
      throw new IllegalStateException(
        "no KV producer layer of matching type for shared layer " + layerIndex);
    }

    private record RopeBases(float theta, float localBaseFreq, float partialRotaryFactor) {
    }

    /**
     * {@code true} when layer {@code layerIndex} uses a sliding attention window.
     *
     * <p>Reads {@link #layerTypes()} when present. If only {@link #slidingWindow()} is set, every
     * layer except every 6th is treated as sliding (Gemma-style fallback when {@code layer_types}
     * is absent).
     *
     * @param layerIndex zero-based transformer layer
     * @return whether this layer is local/sliding attention
     */
    public boolean isSlidingLayer(final int layerIndex) {
      if (this.layerTypes != null && layerIndex >= 0 && layerIndex < this.layerTypes.size()) {
        String type = this.layerTypes.get(layerIndex);
        return type != null && type.toLowerCase(Locale.ROOT).contains("sliding");
      }
      // Compat when sliding_window is set but layer_types is absent: full attention every 6th layer.
      if (this.slidingWindow > 0) {
        return (layerIndex + 1) % 6 != 0;
      }
      return false;
    }
  }

  /**
   * Gemma 4 text extras flattened from nested {@code text_config} (QAT mobile). Other families
   * leave {@link HfConfig#gemma4()} {@code null}. Vision/audio towers in the same checkpoint are
   * skipped at load; MoE ({@code enable_moe_block}) is unsupported.
   *
   * @param hiddenSizePerLayerInput  PLE / per-layer input width
   * @param numKvSharedLayers        how many trailing layers reuse earlier KV
   * @param useDoubleWideMlp         double MLP width on shared-KV layers
   * @param globalHeadDim            head dim for full-attention (non-sliding) layers
   * @param fullPartialRotaryFactor  RoPE fraction on full-attention layers
   * @param finalLogitSoftcapping    logit softcap; {@code 0} disables
   * @param enableMoeBlock           MoE flag from config (this engine does not run MoE)
   * @since 1.1.0
   */
  public record Gemma4Text(
    int hiddenSizePerLayerInput,
    int numKvSharedLayers,
    boolean useDoubleWideMlp,
    int globalHeadDim,
    float fullPartialRotaryFactor,
    float finalLogitSoftcapping,
    boolean enableMoeBlock
  ) {
  }
}
