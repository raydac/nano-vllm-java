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
 * Immutable engine configuration built by {@link Builder}.
 *
 * <p>EOS / stop tokens and KV-block counts are sealed at {@link Builder#build()} time
 * (tokenizer stops + heap estimate when unset).
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
    long bytesPerBlock = 2L * hf.numHiddenLayers() * blockSize
      * hf.numKeyValueHeads() * hf.headDim() * Float.BYTES;
    int heapCap = (int) Math.max(32, (long) (free * kvHeapFraction) / Math.max(1, bytesPerBlock));
    int resolved = Math.min(estimated, heapCap);
    if (resolved <= 0) {
      throw new IllegalStateException("numKvcacheBlocks must be > 0");
    }
    return resolved;
  }

  public static Builder builder(final Path model) {
    return new Builder(model);
  }

  public static Builder builder(final LlmModel model) {
    requireNonNull(model, "model");
    return new Builder(model.path(), model.hfConfig());
  }

  public Path model() {
    return this.model;
  }

  public int maxNumBatchedTokens() {
    return this.maxNumBatchedTokens;
  }

  public int maxNumSeqs() {
    return this.maxNumSeqs;
  }

  public int maxModelLen() {
    return this.maxModelLen;
  }

  /**
   * Fraction of {@link Runtime#maxMemory()} used when auto-sizing KV blocks
   * ({@code numKvcacheBlocks} unset). Default {@code 0.25}.
   */
  public float kvHeapFraction() {
    return this.kvHeapFraction;
  }

  /**
   * CPU workers for dense matmul ({@code 1} = sequential).
   */
  public int cpuThreads() {
    return this.cpuThreads;
  }

  public HfConfig hfConfig() {
    return this.hfConfig;
  }

  public int eos() {
    return this.eos;
  }

  public List<Integer> stopTokenIds() {
    return this.stopTokenIds;
  }

  public int kvcacheBlockSize() {
    return this.kvcacheBlockSize;
  }

  public int numKvcacheBlocks() {
    return this.numKvcacheBlocks;
  }

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

    public Builder maxNumBatchedTokens(final int v) {
      this.maxNumBatchedTokens = v;
      return this;
    }

    public Builder maxNumSeqs(final int v) {
      this.maxNumSeqs = v;
      return this;
    }

    public Builder maxModelLen(final int v) {
      this.maxModelLen = v;
      return this;
    }

    public Builder kvHeapFraction(final float v) {
      this.kvHeapFraction = v;
      return this;
    }

    public Builder cpuThreads(final int v) {
      this.cpuThreads = v;
      return this;
    }

    public Builder eos(final int v) {
      this.eos = v;
      if (this.stopTokenIds.isEmpty()) {
        this.stopTokenIds = List.of(v);
      }
      return this;
    }

    /**
     * Seals EOS / stop ids from the model tokenizer. Tokenizer stop ids win when non-empty;
     * otherwise {@link Tokenizer#eosTokenId()} is used when EOS was not set explicitly.
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

    public Builder kvcacheBlockSize(final int v) {
      this.kvcacheBlockSize = v;
      return this;
    }

    public Builder numKvcacheBlocks(final int v) {
      this.numKvcacheBlocks = v;
      return this;
    }

    public Config build() {
      return new Config(this);
    }
  }

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
    int convLCache
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

    public static HfConfig load(final Path configJson) throws IOException {
      return parse(Files.readString(configJson));
    }

    /**
     * @since 1.1.0
     */
    public static HfConfig parse(final String configJson) {
      Map<String, Object> m = Json.parseObject(requireNonNull(configJson, "configJson"));
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
      List<Object> archArr = Json.asArray(m.get("architectures"));
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
      String modelType = Json.asString(m.get("model_type"));
      float rmsEps = m.containsKey("rms_norm_eps")
        ? Json.asFloat(m.get("rms_norm_eps"), 1e-6f)
        : Json.asFloat(m.get("norm_eps"), 1e-6f);
      float ropeTheta = Json.asFloat(m.get("rope_theta"), 1_000_000f);
      Map<String, Object> ropeScaling = Json.asObject(m.get("rope_scaling"));
      Map<String, Object> ropeParams = Json.asObject(m.get("rope_parameters"));
      if (ropeParams != null && ropeParams.containsKey("rope_theta")) {
        ropeTheta = Json.asFloat(ropeParams.get("rope_theta"), ropeTheta);
      }
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
        Json.asBoolean(m.get("tie_word_embeddings"), false),
        Json.asBoolean(m.get("attention_bias"), false),
        ropeTheta,
        ropeScaling,
        Json.asString(m.getOrDefault("torch_dtype", "float16")),
        modelType,
        architectures,
        hiddenActivation,
        Json.asInt(m.get("sliding_window"), 0),
        layerTypes,
        Json.asFloat(m.get("rope_local_base_freq"), 10_000f),
        queryPre,
        Json.asInt(m.get("conv_L_cache"), 0)
      );
    }

    public boolean isConvLayer(final int layerIndex) {
      if (this.layerTypes != null && layerIndex >= 0 && layerIndex < this.layerTypes.size()) {
        String type = this.layerTypes.get(layerIndex);
        return type != null && type.toLowerCase(Locale.ROOT).contains("conv")
          && !type.toLowerCase(Locale.ROOT).contains("attention");
      }
      return false;
    }

    public boolean isFullAttentionLayer(final int layerIndex) {
      return !this.isConvLayer(layerIndex);
    }

    public String effectiveActivation() {
      if (this.hiddenActivation != null && !this.hiddenActivation.isBlank()) {
        return this.hiddenActivation;
      }
      return this.hiddenAct == null ? "silu" : this.hiddenAct;
    }

    public float attentionScale() {
      float denom = this.queryPreAttnScalar > 0f ? this.queryPreAttnScalar : this.headDim;
      return (float) Math.pow(denom, -0.5);
    }

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
}
