package com.igormaznitsa.nanollvm.llm;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.CONFIG_JSON;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.Model;
import com.igormaznitsa.nanollvm.utils.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class Config {

  private final Path model;
  private final int maxNumBatchedTokens;
  private final int maxNumSeqs;
  private final int maxModelLen;
  private final float gpuMemoryUtilization;
  private final int tensorParallelSize;
  private final boolean enforceEager;
  private final int cpuThreads;
  private final HfConfig hfConfig;
  private final int kvcacheBlockSize;
  private int eos;
  private List<Integer> stopTokenIds = List.of();
  private int numKvcacheBlocks;

  private Config(final Builder b) {
    this.model = b.model;
    this.maxNumBatchedTokens = b.maxNumBatchedTokens;
    this.maxNumSeqs = b.maxNumSeqs;
    this.gpuMemoryUtilization = b.gpuMemoryUtilization;
    this.tensorParallelSize = b.tensorParallelSize;
    this.enforceEager = b.enforceEager;
    this.cpuThreads = b.cpuThreads;
    this.kvcacheBlockSize = b.kvcacheBlockSize;
    this.numKvcacheBlocks = b.numKvcacheBlocks;
    this.eos = b.eos;

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
    if (this.tensorParallelSize < 1 || this.tensorParallelSize > 8) {
      throw new IllegalArgumentException("tensorParallelSize must be in [1,8]");
    }
    if (this.tensorParallelSize != 1) {
      throw new IllegalArgumentException("Java port currently supports tensorParallelSize=1 only");
    }
    if (this.cpuThreads < 1) {
      throw new IllegalArgumentException("cpuThreads must be >= 1, got " + this.cpuThreads);
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
  }

  public static Builder builder(final Path model) {
    return new Builder(model);
  }

  public static Builder builder(final Model model) {
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

  public float gpuMemoryUtilization() {
    return this.gpuMemoryUtilization;
  }

  public int tensorParallelSize() {
    return this.tensorParallelSize;
  }

  public boolean enforceEager() {
    return this.enforceEager;
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

  public void setEos(final int eos) {
    this.eos = eos;
    if (this.stopTokenIds.isEmpty()) {
      this.stopTokenIds = List.of(eos);
    }
  }

  public List<Integer> stopTokenIds() {
    return this.stopTokenIds;
  }

  public void setStopTokenIds(final List<Integer> stopTokenIds) {
    this.stopTokenIds = List.copyOf(stopTokenIds);
    if (!stopTokenIds.isEmpty()) {
      this.eos = stopTokenIds.getFirst();
    }
  }

  public int kvcacheBlockSize() {
    return this.kvcacheBlockSize;
  }

  public int numKvcacheBlocks() {
    return this.numKvcacheBlocks;
  }

  public void setNumKvcacheBlocks(final int numKvcacheBlocks) {
    this.numKvcacheBlocks = numKvcacheBlocks;
  }

  public static final class Builder {
    private final Path model;
    private final HfConfig hfConfig;
    private int maxNumBatchedTokens = 16384;
    private int maxNumSeqs = 512;
    private int maxModelLen = 4096;
    private float gpuMemoryUtilization = 0.9f;
    private int tensorParallelSize = 1;
    private boolean enforceEager = true;
    private int cpuThreads = 1;
    private int eos = -1;
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

    public Builder gpuMemoryUtilization(final float v) {
      this.gpuMemoryUtilization = v;
      return this;
    }

    public Builder tensorParallelSize(final int v) {
      this.tensorParallelSize = v;
      return this;
    }

    public Builder enforceEager(final boolean v) {
      this.enforceEager = v;
      return this;
    }

    public Builder cpuThreads(final int v) {
      this.cpuThreads = v;
      return this;
    }

    public Builder eos(final int v) {
      this.eos = v;
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
    public static HfConfig load(final Path configJson) throws IOException {
      Map<String, Object> m = Json.parseObject(Files.readString(configJson));
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
        Json.asBoolean(m.get("tie_word_embeddings"),
          modelType != null && modelType.toLowerCase(Locale.ROOT).contains("gemma")),
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
      // Gemma 3 default when layer_types absent: full attention every 6th layer (1-indexed).
      if (this.slidingWindow > 0) {
        return (layerIndex + 1) % 6 != 0;
      }
      return false;
    }
  }
}
