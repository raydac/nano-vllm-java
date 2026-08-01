package com.igormaznitsa.nanollvm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.utils.Json;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public final class Config {

  private final Path model;
  private final int maxNumBatchedTokens;
  private final int maxNumSeqs;
  private final int maxModelLen;
  private final float gpuMemoryUtilization;
  private final int tensorParallelSize;
  private final boolean enforceEager;
  private final HfConfig hfConfig;
  private final int kvcacheBlockSize;
  private int eos;
  private List<Integer> stopTokenIds = List.of();
  private int numKvcacheBlocks;

  private Config(Builder b) {
    this.model = b.model;
    this.maxNumBatchedTokens = b.maxNumBatchedTokens;
    this.maxNumSeqs = b.maxNumSeqs;
    this.gpuMemoryUtilization = b.gpuMemoryUtilization;
    this.tensorParallelSize = b.tensorParallelSize;
    this.enforceEager = b.enforceEager;
    this.kvcacheBlockSize = b.kvcacheBlockSize;
    this.numKvcacheBlocks = b.numKvcacheBlocks;
    this.eos = b.eos;

    if (!Files.isDirectory(this.model)) {
      throw new IllegalArgumentException("model path is not a directory: " + this.model);
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

    try {
      this.hfConfig = HfConfig.load(this.model.resolve("config.json"));
    } catch (IOException e) {
      throw new IllegalArgumentException("failed to load config.json from " + this.model, e);
    }
    this.maxModelLen = Math.min(b.maxModelLen, this.hfConfig.maxPositionEmbeddings());
  }

  public static Builder builder(Path model) {
    return new Builder(model);
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

  public HfConfig hfConfig() {
    return this.hfConfig;
  }

  public int eos() {
    return this.eos;
  }

  public void setEos(int eos) {
    this.eos = eos;
    if (this.stopTokenIds.isEmpty()) {
      this.stopTokenIds = List.of(eos);
    }
  }

  public List<Integer> stopTokenIds() {
    return this.stopTokenIds;
  }

  public void setStopTokenIds(List<Integer> stopTokenIds) {
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

  public void setNumKvcacheBlocks(int numKvcacheBlocks) {
    this.numKvcacheBlocks = numKvcacheBlocks;
  }

  public static final class Builder {
    private final Path model;
    private int maxNumBatchedTokens = 16384;
    private int maxNumSeqs = 512;
    private int maxModelLen = 4096;
    private float gpuMemoryUtilization = 0.9f;
    private int tensorParallelSize = 1;
    private boolean enforceEager = true;
    private int eos = -1;
    private int kvcacheBlockSize = 256;
    private int numKvcacheBlocks = -1;

    private Builder(Path model) {
      this.model = requireNonNull(model, "model").toAbsolutePath().normalize();
    }

    public Builder maxNumBatchedTokens(int v) {
      this.maxNumBatchedTokens = v;
      return this;
    }

    public Builder maxNumSeqs(int v) {
      this.maxNumSeqs = v;
      return this;
    }

    public Builder maxModelLen(int v) {
      this.maxModelLen = v;
      return this;
    }

    public Builder gpuMemoryUtilization(float v) {
      this.gpuMemoryUtilization = v;
      return this;
    }

    public Builder tensorParallelSize(int v) {
      this.tensorParallelSize = v;
      return this;
    }

    public Builder enforceEager(boolean v) {
      this.enforceEager = v;
      return this;
    }

    public Builder eos(int v) {
      this.eos = v;
      return this;
    }

    public Builder kvcacheBlockSize(int v) {
      this.kvcacheBlockSize = v;
      return this;
    }

    public Builder numKvcacheBlocks(int v) {
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
      float queryPreAttnScalar
  ) {
    public static HfConfig load(Path configJson) throws IOException {
      Map<String, Object> m = Json.parseObject(Files.readString(configJson));
      int hiddenSize = Json.asInt(m.get("hidden_size"), 0);
      int numAttentionHeads = Json.asInt(m.get("num_attention_heads"), 0);
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
      boolean gemma = modelType != null && modelType.toLowerCase().contains("gemma");
      boolean tieDefault = gemma; // Gemma checkpoints often omit the flag but have no lm_head
      return new HfConfig(
          Json.asInt(m.get("vocab_size"), 0),
          hiddenSize,
          Json.asInt(m.get("intermediate_size"), 0),
          Json.asInt(m.get("num_hidden_layers"), 0),
          numAttentionHeads,
          Json.asInt(m.get("num_key_value_heads"), numAttentionHeads),
          headDim,
          Json.asInt(m.get("max_position_embeddings"), 32768),
          Json.asFloat(m.get("rms_norm_eps"), 1e-6f),
          hiddenAct,
          Json.asBoolean(m.get("tie_word_embeddings"), tieDefault),
          Json.asBoolean(m.get("attention_bias"), false),
          Json.asFloat(m.get("rope_theta"), 1_000_000f),
          Json.asObject(m.get("rope_scaling")),
          Json.asString(m.getOrDefault("torch_dtype", "float16")),
          modelType,
          architectures,
          hiddenActivation,
          Json.asInt(m.get("sliding_window"), 0),
          layerTypes,
          Json.asFloat(m.get("rope_local_base_freq"), 10_000f),
          queryPre
      );
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

    public boolean isSlidingLayer(int layerIndex) {
      if (this.layerTypes != null && layerIndex >= 0 && layerIndex < this.layerTypes.size()) {
        String type = this.layerTypes.get(layerIndex);
        return type != null && type.toLowerCase().contains("sliding");
      }
      // Gemma 3 default when layer_types absent: full attention every 6th layer (1-indexed).
      if (this.slidingWindow > 0) {
        return (layerIndex + 1) % 6 != 0;
      }
      return false;
    }
  }
}
