package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.layers.GroupedEncoderAttention;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.llmarch.ModelLoader;
import com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsTransport;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Gemma3EmbeddingTest {

  private static Tensor matrix(final String name, final Config.HfConfig config) {
    if (name.equals(WeightNames.EMBED_TOKENS)) {
      return filled(name, config.vocabSize(), config.hiddenSize());
    }
    if (name.endsWith("q_norm.weight") || name.endsWith("k_norm.weight")) {
      return filled(name, config.headDim());
    }
    if (name.endsWith("layernorm.weight") || name.equals(WeightNames.MODEL_NORM)) {
      return filled(name, config.hiddenSize());
    }
    if (name.endsWith(WeightNames.O_PROJ_WEIGHT)) {
      return filled(name, config.hiddenSize(), config.numAttentionHeads() * config.headDim());
    }
    if (name.endsWith(WeightNames.DOWN_PROJ_WEIGHT)) {
      return filled(name, config.hiddenSize(), config.intermediateSize());
    }
    throw new IllegalStateException(name);
  }

  private static Tensor filled(final String name, final int... shape) {
    int count = 1;
    for (int axis : shape) {
      count *= axis;
    }
    float[] data = new float[count];
    int salt = name.hashCode();
    for (int i = 0; i < count; i++) {
      data[i] = ((Math.floorMod(salt + i, 5)) - 2) * 0.02f;
    }
    return Tensor.of(data, shape);
  }

  private static double l2(final float[] vector) {
    double sum = 0.0;
    for (float value : vector) {
      sum += (double) value * value;
    }
    return Math.sqrt(sum);
  }

  private static float cosine(final float[] left, final float[] right) {
    float dot = 0f;
    for (int i = 0; i < left.length; i++) {
      dot += left[i] * right[i];
    }
    return dot;
  }

  private static void writeSafetensors(final Path file, final String tensorName)
    throws IOException {
    Files.createDirectories(file.getParent());
    String header = "{\"%s\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}"
      .formatted(tensorName);
    byte[] json = header.getBytes(StandardCharsets.UTF_8);
    ByteBuffer body = ByteBuffer.allocate(8 + json.length + 4).order(ByteOrder.LITTLE_ENDIAN);
    body.putLong(json.length);
    body.put(json);
    body.putFloat(0.5f);
    Files.write(file, body.array());
  }

  @Test
  void bidirectionalGemma3TextIsAnEmbedding() {
    Config.HfConfig embedding = Config.HfConfig.parse("""
      {"model_type":"gemma3_text","architectures":["Gemma3TextModel"],
       "use_bidirectional_attention":true,"sliding_window":512}
      """);
    assertTrue(embedding.bidirectionalAttention());
    assertEquals(257, embedding.bidirectionalWindow());
    assertTrue(ModelSupport.isEmbedding(embedding));
    assertEquals(
      WeightNames.ARCH_GEMMA3_EMBEDDING,
      ModelSupport.resolve(embedding).architectureId());
    assertThrows(
      UnsupportedModelException.class,
      () -> ModelSupport.require(embedding, ModelSupport.Source.GGUF));

    Config.HfConfig chat = Config.HfConfig.parse("""
      {"model_type":"gemma3_text","architectures":["Gemma3ForCausalLM"],"sliding_window":512}
      """);
    assertFalse(chat.bidirectionalAttention());
    assertEquals(0, chat.bidirectionalWindow());
    assertFalse(ModelSupport.isEmbedding(chat));
    assertEquals(WeightNames.ARCH_GEMMA3, ModelSupport.resolve(chat).architectureId());

    assertThrows(UnsupportedModelException.class,
      () -> ModelSupport.resolve(Config.HfConfig.parse("""
        {"model_type":"gemma3_text","architectures":["Gemma3TextModel"]}
        """)));
  }

  @Test
  void slidingWindowIgnoresKeysOutsideTheExclusiveDistance() {
    GroupedEncoderAttention local = new GroupedEncoderAttention(1, 1, 1, 1f, 1);
    GroupedEncoderAttention full = new GroupedEncoderAttention(1, 1, 1, 1f, 0);
    Tensor query = Tensor.of(new float[] {1f, 1f, 1f}, 3, 1, 1);
    Tensor key = Tensor.of(new float[] {1f, 1f, 1f}, 3, 1, 1);
    Tensor value = Tensor.of(new float[] {1f, 10f, 100f}, 3, 1, 1);

    Tensor windowed = local.forward(query, key, value, null);
    Tensor opened = full.forward(query, key, value, null);

    assertEquals(1f, windowed.data()[0], 1e-5f);
    assertEquals(100f, windowed.data()[2], 1e-5f);
    assertEquals(111f / 3f, opened.data()[0], 1e-4f);
  }

  @Test
  void encoderMeanPoolsAndNormalizesTheDenseProjection() {
    Config.HfConfig config = Config.HfConfig.parse("""
      {"model_type":"gemma3_text","architectures":["Gemma3TextModel"],
       "use_bidirectional_attention":true,"hidden_size":32,"intermediate_size":16,
       "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,
       "vocab_size":16,"max_position_embeddings":32,"rms_norm_eps":1e-6,
       "hidden_activation":"gelu_pytorch_tanh","sliding_window":4,"rope_theta":10000,
       "rope_local_base_freq":10000,
       "layer_types":["sliding_attention","full_attention"]}
      """);
    WeightSchema schema = WeightSchema.gemma3Embedding(config);
    Map<String, Tensor> named = new LinkedHashMap<>();
    int qRows = config.numAttentionHeads() * config.headDim();
    int kvRows = config.numKeyValueHeads() * config.headDim();
    for (String expected : schema.expectedParameters()) {
      if (expected.endsWith(WeightNames.QKV_PROJ_WEIGHT)) {
        String prefix =
          expected.substring(0, expected.length() - WeightNames.QKV_PROJ_WEIGHT.length());
        named.put(prefix + "q_proj.weight", filled(expected + "q", qRows, config.hiddenSize()));
        named.put(prefix + "k_proj.weight", filled(expected + "k", kvRows, config.hiddenSize()));
        named.put(prefix + "v_proj.weight", filled(expected + "v", kvRows, config.hiddenSize()));
      } else if (expected.endsWith(WeightNames.GATE_UP_PROJ_WEIGHT)) {
        String prefix = expected.substring(
          0, expected.length() - WeightNames.GATE_UP_PROJ_WEIGHT.length());
        named.put(prefix + "gate_proj.weight",
          filled(expected + "g", config.intermediateSize(), config.hiddenSize()));
        named.put(prefix + "up_proj.weight",
          filled(expected + "u", config.intermediateSize(), config.hiddenSize()));
      } else if (expected.equals(WeightNames.EMBEDDING_UP_PROJ)) {
        named.put("2_Dense/linear.weight", filled(expected, 48, config.hiddenSize()));
      } else if (expected.equals(WeightNames.EMBEDDING_DOWN_PROJ)) {
        named.put("3_Dense/linear.weight", filled(expected, 24, 48));
      } else {
        named.put(expected, matrix(expected, config));
      }
    }

    WeightBag weights = ModelLoader.assembleFromNamedTensors(
      named, "synthetic", config, schema, null);
    Gemma3ForEmbedding encoder = new Gemma3ForEmbedding(config, weights);
    assertEquals(WeightNames.ARCH_GEMMA3_EMBEDDING, encoder.architectureName());
    assertEquals(24, encoder.embeddingDim());

    float[] first = encoder.encode(new int[] {1, 2, 3, 4}, MatmulRuntime.sequential());
    float[] second = encoder.encode(new int[] {1, 2, 3, 4}, MatmulRuntime.sequential());
    assertEquals(24, first.length);
    assertEquals(1.0, l2(first), 1e-4);
    assertEquals(1f, cosine(first, second), 1e-5f);
    assertTrue(
      cosine(first, encoder.encode(new int[] {5, 6, 7, 8}, MatmulRuntime.sequential())) < 0.999f);
  }

  @Test
  void nestedDenseSafetensorsKeepDistinctNames(@TempDir final Path dir) throws IOException {
    writeSafetensors(dir.resolve("model.safetensors"), "model.norm.weight");
    writeSafetensors(dir.resolve("2_Dense").resolve("model.safetensors"), "linear.weight");
    writeSafetensors(dir.resolve("3_Dense").resolve("model.safetensors"), "linear.weight");

    try (SafetensorsTransport transport = SafetensorsTransport.open(dir,
      "{\"model_type\":\"gemma3_text\"}")) {
      assertTrue(transport.tensorIndex().stream().anyMatch(ref ->
        "model.norm.weight".equals(ref.name()) && "model.norm.weight".equals(ref.storageName())));
      assertTrue(transport.tensorIndex().stream().anyMatch(ref ->
        "2_Dense/linear.weight".equals(ref.name()) && "linear.weight".equals(ref.storageName())));
      assertTrue(transport.tensorIndex().stream().anyMatch(ref ->
        "3_Dense/linear.weight".equals(ref.name()) && "linear.weight".equals(ref.storageName())));
    }
  }
}
