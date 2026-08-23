package com.igormaznitsa.nanollvm.models;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Locale.ROOT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.CausalLMFactory;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoderFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelSupportTest {

  private static void assertRejected(final String modelType, final String expectedSnippet) {
    UnsupportedModelException ex = assertThrows(
      UnsupportedModelException.class,
      () -> ModelSupport.resolve(parse("{\"model_type\":\"" + modelType + "\"}")));
    assertTrue(ex.getMessage().contains(expectedSnippet), ex.getMessage());
    assertTrue(ex.getMessage().contains("Supported by this library"));
  }

  private static Config.HfConfig parse(final String json) {
    return Config.HfConfig.parse(json);
  }

  @Test
  void detectsSupportedChatFamilies() {
    assertEquals(WeightNames.ARCH_QWEN3, CausalLMFactory.detect(parse("""
      {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}
      """)));
    assertEquals(WeightNames.ARCH_GEMMA3, CausalLMFactory.detect(parse("""
      {"model_type":"gemma3_text","architectures":["Gemma3ForCausalLM"]}
      """)));
    assertEquals(WeightNames.ARCH_GEMMA4, CausalLMFactory.detect(parse("""
      {"model_type":"gemma4","architectures":["Gemma4ForCausalLM"]}
      """)));
  }

  private static byte[] tinyGguf(final String architecture) {
    ByteBuffer buf = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(0x46554747);
    buf.putInt(3);
    buf.putLong(0);
    buf.putLong(1);
    putGgufString(buf, "general.architecture");
    buf.putInt(8);
    putGgufString(buf, architecture);
    buf.flip();
    byte[] bytes = new byte[buf.remaining()];
    buf.get(bytes);
    return bytes;
  }

  private static void putGgufString(final ByteBuffer buf, final String value) {
    byte[] bytes = value.getBytes(UTF_8);
    buf.putLong(bytes.length);
    buf.put(bytes);
  }

  @Test
  void rejectsQwen35AndDoesNotCallItQwen3() {
    Config.HfConfig fara = parse("""
      {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 1,
        "vision_config": {"hidden_size": 1024},
        "text_config": {"hidden_size": 2560, "layer_types": ["linear_attention"]}
      }
      """);
    UnsupportedModelException ex = assertThrows(
      UnsupportedModelException.class, () -> CausalLMFactory.detect(fara));
    assertEquals("qwen3_5", ex.modelType());
    assertTrue(ex.getMessage().contains("Qwen3.5"));
    assertFalse(ex.getMessage().contains("unsupported chat architecture after detect"));
    assertTrue(ex.getMessage().contains("qwen3"));
    assertTrue(ex.getMessage().contains(ModelSupport.CATALOG.split("\n", 2)[0]));
  }

  @Test
  void rejectsQwen2Gemma2MistralAndPhi() {
    assertRejected("qwen2", "Qwen2");
    assertRejected("qwen2_5", "Qwen2");
    assertRejected("gemma2", "Gemma 2");
    assertRejected("mistral", "Mistral");
    assertRejected("phi3", "Phi");
  }

  @Test
  void acceptsGemma4QatMobileTransformersAsTextChat() {
    Config.HfConfig gemma4 = parse("""
      {
        "model_type": "gemma4",
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_config": {"model_type": "gemma4_vision", "hidden_size": 768},
        "audio_config": {"model_type": "gemma4_audio", "hidden_size": 1024},
        "text_config": {
          "model_type": "gemma4_text",
          "hidden_size": 1536,
          "num_hidden_layers": 35,
          "num_attention_heads": 8,
          "num_key_value_heads": 1,
          "head_dim": 256,
          "global_head_dim": 512,
          "hidden_size_per_layer_input": 256,
          "use_double_wide_mlp": true,
          "num_kv_shared_layers": 20,
          "intermediate_size": 6144,
          "sliding_window": 512,
          "layer_types": [
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
            "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
          ]
        }
      }
      """);
    assertEquals(WeightNames.ARCH_GEMMA4, ModelSupport.resolve(gemma4).architectureId());
    assertEquals(WeightNames.ARCH_GEMMA4, CausalLMFactory.detect(gemma4));
    assertEquals("gemma4", gemma4.modelType());
    assertEquals(1536, gemma4.hiddenSize());
    assertEquals(35, gemma4.numHiddenLayers());
    assertEquals(256, gemma4.headDim());
    assertTrue(gemma4.isGemma4());
    assertTrue(gemma4.visionConfigPresent());
    assertTrue(gemma4.imageConfigPresent());
    assertTrue(gemma4.audioConfigPresent());
    assertTrue(gemma4.videoConfigPresent());
    assertTrue(gemma4.nestedTextConfig());
    assertEquals(15, gemma4.firstKvSharedLayer());
    assertFalse(gemma4.isKvSharedLayer(14));
    assertTrue(gemma4.isKvSharedLayer(15));
    assertEquals(13, gemma4.kvProducerLayer(15));
    assertEquals(14, gemma4.kvProducerLayer(19));
    assertEquals(256, gemma4.layerHeadDim(0));
    assertEquals(512, gemma4.layerHeadDim(4));
    assertEquals(6144, gemma4.mlpIntermediateSize(0));
    assertEquals(12288, gemma4.mlpIntermediateSize(15));
    assertEquals(1.0f, gemma4.attentionScale());
  }

  @Test
  void rejectsVisionGemma3EvenWhenModelTypeLooksSupported() {
    Config.HfConfig vlm = parse("""
      {
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
        "vision_config": {"hidden_size": 64}
      }
      """);
    UnsupportedModelException ex = assertThrows(
      UnsupportedModelException.class, () -> ModelSupport.resolve(vlm));
    assertTrue(ex.getMessage().toLowerCase(ROOT).contains("vision")
      || ex.getMessage().toLowerCase(ROOT).contains("multimodal"));
  }

  @Test
  void acceptsGgufQwen3AndStillRejectsLlamaGguf() {
    assertEquals(WeightNames.ARCH_QWEN3, ModelSupport.requireGguf("qwen3").architectureId());

    UnsupportedModelException llama = assertThrows(
      UnsupportedModelException.class, () -> ModelSupport.requireGguf("llama"));
    assertTrue(llama.getMessage().contains("llama"));
    assertTrue(llama.getMessage().contains("GGUF"));
  }

  @Test
  void detectsBertEncoderFamilyFromConfig() {
    Config.HfConfig bert = parse("""
      {"model_type":"bert","architectures":["BertModel"]}
      """);
    assertTrue(ModelSupport.isEmbedding(bert));
    assertEquals(WeightNames.ARCH_BERT, EmbeddingEncoderFactory.detect(bert));

    Config.HfConfig roberta = parse("""
      {"model_type":"roberta","architectures":["RobertaModel"]}
      """);
    assertTrue(ModelSupport.isEmbedding(roberta));
    assertEquals(WeightNames.ARCH_BERT, EmbeddingEncoderFactory.detect(roberta));
    assertTrue(ModelSupport.require(roberta, ModelSupport.Source.ONNX).isEmbedding());
  }

  @Test
  void detectsXlmRobertaOnnxAsBertEmbeddings() {
    Config.HfConfig xlm = parse("""
      {"model_type":"xlm-roberta","architectures":["XLMRobertaModel"]}
      """);
    assertTrue(ModelSupport.isEmbedding(xlm));
    assertEquals(WeightNames.ARCH_BERT, EmbeddingEncoderFactory.detect(xlm));
    assertTrue(ModelSupport.require(xlm, ModelSupport.Source.ONNX).isEmbedding());

    UnsupportedModelException safetensors = assertThrows(
      UnsupportedModelException.class,
      () -> ModelSupport.require(xlm, ModelSupport.Source.HF_SAFETENSORS));
    assertTrue(safetensors.getMessage().contains("safetensors"));

    Config.HfConfig masked = parse("""
      {"model_type":"xlm-roberta","architectures":["XLMRobertaForMaskedLM"]}
      """);
    assertTrue(ModelSupport.isEmbedding(masked));
    assertEquals(WeightNames.ARCH_BERT, EmbeddingEncoderFactory.detect(masked));
  }

  @Test
  void rejectsHfLfm2AndBertSafetensors() {
    Config.HfConfig lfm2 =
      parse("{\"model_type\":\"lfm2\",\"architectures\":[\"Lfm2ForCausalLM\"]}");
    UnsupportedModelException lfmEx = assertThrows(
      UnsupportedModelException.class,
      () -> ModelSupport.require(lfm2, ModelSupport.Source.HF_SAFETENSORS));
    assertTrue(lfmEx.getMessage().contains("GGUF"));

    Config.HfConfig bert = parse("{\"model_type\":\"bert\",\"architectures\":[\"BertModel\"]}");
    UnsupportedModelException bertEx = assertThrows(
      UnsupportedModelException.class,
      () -> ModelSupport.require(bert, ModelSupport.Source.HF_SAFETENSORS));
    assertTrue(bertEx.getMessage().contains("safetensors"));
    assertTrue(ModelSupport.require(bert, ModelSupport.Source.ONNX).isEmbedding());
  }

  @Test
  void unknownArchitectureListsTheCatalog() {
    UnsupportedModelException ex = assertThrows(
      UnsupportedModelException.class,
      () -> ModelSupport.resolve(parse("{\"model_type\":\"totally_unknown_arch\"}")));
    assertTrue(ex.getMessage().contains("totally_unknown_arch"));
    assertTrue(ex.getMessage().contains("Supported by this library"));
  }

  @Test
  void acceptsGgufLfm2AndBert() {
    assertEquals(WeightNames.ARCH_LFM2, ModelSupport.requireGguf("lfm2").architectureId());
    assertTrue(ModelSupport.requireGguf("bert").isEmbedding());
    assertTrue(ModelSupport.requireGguf("roberta").isEmbedding());
    assertTrue(ModelSupport.requireGguf("xlm-roberta").isEmbedding());
  }

  @Test
  void rejectsNonBertEmbeddingLookalikes() {
    assertRejected("deberta", "not implemented");
    assertRejected("distilbert", "not implemented");
    assertRejected("albert", "not implemented");
    assertRejected("electra", "not implemented");
    assertRejected("modernbert", "not implemented");
  }

  @Test
  void isEmbeddingCheckpointReadsConfigWithoutLoadingWeights(@TempDir final Path dir)
    throws IOException {
    Path config = dir.resolve("config.json");
    Files.writeString(config, """
      {"model_type":"roberta","architectures":["RobertaModel"]}
      """, UTF_8);
    assertTrue(ModelSupport.isEmbeddingCheckpoint(dir));
    assertTrue(ModelSupport.isEmbeddingCheckpoint(config));

    Files.writeString(config, """
      {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}
      """, UTF_8);
    assertFalse(ModelSupport.isEmbeddingCheckpoint(dir));

    Files.writeString(config, """
      {"model_type":"deberta","architectures":["DebertaModel"]}
      """, UTF_8);
    assertFalse(ModelSupport.isEmbeddingCheckpoint(dir));
  }

  @Test
  void isEmbeddingCheckpointPeeksGgufMetadata(@TempDir final Path dir) throws IOException {
    Path bert = dir.resolve("encoder.gguf");
    Files.write(bert, tinyGguf("bert"));
    assertEquals("bert", GgufReader.peekArchitecture(bert));
    assertTrue(ModelSupport.isEmbeddingCheckpoint(bert));

    Path chat = dir.resolve("chat.gguf");
    Files.write(chat, tinyGguf("qwen3"));
    assertEquals("qwen3", GgufReader.peekArchitecture(chat));
    assertFalse(ModelSupport.isEmbeddingCheckpoint(chat));
  }
}
