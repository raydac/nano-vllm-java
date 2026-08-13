package com.igormaznitsa.nanollvm.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.CausalLMFactory;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoderFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import org.junit.jupiter.api.Test;

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
    assertEquals(WeightNames.ARCH_LLAMA, CausalLMFactory.detect(parse("""
      {"model_type":"llama","architectures":["LlamaForCausalLM"]}
      """)));
  }

  @Test
  void detectsBertEmbeddingsWithoutTreatingRobertaAsBert() {
    Config.HfConfig bert = parse("""
      {"model_type":"bert","architectures":["BertModel"]}
      """);
    assertTrue(ModelSupport.isEmbedding(bert));
    assertEquals(WeightNames.ARCH_BERT, EmbeddingEncoderFactory.detect(bert));

    Config.HfConfig roberta = parse("""
      {"model_type":"roberta","architectures":["RobertaModel"]}
      """);
    assertFalse(ModelSupport.isEmbedding(roberta));
    UnsupportedModelException ex = assertThrows(
      UnsupportedModelException.class, () -> ModelSupport.resolve(roberta));
    assertTrue(ex.getMessage().contains("roberta"));
    assertTrue(ex.getMessage().contains("bert"));
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
    assertTrue(ex.getMessage().toLowerCase().contains("vision")
      || ex.getMessage().toLowerCase().contains("multimodal"));
  }

  @Test
  void rejectsGgufQwen3AndLlamaWithContainerHint() {
    UnsupportedModelException qwen = assertThrows(
      UnsupportedModelException.class, () -> ModelSupport.requireGguf("qwen3"));
    assertTrue(qwen.getMessage().contains("GGUF"));
    assertTrue(qwen.getMessage().contains("Hugging Face"));

    UnsupportedModelException llama = assertThrows(
      UnsupportedModelException.class, () -> ModelSupport.requireGguf("llama"));
    assertTrue(llama.getMessage().contains("llama"));
    assertTrue(llama.getMessage().contains("GGUF"));
  }

  @Test
  void acceptsGgufLfm2AndBert() {
    assertEquals(WeightNames.ARCH_LFM2, ModelSupport.requireGguf("lfm2").architectureId());
    assertTrue(ModelSupport.requireGguf("bert").isEmbedding());
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
}
