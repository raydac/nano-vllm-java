package com.igormaznitsa.nanollvm.models.llmarch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

final class ModelBindingTest {

  private static ContainerCatalog qwen3Catalog(final int layers, final boolean withHeadDim) {
    Map<String, Object> meta = new LinkedHashMap<>();
    meta.put("general.architecture", "qwen3");
    meta.put("qwen3.embedding_length", 1024);
    meta.put("qwen3.block_count", layers);
    meta.put("qwen3.feed_forward_length", 3072);
    meta.put("qwen3.attention.head_count", 16);
    meta.put("qwen3.attention.head_count_kv", 8);
    if (withHeadDim) {
      meta.put("qwen3.attention.key_length", 128);
    }
    meta.put("qwen3.context_length", 4096);
    meta.put("qwen3.rope.freq_base", 1_000_000f);
    meta.put("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    meta.put("tokenizer.ggml.tokens", IntStream.range(0, 32).mapToObj(i -> "t" + i).toList());

    Set<String> tensors = new LinkedHashSet<>();
    tensors.add("token_embd.weight");
    tensors.add("output_norm.weight");
    for (int i = 0; i < layers; i++) {
      String blk = "blk." + i + ".";
      tensors.addAll(List.of(
        blk + "attn_norm.weight",
        blk + "ffn_norm.weight",
        blk + "attn_q.weight",
        blk + "attn_k.weight",
        blk + "attn_v.weight",
        blk + "attn_output.weight",
        blk + "attn_q_norm.weight",
        blk + "attn_k_norm.weight",
        blk + "ffn_gate.weight",
        blk + "ffn_up.weight",
        blk + "ffn_down.weight"));
    }
    return new ContainerCatalog(ModelSupport.Source.GGUF, "qwen3.gguf", "qwen3", meta, tensors);
  }

  private static ContainerCatalog qwen35Catalog() {
    Map<String, Object> meta = new LinkedHashMap<>();
    meta.put("general.architecture", "qwen35");
    meta.put("qwen35.embedding_length", 4096);
    meta.put("qwen35.block_count", 2);
    meta.put("qwen35.feed_forward_length", 12288);
    meta.put("qwen35.attention.head_count", 16);
    meta.put("qwen35.attention.head_count_kv", 4);
    meta.put("qwen35.attention.key_length", 256);
    meta.put("qwen35.context_length", 262144);
    meta.put("qwen35.rope.freq_base", 10_000_000f);
    meta.put("qwen35.rope.dimension_count", 64);
    meta.put("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    meta.put("qwen35.ssm.conv_kernel", 4);
    meta.put("qwen35.ssm.state_size", 128);
    meta.put("qwen35.ssm.group_count", 16);
    meta.put("qwen35.ssm.time_step_rank", 32);
    meta.put("qwen35.ssm.inner_size", 4096);
    meta.put("tokenizer.ggml.tokens", IntStream.range(0, 32).mapToObj(i -> "t" + i).toList());

    Set<String> tensors = new LinkedHashSet<>();
    tensors.add("token_embd.weight");
    tensors.add("output_norm.weight");
    tensors.add("output.weight");
    tensors.addAll(List.of(
      "blk.0.attn_norm.weight",
      "blk.0.post_attention_norm.weight",
      "blk.0.ffn_gate.weight",
      "blk.0.ffn_up.weight",
      "blk.0.ffn_down.weight",
      "blk.0.attn_qkv.weight",
      "blk.0.attn_gate.weight",
      "blk.0.ssm_beta.weight",
      "blk.0.ssm_alpha.weight",
      "blk.0.ssm_conv1d.weight",
      "blk.0.ssm_a",
      "blk.0.ssm_dt.bias",
      "blk.0.ssm_norm.weight",
      "blk.0.ssm_out.weight",
      "blk.1.attn_norm.weight",
      "blk.1.post_attention_norm.weight",
      "blk.1.ffn_gate.weight",
      "blk.1.ffn_up.weight",
      "blk.1.ffn_down.weight",
      "blk.1.attn_q.weight",
      "blk.1.attn_k.weight",
      "blk.1.attn_v.weight",
      "blk.1.attn_output.weight",
      "blk.1.attn_q_norm.weight",
      "blk.1.attn_k_norm.weight"));
    return new ContainerCatalog(ModelSupport.Source.GGUF, "qwen35.gguf", "qwen35", meta, tensors);
  }

  @Test
  void bindsQwen3GgufFromCatalogWithoutLoadingPayloads() {
    ContainerCatalog catalog = qwen3Catalog(2, true);
    ModelBinding.BoundModel bound = ModelBinding.bind(catalog);
    assertEquals(WeightNames.ARCH_QWEN3, bound.selection().architectureId());
    assertEquals(128, bound.config().headDim());
    assertEquals(1024, bound.config().hiddenSize());
    assertEquals(16, bound.config().numAttentionHeads());
    assertEquals(8, bound.config().numKeyValueHeads());
    assertTrue(bound.config().tieWordEmbeddings());
    assertTrue(bound.schema().expects("token_embd.weight"));
    assertTrue(bound.schema().expects("output_norm.weight"));
    assertTrue(bound.schema().expects("blk.0.attn_q_norm.weight"));
    assertEquals(WeightNames.ARCH_QWEN3, bound.processor().architectureId());
    assertFalse(bound.processor().isEmbedding());
  }

  @Test
  void bindsQwen35GgufFromCatalogWithoutLoadingPayloads() {
    ContainerCatalog catalog = qwen35Catalog();
    ModelBinding.BoundModel bound = ModelBinding.bind(catalog);
    assertEquals(WeightNames.ARCH_QWEN3_5, bound.selection().architectureId());
    assertEquals(256, bound.config().headDim());
    assertEquals(4096, bound.config().hiddenSize());
    assertEquals(16, bound.config().numAttentionHeads());
    assertEquals(4, bound.config().numKeyValueHeads());
    assertFalse(bound.config().tieWordEmbeddings());
    assertTrue(bound.config().isLinearAttentionLayer(0));
    assertFalse(bound.config().isLinearAttentionLayer(1));
    assertEquals(16, bound.config().qwen35().linearNumKeyHeads());
    assertEquals(32, bound.config().qwen35().linearNumValueHeads());
    assertEquals(128, bound.config().qwen35().linearKeyHeadDim());
    assertEquals(128, bound.config().qwen35().linearValueHeadDim());
    assertEquals(0.25f, bound.config().qwen35().partialRotaryFactor());
    assertTrue(bound.schema().expects("blk.0.attn_qkv.weight"));
    assertTrue(bound.schema().expects("blk.0.ssm_a"));
    assertTrue(bound.schema().expects("blk.1.attn_q.weight"));
    assertFalse(bound.schema().expects("blk.0.attn_q.weight"));
    assertEquals(WeightNames.ARCH_QWEN3_5, bound.processor().architectureId());
  }

  @Test
  void qwen3GgufRequiresExplicitHeadDim() {
    ContainerCatalog catalog = qwen3Catalog(1, false);
    IllegalStateException ex = assertThrows(
      IllegalStateException.class, () -> ModelBinding.bindGguf(catalog));
    assertTrue(ex.getMessage().contains("head_dim"), ex.getMessage());
  }

  @Test
  void stillRejectsLlamaGgufAtBinding() {
    ContainerCatalog catalog = new ContainerCatalog(
      ModelSupport.Source.GGUF,
      "llama.gguf",
      "llama",
      Map.of(),
      Set.of("token_embd.weight"));
    UnsupportedModelException ex = assertThrows(
      UnsupportedModelException.class, () -> ModelBinding.bind(catalog));
    assertTrue(ex.getMessage().contains("llama"));
  }

  @Test
  void bindsHfSafetensorsCatalogFromConfigJson() {
    String configJson = """
      {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": true
      }
      """;
    ContainerCatalog catalog = ContainerCatalog.ofHf(
      ModelSupport.Source.HF_SAFETENSORS,
      "qwen3-folder",
      configJson,
      Set.of("model.embed_tokens.weight"));
    ModelBinding.BoundModel bound = ModelBinding.bind(catalog);
    assertEquals(WeightNames.ARCH_QWEN3, bound.selection().architectureId());
    assertEquals(16, bound.config().hiddenSize());
    assertTrue(bound.schema().expects("model.embed_tokens.weight"));
    assertEquals(WeightNames.ARCH_QWEN3, bound.processor().architectureId());
  }

  @Test
  void bindsOnnxCatalogFromConfigJson() {
    String configJson = """
      {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "tie_word_embeddings": false
      }
      """;
    ContainerCatalog catalog = ContainerCatalog.ofHf(
      ModelSupport.Source.ONNX,
      "llama.onnx",
      configJson,
      Set.of("model.embed_tokens.weight"));
    ModelBinding.BoundModel bound = ModelBinding.bind(catalog);
    assertEquals(WeightNames.ARCH_LLAMA, bound.selection().architectureId());
    assertFalse(bound.selection().isEmbedding());
    assertEquals(WeightNames.ARCH_LLAMA, bound.processor().architectureId());
  }
}
