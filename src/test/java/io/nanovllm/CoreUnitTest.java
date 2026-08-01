package io.nanovllm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.nanovllm.chat.AssistantParts;
import io.nanovllm.chat.ChatMessages;
import io.nanovllm.engine.BlockManager;
import io.nanovllm.engine.Sequence;
import io.nanovllm.prompts.ChatPrompts;
import io.nanovllm.tensor.FloatKernels;
import io.nanovllm.tensor.FloatKernelsFactory;
import io.nanovllm.tensor.Ops;
import io.nanovllm.tensor.Tensor;
import io.nanovllm.utils.BundledModels;
import io.nanovllm.utils.Json;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CoreUnitTest {

  @Test
  void jsonParsesObject() {
    Map<String, Object> m = Json.parseObject("{\"a\":1,\"b\":[true,null,\"x\"],\"c\":1.5}");
    assertEquals(1, Json.asInt(m.get("a"), 0));
    assertEquals(1.5f, Json.asFloat(m.get("c"), 0), 1e-6);
    assertEquals(3, Json.asArray(m.get("b")).size());
  }

  @Test
  void linearAndSoftmax() {
    Tensor w = Tensor.of(new float[] {1, 0, 0, 1}, 2, 2);
    Tensor x = Tensor.of(new float[] {2, 3}, 1, 2);
    Tensor y = Ops.linear(x, w, null);
    assertEquals(2f, y.get(0), 1e-5);
    assertEquals(3f, y.get(1), 1e-5);

    Tensor probs = Ops.softmaxLastDim(Tensor.of(new float[] {0, 0, 0}, 1, 3));
    assertEquals(1f / 3f, probs.get(0), 1e-5);
  }

  @Test
  void siluAndMul() {
    Tensor x2 = Tensor.of(new float[] {1f, 2f}, 2);
    float silu1 = 1f / (1f + (float) Math.exp(-1f));
    assertEquals(silu1 * 2f, Ops.siluAndMul(x2).get(0), 1e-5);
  }

  @Test
  void addRmsNormFusedMatchesSeparateMath() {
    Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);
    Tensor residual = Tensor.of(new float[] {0.5f, -0.5f, 1f, 0f}, 2, 2);
    Tensor weight = Tensor.of(new float[] {1f, 1f}, 2);
    float eps = 1e-6f;

    Tensor[] fused = Ops.addRmsNorm(x, residual, weight, eps);
    Tensor summed = fused[1];
    assertEquals(1.5f, summed.get(0), 1e-5);
    assertEquals(1.5f, summed.get(1), 1e-5);
    assertEquals(4f, summed.get(2), 1e-5);
    assertEquals(4f, summed.get(3), 1e-5);

    Tensor expected = Ops.rmsNorm(summed, weight, eps);
    for (int i = 0; i < expected.numel(); i++) {
      assertEquals(expected.get(i), fused[0].get(i), 1e-5);
    }
  }

  @Test
  void blockManagerPrefixCache() {
    Sequence.setBlockSize(4);
    BlockManager bm = new BlockManager(16, 4);
    Sequence a = new Sequence(List.of(1, 2, 3, 4, 5, 6, 7, 8, 9), new SamplingParams(0.6f, 8));
    int cached = bm.canAllocate(a);
    assertTrue(cached >= 0);
    bm.allocate(a, cached);
    a.setNumScheduledTokens(a.numTokens() - a.numCachedTokens());
    bm.hashBlocks(a);
    a.addCachedTokens(a.numScheduledTokens());
    a.setNumScheduledTokens(0);

    Sequence b = new Sequence(List.of(1, 2, 3, 4, 5, 6, 7, 8, 10), new SamplingParams(0.6f, 8));
    int cachedB = bm.canAllocate(b);
    assertEquals(2, cachedB); // first two full blocks shared
  }

  @Test
  void float16Conversion() {
    float one = io.nanovllm.utils.SafetensorsReader.float16ToFloat(0x3C00);
    assertEquals(1.0f, one, 1e-3);
    float bf = io.nanovllm.utils.SafetensorsReader.bfloat16ToFloat(0x3F80);
    assertEquals(1.0f, bf, 1e-3);
  }

  @Test
  void bundledQwenModelIsPresent() {
    var path = io.nanovllm.utils.BundledModels.find(io.nanovllm.utils.BundledModels.QWEN3_0_6B);
    assertTrue(path.isPresent(), "run models/download-qwen3-0.6b.sh");
    assertTrue(java.nio.file.Files.isRegularFile(path.get().resolve("config.json")));
    assertTrue(java.nio.file.Files.isRegularFile(path.get().resolve("model.safetensors")));
  }

  @Test
  void vectorLinearMatchesScalar() {
    int rows = 4;
    int in = 64;
    int out = 32;
    float[] x = new float[rows * in];
    float[] w = new float[out * in];
    for (int i = 0; i < x.length; i++) {
      x[i] = (i % 7) * 0.1f;
    }
    for (int i = 0; i < w.length; i++) {
      w[i] = ((i * 3) % 11) * 0.05f;
    }
    Tensor xt = Tensor.of(x, rows, in);
    Tensor wt = Tensor.of(w, out, in);
    Tensor y = Ops.linear(xt, wt, null);
    for (int r = 0; r < rows; r++) {
      for (int o = 0; o < out; o++) {
        float expected = 0f;
        for (int i = 0; i < in; i++) {
          expected += x[r * in + i] * w[o * in + i];
        }
        assertEquals(expected, y.get(r * out + o), 1e-4f);
      }
    }
    assertTrue(io.nanovllm.tensor.VectorMath.backendInfo().contains("tileN"));
    String kernels = io.nanovllm.tensor.FloatKernels.get().name();
    assertTrue(kernels.contains("Vector API") || kernels.equals("scalar"), kernels);
  }

  @Test
  void scalarAndVectorKernelsAgree() {
    assertTrue(FloatKernelsFactory.isVectorApiAvailable());
    FloatKernels scalar = FloatKernelsFactory.create("scalar");
    FloatKernels vector = FloatKernelsFactory.create("vector");

    float[] a = new float[64];
    float[] b = new float[64];
    float[] w = new float[64];
    float[] outS = new float[64];
    float[] outV = new float[64];
    for (int i = 0; i < 64; i++) {
      a[i] = i * 0.1f;
      b[i] = 1.0f + (i % 7) * 0.01f;
      w[i] = 0.5f + (i % 5) * 0.02f;
    }

    assertEquals(scalar.dot(a, 0, b, 0, 64), vector.dot(a, 0, b, 0, 64), 1e-4f);
    assertEquals(scalar.sumSquares(a, 0, 64), vector.sumSquares(a, 0, 64), 1e-3f);
    scalar.scaleAdd(a, 0, w, 0, 1.5f, outS, 0, 64);
    vector.scaleAdd(a, 0, w, 0, 1.5f, outV, 0, 64);
    for (int i = 0; i < 64; i++) {
      assertEquals(outS[i], outV[i], 1e-5f);
    }

    FloatKernels best = FloatKernelsFactory.createBestAvailable();
    assertTrue(best.name().contains("Vector API"), best.name());
  }

  @Test
  void cleanAssistantTextHidesTemplateNoise() {
    String raw = "</think>\n\nHello there<|im_end|>";
    assertEquals("Hello there", AssistantParts.cleanAssistantText(raw));
  }

  @Test
  void cleanAssistantTextKeepsGreetingReplies() {
    assertEquals("Hello! How can I assist you today?",
        AssistantParts.cleanAssistantText(
            "</think>\n\nHello! How can I assist you today?<|im_end|>"));
    assertEquals("hello", AssistantParts.cleanAssistantText("hello"));
  }

  @Test
  void cleanAssistantTextUsesThinkBodyWhenAnswerEmpty() {
    assertEquals("Tere hommikust",
        AssistantParts.cleanAssistantText("<think>\nTere hommikust\n</think>\n\n"));
  }

  @Test
  void streamingDecodeHoldsIncompleteUtf8() {
    byte[] shch = "щ".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    assertEquals(2, shch.length);
    assertEquals("", io.nanovllm.tokenizer.Tokenizer.decodeUtf8Complete(new byte[] {shch[0]}));
    assertEquals("щ", io.nanovllm.tokenizer.Tokenizer.decodeUtf8Complete(shch));
    assertEquals("ащ", io.nanovllm.tokenizer.Tokenizer.decodeUtf8Complete(
        ("ащ").getBytes(java.nio.charset.StandardCharsets.UTF_8)));

    var path = io.nanovllm.utils.BundledModels.require(io.nanovllm.utils.BundledModels.QWEN3_0_6B);
    var tok = io.nanovllm.tokenizer.Tokenizer.fromPretrained(path);
    List<Integer> ids = tok.encode("обычное средство щелочное");
    for (int n = 1; n <= ids.size(); n++) {
      String partial = tok.decode(ids.subList(0, n), false);
      assertFalse(partial.contains("\uFFFD"), () -> "replacement in: " + partial);
    }
    assertTrue(tok.decode(ids, false).contains("щ"));
  }

  @Test
  void tokenizerEncodesSpecialTokensAtomically() {
    var path = io.nanovllm.utils.BundledModels.require(io.nanovllm.utils.BundledModels.QWEN3_0_6B);
    var tok = io.nanovllm.tokenizer.Tokenizer.fromPretrained(path);
    List<Integer> ids = tok.encode(
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assertTrue(ids.contains(151644)); // im_start
    assertTrue(ids.contains(151645)); // im_end
    assertTrue(ids.contains(151667)); // <think>
    assertTrue(ids.contains(151668)); // </think>
    assertTrue(tok.stopTokenIds().contains(151645));
  }

  @Test
  void samplingParamsDefaultsDisableTopK() {
    SamplingParams sp = new SamplingParams(0.6f, 64);
    assertEquals(0, sp.topK());
    assertEquals(0.9f, sp.topP(), 1e-6);
  }

  @Test
  void streamDisplayStripsThinkAndSpecials() {
    assertEquals("Hello", AssistantParts.streamDisplayText("<think>secret</think>Hello"));
    assertEquals("still thinking", AssistantParts.streamDisplayText("<think>still thinking"));
    assertEquals("ok", AssistantParts.streamDisplayText("ok<|im_end|>"));
  }

  @Test
  void assistantPartsHoldsIncompleteThinkTag() {
    AssistantParts partial = AssistantParts.parse("<think");
    assertEquals("", partial.thinking());
    assertEquals("", partial.answer());
    assertEquals(false, partial.thinkOpen());

    AssistantParts afterClose = AssistantParts.parse(
        "<think>\nplan\n</think>\n<think");
    assertEquals("plan", afterClose.thinking());
    assertEquals("", afterClose.answer());
    assertEquals(false, afterClose.thinkOpen());
  }

  @Test
  void assistantPartsHandlesSecondThinkBlock() {
    AssistantParts openSecond = AssistantParts.parse(
        "<think>\nplan\n</think>\n\n<think>\nmore");
    assertEquals("plan\nmore", openSecond.thinking());
    assertEquals("", openSecond.answer());
    assertEquals(true, openSecond.thinkOpen());

    AssistantParts withAnswer = AssistantParts.parse(
        "<think>\nplan\n</think>\n\n1\n<think>\nnoise</think>\n");
    assertEquals("plan\nnoise", withAnswer.thinking());
    assertEquals("1", withAnswer.answer());
    assertEquals(false, withAnswer.thinkOpen());
  }

  @Test
  void salvageFromThinkingPrefersStatedShortAnswer() {
    assertEquals("1", AssistantParts.salvageFromThinking("""
        Okay, the user wants a score.
        Therefore, the answer should be 1.
        """.stripIndent()));
    assertEquals("1", AssistantParts.salvageFromThinking("""
        some reasoning
        1
        """.stripIndent()));
  }

  @Test
  void assistantPartsSplitsThinkAndAnswer() {
    AssistantParts parts = AssistantParts.parse(
        "<think>\nplan\n</think>\n\nTere hommikust<|im_end|>");
    assertEquals("plan", parts.thinking());
    assertEquals("Tere hommikust", parts.answer());
    assertEquals(false, parts.thinkOpen());
  }

  @Test
  void chatSystemPromptUsesDialogHistory() {
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("You are the Assistant"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("User"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.toLowerCase().contains("conversation"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.toLowerCase().contains("do not repeat")
        || ChatPrompts.CHAT_SYSTEM.toLowerCase().contains("do not greet"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("<think>"));
    assertFalse(ChatPrompts.CHAT_SYSTEM.toLowerCase().contains("knowledge base"));
    assertEquals(ChatPrompts.CHAT_SYSTEM, ChatPrompts.systemFor(false));
    assertEquals(ChatPrompts.GEMMA_CHAT_SYSTEM, ChatPrompts.systemFor(true));
    assertTrue(ChatPrompts.GEMMA_CHAT_SYSTEM.isBlank());
    assertTrue(ChatPrompts.gemmaUserContent("SYS", "hi", true).startsWith("SYS"));
    assertEquals("hi", ChatPrompts.gemmaUserContent("SYS", "hi", false));
    assertEquals("hi", ChatPrompts.gemmaUserContent(null, "hi", true));
    assertEquals("hi", ChatPrompts.gemmaUserContent("", "hi", true));
    assertTrue(ChatPrompts.isSetupBoilerplate("Okay, I'm ready."));
    assertTrue(ChatPrompts.isSetupBoilerplate("Okay, I understand. Let's begin."));
    assertFalse(ChatPrompts.isSetupBoilerplate("Hello! How can I help you today?"));
    assertFalse(ChatPrompts.isSetupBoilerplate("The president of Estonia is Alar Karis."));
    assertTrue(ChatMessages.newConversation(true).isEmpty());
    assertFalse(ChatMessages.newConversation(false).isEmpty());
  }

  @Test
  void causalLmFactoryDetectsArchFromConfig() throws Exception {
    Path qwenCfg = java.nio.file.Files.createTempFile("qwen-cfg", ".json");
    Path gemmaCfg = java.nio.file.Files.createTempFile("gemma-cfg", ".json");
    try {
      java.nio.file.Files.writeString(qwenCfg, """
          {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":64,
           "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
           "vocab_size":100,"intermediate_size":128,"num_hidden_layers":1,
           "max_position_embeddings":128,"rms_norm_eps":1e-6,"hidden_act":"silu"}
          """);
      java.nio.file.Files.writeString(gemmaCfg, """
          {"model_type":"gemma3_text","architectures":["Gemma3ForCausalLM"],"hidden_size":64,
           "num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
           "vocab_size":100,"intermediate_size":128,"num_hidden_layers":2,
           "max_position_embeddings":128,"rms_norm_eps":1e-6,
           "hidden_activation":"gelu_pytorch_tanh","sliding_window":32,
           "layer_types":["sliding_attention","full_attention"],
           "rope_local_base_freq":10000,"query_pre_attn_scalar":16,"rope_theta":1000000}
          """);
      Config.HfConfig qwen = Config.HfConfig.load(qwenCfg);
      Config.HfConfig gemma = Config.HfConfig.load(gemmaCfg);
      assertEquals("qwen3", io.nanovllm.models.CausalLMFactory.detect(qwen));
      assertEquals("gemma3", io.nanovllm.models.CausalLMFactory.detect(gemma));
      assertTrue(gemma.isSlidingLayer(0));
      assertFalse(gemma.isSlidingLayer(1));
      assertEquals((float) Math.pow(16, -0.5), gemma.attentionScale(), 1e-6f);
      assertEquals("gelu_pytorch_tanh", gemma.effectiveActivation());
    } finally {
      java.nio.file.Files.deleteIfExists(qwenCfg);
      java.nio.file.Files.deleteIfExists(gemmaCfg);
    }
  }

  @Test
  void gemmaRmsNormAndGeluSmoke() {
    Tensor x = Tensor.of(new float[] {1f, -1f, 2f, 0f}, 2, 2);
    Tensor w = Tensor.of(new float[] {0f, 0.5f}, 2);
    Tensor out = Ops.rmsNorm(x, w, 1e-6f, true);
    assertTrue(out.get(0) != 0f);
    float scale1 = 1f + 0f;
    float scale2 = 1f + 0.5f;
    Tensor plain = Ops.rmsNorm(x, Tensor.of(new float[] {1f, 1f}, 2), 1e-6f, false);
    assertEquals(plain.get(0) * scale1, out.get(0), 1e-5f);
    assertEquals(plain.get(1) * scale2, out.get(1), 1e-5f);

    Tensor gateUp = Tensor.of(new float[] {1f, 2f}, 2);
    float gelu1 = 0.5f * 1f * (1f + (float) Math.tanh(0.7978845608028654 * (1f + 0.044715)));
    assertEquals(gelu1 * 2f, Ops.geluPytorchTanhAndMul(gateUp).get(0), 1e-5f);
  }

  @Test
  void qwenTokenizerIsNotGemmaChat() {
    var path = io.nanovllm.utils.BundledModels.require(io.nanovllm.utils.BundledModels.QWEN3_0_6B);
    var tok = io.nanovllm.tokenizer.Tokenizer.fromPretrained(path);
    assertFalse(tok.isGemmaChat());
    String chat = tok.applyChatTemplate(
        List.of(Map.of("role", "user", "content", "hi")), true, false);
    assertTrue(chat.contains("<|im_start|>"));
    assertFalse(chat.contains("<start_of_turn>"));
  }

  @Test
  void gemmaChatTemplateBranchingWithoutWeights() throws Exception {
    assertEquals("gemma3",
        io.nanovllm.models.CausalLMFactory.detect(
            new Config.HfConfig(
                100, 64, 128, 1, 4, 1, 16, 128, 1e-6f, "gelu", false, false,
                1e6f, null, "float32", "gemma3_text",
                List.of("Gemma3ForCausalLM"), "gelu_pytorch_tanh",
                512, List.of("sliding_attention"), 10_000f, 256f)));

    Path dir = java.nio.file.Files.createTempDirectory("gemma-tok");
    try {
      java.nio.file.Files.writeString(dir.resolve("config.json"),
          "{\"model_type\":\"gemma3_text\",\"vocab_size\":32}");
      java.nio.file.Files.writeString(dir.resolve("tokenizer_config.json"), """
          {"eos_token":"<eos>","pad_token":"<pad>",
           "chat_template":"{% for m in messages %}<start_of_turn>{{ m.role }}\\n{{ m.content }}<end_of_turn>\\n{% endfor %}"}
          """);
      java.nio.file.Files.writeString(dir.resolve("tokenizer.json"), """
          {"model":{"type":"BPE","vocab":{"<bos>":0,"<eos>":1,"<pad>":2,"a":3,"▁":4},"merges":[]},
           "added_tokens":[
             {"id":0,"content":"<bos>","special":true},
             {"id":1,"content":"<eos>","special":true},
             {"id":5,"content":"<start_of_turn>","special":true},
             {"id":6,"content":"<end_of_turn>","special":true}
           ],
           "pre_tokenizer":{"type":"Metaspace","replacement":"▁"}}
          """);
      var tok = io.nanovllm.tokenizer.Tokenizer.fromPretrained(dir);
      assertTrue(tok.isGemmaChat());
      String chatThinkFlag = tok.applyChatTemplate(
          List.of(Map.of("role", "user", "content", "hi")), true, true);
      assertTrue(
          chatThinkFlag.startsWith("<bos>") || chatThinkFlag.contains("<start_of_turn>user"));
      assertTrue(chatThinkFlag.contains("<start_of_turn>model"));
      assertFalse(chatThinkFlag.contains("<think>"));
      assertEquals(
          tok.applyChatTemplate(List.of(Map.of("role", "user", "content", "hi")), true, false),
          chatThinkFlag);
    } finally {
      try (var walk = java.nio.file.Files.walk(dir)) {
        walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
          try {
            java.nio.file.Files.deleteIfExists(p);
          } catch (Exception ignored) {
          }
        });
      }
    }
  }

  @Test
  void gemmaSmokeWhenWeightsPresent() {
    var path = io.nanovllm.utils.BundledModels.find(io.nanovllm.utils.BundledModels.GEMMA3_270M);
    org.junit.jupiter.api.Assumptions.assumeTrue(path.isPresent(), "Gemma3-270M not downloaded");
    var tok = io.nanovllm.tokenizer.Tokenizer.fromPretrained(path.get());
    assertTrue(tok.isGemmaChat());
    assertEquals(List.of(23391), tok.encode("hello"));
    assertEquals(List.of(23391, 1902), tok.encode("hello world"));
    String chatNoThink = tok.applyChatTemplate(
        List.of(Map.of("role", "user", "content", "hello")), true, false);
    assertEquals(
        List.of(2, 105, 2364, 107, 23391, 106, 107, 105, 4368, 107),
        tok.encode(chatNoThink));
    String chatThinkFlag = tok.applyChatTemplate(
        List.of(Map.of("role", "user", "content", "hello")), true, true);
    assertEquals(chatNoThink, chatThinkFlag);
    assertTrue(chatNoThink.contains("<start_of_turn>user"));
    assertTrue(chatNoThink.contains("<start_of_turn>model"));
    assertFalse(chatNoThink.contains("<think>"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("<think>"));
    assertFalse(ChatPrompts.GEMMA_CHAT_SYSTEM.contains("<think>"));
    Config.HfConfig hf = null;
    try {
      hf = Config.HfConfig.load(path.get().resolve("config.json"));
    } catch (Exception e) {
      throw new AssertionError(e);
    }
    assertTrue(hf.tieWordEmbeddings());
    assertEquals("gemma3", io.nanovllm.models.CausalLMFactory.detect(hf));
    io.nanovllm.models.CausalLM model = io.nanovllm.models.CausalLMFactory.create(hf);
    assertEquals("gemma3", model.architectureName());
    assertTrue(model.hasParameter("model.layers.0.pre_feedforward_layernorm.weight"));
  }

  @Test
  void stripChatMarkupRemovesGemmaTurnTokens() {
    assertEquals("", AssistantParts.stripChatMarkup("<end_of_turn>"));
    assertEquals("Hi", AssistantParts.stripChatMarkup("Hi<end_of_turn>"));
    assertEquals("model\nplan",
        AssistantParts.stripChatMarkup("<start_of_turn>model\nplan<end_of_turn>"));
  }

  @Test
  void exampleModelMenuSelectsBundledPaths() throws Exception {
    var qwen = BundledModels.find(BundledModels.QWEN3_0_6B);
    org.junit.jupiter.api.Assumptions.assumeTrue(qwen.isPresent());
    Path chosen = Example.resolveModel(
        new String[0],
        new java.io.BufferedReader(new java.io.StringReader("1\n")));
    assertEquals(qwen.get(), chosen);

    assertTrue(Example.resolveModel(
        new String[0],
        new java.io.BufferedReader(new java.io.StringReader("3\n"))) == null);
  }

  @Test
  void exampleSkipsMenuWhenCliPathGiven() throws Exception {
    var qwen = BundledModels.require(BundledModels.QWEN3_0_6B);
    Path chosen = Example.resolveModel(
        new String[] {qwen.toString()},
        new java.io.BufferedReader(new java.io.StringReader("2\n")));
    assertEquals(qwen.toAbsolutePath().normalize(), chosen.toAbsolutePath().normalize());
  }
}
