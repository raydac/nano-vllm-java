package com.igormaznitsa.nanollvm;

import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.createTempFile;
import static java.nio.file.Files.deleteIfExists;
import static java.nio.file.Files.isRegularFile;
import static java.nio.file.Files.walk;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatRole;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.chat.LlmTextKind;
import com.igormaznitsa.nanollvm.engine.BlockManager;
import com.igormaznitsa.nanollvm.engine.Sequence;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.GenerationStats;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.LlmAdvisor;
import com.igormaznitsa.nanollvm.llm.LlmAdvisorMixer;
import com.igormaznitsa.nanollvm.llm.SamplingDefaults;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.tensor.FloatKernels;
import com.igormaznitsa.nanollvm.tensor.FloatKernelsFactory;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CoreUnitTest {

  @Test
  void llmListenerSilentAndStatus() {
    assertTrue(LlmListeners.isSilent(LlmListeners.silent()));
    assertFalse(LlmListeners.isSilent(LlmListeners.toSystem()));
    LlmListeners.info(LlmListeners.silent(), null, "must not reach the console");

    var captured = new java.util.ArrayList<String>();
    LlmListener probe = (source, event) -> captured.add(event.kind() + ":" + event.text());
    LlmListeners.info(probe, null, "hello");
    LlmListeners.progressf(probe, null, "gen %d", 1);
    assertEquals(2, captured.size());
    assertTrue(captured.getFirst().startsWith(LlmTextKind.STATUS_INFO + ":hello"));
    assertEquals(LlmTextKind.STATUS_PROGRESS + ":gen 1", captured.get(1));
  }

  @Test
  void llmBuilderIsFluentAndDefaultsQuiet() {
    Path path = BundledModelAssumptions.requireQwen3();

    LlmModel model = LlmModelFactory.make(path);
    try (model) {
      LLM.Builder builder = LLM.builder(model);
      assertSame(builder, builder
        .maxModelLen(512)
        .maxNumSeqs(2)
        .maxNumBatchedTokens(1024)
        .kvcacheBlockSize(256)
        .numKvcacheBlocks(32)
        .kvHeapFraction(0.5f)
        .warmup()
        .skipWarmup()
        .warmup(false)
        .allowUnpackParameters()
        .allowUnpackParameters(false)
        .listen(LlmListeners.toSystem())
        .listen(LlmListeners.silent())
        .systemPrompt("Answer briefly.")
        .noSystemPrompt()
        .defaultSystemPrompt());
    }
  }

  @Test
  void sharedModelIsReusedAcrossLlmsWhenWeightsPresent() {
    Path path = BundledModelAssumptions.requireQwen3();

    LlmModel model = LlmModelFactory.make(path);
    try (model;
         LLM a = LLM.builder(model).maxModelLen(256).numKvcacheBlocks(32).build();
         LLM b = LLM.builder(model).maxModelLen(256).numKvcacheBlocks(32).build()) {
      assertSame(model, a.model());
      assertSame(model, b.model());
      assertSame(model.tokenizer(), a.tokenizer());
      assertSame(model.tokenizer(), b.tokenizer());
      assertEquals(model.architectureName(), a.model().architectureName());
    }
  }

  @Test
  void closedLlmAndModelRejectFurtherUseWhenWeightsPresent() {
    Path path = BundledModelAssumptions.requireQwen3();

    LlmModel model = LlmModelFactory.make(path);
    LLM llm = LLM.builder(model).maxModelLen(256).numKvcacheBlocks(32).build();
    llm.close();
    assertTrue(llm.isClosed());
    assertThrows(IllegalStateException.class, () -> llm.chat(32));
    assertThrows(IllegalStateException.class, llm::newConversation);
    llm.close();

    model.close();
    assertTrue(model.isClosed());
    assertThrows(IllegalStateException.class, model::architectureName);
    assertThrows(IllegalStateException.class, () -> LLM.builder(model).build());
    model.close();
  }

  @Test
  void advisorsConfiguredOnBuilderWhenWeightsPresent() {
    Path path = BundledModelAssumptions.requireQwen3();

    LlmModel model = LlmModelFactory.make(path);
    LlmAdvisor facts = LlmAdvisor.builder().name("Facts").prompt("Fact check").build();
    LlmAdvisor risks = LlmAdvisor.builder().name("Risks").prompt("Argue risks").build();
    try (model) {
      try (LLM llm = LLM.builder(model)
        .maxModelLen(256)
        .numKvcacheBlocks(32)
        .advisors(LlmAdvisorMixer.defaults(), facts, risks)
        .build()) {
        assertEquals(2, llm.advisors().size());
        assertEquals("Facts", llm.advisors().get(0).name());
        assertEquals("Fact check", llm.advisors().get(0).prompt());
        assertEquals("Risks", llm.advisors().get(1).name());
        assertNotNull(llm.advisorMixer());
      }

      try (LLM llm = LLM.builder(model).maxModelLen(256).numKvcacheBlocks(32).build()) {
        assertTrue(llm.advisors().isEmpty());
      }

      LLM.Builder rejectsDuplicate =
        LLM.builder(model).maxModelLen(256).numKvcacheBlocks(32);
      assertThrows(
        IllegalArgumentException.class,
        () -> rejectsDuplicate.advisors(
          LlmAdvisorMixer.defaults(),
          LlmAdvisor.builder().name("Same").prompt("a").build(),
          LlmAdvisor.builder().name("same").prompt("b").build()));

      try (LLM llm = LLM.builder(model)
        .maxModelLen(256)
        .numKvcacheBlocks(32)
        .advisors(LlmAdvisorMixer.defaults())
        .build()) {
        assertTrue(llm.advisors().isEmpty());
      }

      try (LLM llm = LLM.builder(model)
        .maxModelLen(256)
        .numKvcacheBlocks(32)
        .advisors(LlmAdvisorMixer.defaults(), facts)
        .noAdvisors()
        .build()) {
        assertTrue(llm.advisors().isEmpty());
      }
    }
  }

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
  void addAndSeparateSiluAndMul() {
    Tensor a = Tensor.of(new float[] {1f, 2f}, 2);
    Tensor b = Tensor.of(new float[] {3f, 4f}, 2);
    Tensor sum = Ops.add(a, b);
    assertEquals(4f, sum.get(0), 1e-5);
    assertEquals(6f, sum.get(1), 1e-5);

    Tensor gate = Tensor.of(new float[] {1f, -1f}, 2);
    Tensor up = Tensor.of(new float[] {2f, 3f}, 2);
    float silu1 = 1f / (1f + (float) Math.exp(-1f));
    float siluNeg = -1f / (1f + (float) Math.exp(1f));
    Tensor out = Ops.siluAndMul(gate, up);
    assertEquals(silu1 * 2f, out.get(0), 1e-5);
    assertEquals(siluNeg * 3f, out.get(1), 1e-5);
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
    BlockManager bm = new BlockManager(16, 4);
    Sequence a = new Sequence(List.of(1, 2, 3, 4, 5, 6, 7, 8, 9), new SamplingParams(0.6f, 8), 4);
    int cached = bm.canAllocate(a);
    assertTrue(cached >= 0);
    bm.allocate(a, cached);
    a.setNumScheduledTokens(a.numTokens() - a.numCachedTokens());
    bm.hashBlocks(a);
    a.addCachedTokens(a.numScheduledTokens());
    a.setNumScheduledTokens(0);

    Sequence b = new Sequence(List.of(1, 2, 3, 4, 5, 6, 7, 8, 10), new SamplingParams(0.6f, 8), 4);
    int cachedB = bm.canAllocate(b);
    assertEquals(2, cachedB); // first two full blocks shared
  }

  @Test
  void float16Conversion() {
    float one = com.igormaznitsa.nanollvm.internal.SafetensorsReader.float16ToFloat(0x3C00);
    assertEquals(1.0f, one, 1e-3);
    float bf = com.igormaznitsa.nanollvm.internal.SafetensorsReader.bfloat16ToFloat(0x3F80);
    assertEquals(1.0f, bf, 1e-3);
  }

  @Test
  void bundledQwenModelIsPresent() {
    Path path = BundledModelAssumptions.requireQwen3();
    assertTrue(isRegularFile(path.resolve("config.json")));
    assertTrue(isRegularFile(path.resolve("model.safetensors")));
  }

  @Test
  void vectorLinearMatchesScalar() {
    int rows = 4;
    int in = 64;
    int out = 32;
    float[] x = new float[rows * in];
    float[] w = new float[out * in];
    for (int i = 0; i < x.length; i++) {
      x[i] = i % 7 * 0.1f;
    }
    for (int i = 0; i < w.length; i++) {
      w[i] = i * 3 % 11 * 0.05f;
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
    assertTrue(
      com.igormaznitsa.nanollvm.tensor.MatmulRuntime.sequential().backendInfo().contains("tileN"));
    String kernels = com.igormaznitsa.nanollvm.tensor.FloatKernels.get().name();
    assertTrue(kernels.contains("Vector API") || kernels.equals("scalar"), kernels);
  }

  @Test
  void parallelLinearMatchesSequential() {
    int rows = 3;
    int in = 128;
    int out = 256;
    float[] x = new float[rows * in];
    float[] w = new float[out * in];
    float[] bias = new float[out];
    for (int i = 0; i < x.length; i++) {
      x[i] = i % 9 * 0.07f;
    }
    for (int i = 0; i < w.length; i++) {
      w[i] = i * 5 % 13 * 0.03f;
    }
    for (int i = 0; i < out; i++) {
      bias[i] = i % 5 * 0.01f;
    }
    float[] sequential = new float[rows * out];
    float[] parallel = new float[rows * out];
    MatmulRuntime.sequential().linear(
      x, 0, w, 0, bias, sequential, 0, rows, in, out);
    try (MatmulRuntime multi = MatmulRuntime.builder()
      .cpuThreads(Math.max(2, Runtime.getRuntime().availableProcessors()))
      .build()) {
      multi.linear(x, 0, w, 0, bias, parallel, 0, rows, in, out);
    }
    for (int i = 0; i < sequential.length; i++) {
      assertEquals(sequential[i], parallel[i], 1e-4f, "index " + i);
    }
    assertTrue(com.igormaznitsa.nanollvm.tensor.MatmulRuntime.sequential().backendInfo()
      .contains("cpuThreads"));
  }

  @Test
  void disableMultiCpuUsesSequentialRuntimeWithoutExecutor() {
    try (MatmulRuntime withDisable = MatmulRuntime.builder().disableMultiCpu().build();
         MatmulRuntime withOne = MatmulRuntime.builder().cpuThreads(1).build()) {
      assertSame(MatmulRuntime.sequential(), withDisable);
      assertSame(MatmulRuntime.sequential(), withOne);
      assertEquals(1, withDisable.cpuThreads());
      assertTrue(withDisable.backendInfo().contains("sequential"));
    }

    var ignored = java.util.concurrent.Executors.newSingleThreadExecutor();
    try (MatmulRuntime runtime = MatmulRuntime.builder()
      .disableMultiCpu()
      .executor(ignored)
      .build()) {
      assertSame(MatmulRuntime.sequential(), runtime);
      assertTrue(runtime.backendInfo().contains("sequential"));
    } finally {
      ignored.shutdownNow();
    }
  }

  @Test
  void disableMultiCpuWinsOverCpuThreadsSystemPropertyWhenWeightsPresent() {
    Path path = BundledModelAssumptions.requireQwen3();

    String previous = System.getProperty("nanollvm.cpu.threads");
    System.setProperty("nanollvm.cpu.threads", "8");
    try (LlmModel model = LlmModelFactory.make(path);
         LLM llm = LLM.builder(model)
           .disableMultiCpu()
           .maxModelLen(256)
           .numKvcacheBlocks(32)
           .build()) {
      assertEquals(1, llm.config().cpuThreads());
    } finally {
      if (previous == null) {
        System.clearProperty("nanollvm.cpu.threads");
      } else {
        System.setProperty("nanollvm.cpu.threads", previous);
      }
    }
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
      b[i] = 1.0f + i % 7 * 0.01f;
      w[i] = 0.5f + i % 5 * 0.02f;
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
    assertEquals("Hello there", ChatReply.cleanAssistantText(raw));
  }

  @Test
  void cleanAssistantTextKeepsGreetingReplies() {
    assertEquals("Hello! How can I assist you today?",
      ChatReply.cleanAssistantText(
            "</think>\n\nHello! How can I assist you today?<|im_end|>"));
    assertEquals("hello", ChatReply.cleanAssistantText("hello"));
  }

  @Test
  void cleanAssistantTextUsesThinkBodyWhenAnswerEmpty() {
    assertEquals("Tere hommikust",
      ChatReply.cleanAssistantText("<think>\nTere hommikust\n</think>\n\n"));
  }

  @Test
  void streamingDecodeHoldsIncompleteUtf8() {
    byte[] shch = "щ".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    assertEquals(2, shch.length);
    assertEquals("",
        com.igormaznitsa.nanollvm.tokenizer.Tokenizer.decodeUtf8Complete(new byte[] {shch[0]}));
    assertEquals("щ", com.igormaznitsa.nanollvm.tokenizer.Tokenizer.decodeUtf8Complete(shch));
    assertEquals("ащ", com.igormaznitsa.nanollvm.tokenizer.Tokenizer.decodeUtf8Complete(
      "ащ".getBytes(java.nio.charset.StandardCharsets.UTF_8)));

    Path path = BundledModelAssumptions.requireQwen3();
    var tok = com.igormaznitsa.nanollvm.tokenizer.Tokenizer.fromPretrained(path);
    List<Integer> ids = tok.encode("обычное средство щелочное");
    for (int n = 1; n <= ids.size(); n++) {
      String partial = tok.decode(ids.subList(0, n), false);
      assertFalse(partial.contains("\uFFFD"), () -> "replacement in: " + partial);
    }
    assertTrue(tok.decode(ids, false).contains("щ"));
  }

  @Test
  void tokenizerEncodesSpecialTokensAtomically() {
    Path path = BundledModelAssumptions.requireQwen3();
    var tok = com.igormaznitsa.nanollvm.tokenizer.Tokenizer.fromPretrained(path);
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
    assertEquals("Hello", ChatReply.streamDisplayText("<think>secret</think>Hello"));
    assertEquals("still thinking", ChatReply.streamDisplayText("<think>still thinking"));
    assertEquals("ok", ChatReply.streamDisplayText("ok<|im_end|>"));
  }

  @Test
  void generationStatsThroughputAndChatReplyAttachment() {
    GenerationStats stats = new GenerationStats(10, 5, 2_000_000_000L);
    assertEquals(15, stats.totalTokens());
    assertEquals(2.5d, stats.completionTokensPerSecond(), 1e-9);

    ChatReply reply = ChatReply.parse("hi").withStats(stats);
    assertEquals(5, reply.stats().completionTokens());
    assertEquals(GenerationStats.NONE, ChatReply.parse("x").stats());
  }

  @Test
  void assistantPartsHoldsIncompleteThinkTag() {
    ChatReply partial = ChatReply.parse("<think");
    assertEquals("", partial.thinking());
    assertEquals("", partial.answer());
    assertEquals(false, partial.thinkOpen());

    ChatReply afterClose = ChatReply.parse(
        "<think>\nplan\n</think>\n<think");
    assertEquals("plan", afterClose.thinking());
    assertEquals("", afterClose.answer());
    assertEquals(false, afterClose.thinkOpen());
  }

  @Test
  void assistantPartsHandlesSecondThinkBlock() {
    ChatReply openSecond = ChatReply.parse(
        "<think>\nplan\n</think>\n\n<think>\nmore");
    assertEquals("plan\nmore", openSecond.thinking());
    assertEquals("", openSecond.answer());
    assertEquals(true, openSecond.thinkOpen());

    ChatReply withAnswer = ChatReply.parse(
        "<think>\nplan\n</think>\n\n1\n<think>\nnoise</think>\n");
    assertEquals("plan\nnoise", withAnswer.thinking());
    assertEquals("1", withAnswer.answer());
    assertEquals(false, withAnswer.thinkOpen());
  }

  @Test
  void salvageFromThinkingPrefersStatedShortAnswer() {
    assertEquals("1", ChatReply.salvageFromThinking("""
        Okay, the user wants a score.
        Therefore, the answer should be 1.
        """.stripIndent()));
    assertEquals("1", ChatReply.salvageFromThinking("""
        some reasoning
        1
        """.stripIndent()));
  }

  @Test
  void assistantPartsSplitsThinkAndAnswer() {
    ChatReply parts = ChatReply.parse(
        "<think>\nplan\n</think>\n\nTere hommikust<|im_end|>");
    assertEquals("plan", parts.thinking());
    assertEquals("Tere hommikust", parts.answer());
    assertEquals(false, parts.thinkOpen());
  }

  @Test
  void chatSystemPromptUsesDialogHistory() {
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("You are the Assistant"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("User"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.toLowerCase(Locale.ROOT).contains("conversation"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.toLowerCase(Locale.ROOT).contains("do not repeat")
      || ChatPrompts.CHAT_SYSTEM.toLowerCase(Locale.ROOT).contains("do not greet"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("<think>"));
    assertFalse(ChatPrompts.CHAT_SYSTEM.toLowerCase(Locale.ROOT).contains("knowledge base"));
    assertEquals(ChatPrompts.CHAT_SYSTEM, ChatPrompts.systemFor(false));
    assertEquals(ChatPrompts.GEMMA_CHAT_SYSTEM, ChatPrompts.systemFor(true));
    assertFalse(ChatPrompts.PLAIN_CHAT_SYSTEM.contains("<think>"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("<think>"));
    assertTrue(ChatPrompts.GEMMA_CHAT_SYSTEM.isBlank());
    assertTrue(ChatPrompts.gemmaUserContent("SYS", "hi", true).startsWith("SYS"));
    assertEquals("hi", ChatPrompts.gemmaUserContent("SYS", "hi", false));
    assertEquals("hi", ChatPrompts.gemmaUserContent(null, "hi", true));
    assertEquals("hi", ChatPrompts.gemmaUserContent("", "hi", true));
    assertTrue(ChatPrompts.isSetupBoilerplate("Okay, I'm ready."));
    assertTrue(ChatPrompts.isSetupBoilerplate("Okay, I understand. Let's begin."));
    assertTrue(ChatPrompts.isSetupBoilerplate(
      "Okay, I understand. I will respond with a short viewpoint, not the user-facing assistant."));
    assertFalse(ChatPrompts.isSetupBoilerplate("Hello! How can I help you today?"));
    assertFalse(ChatPrompts.isSetupBoilerplate("The president of Estonia is Alar Karis."));
    assertFalse(ChatPrompts.isSetupBoilerplate(
      "The universe is the totality of space, time, matter, and energy."));
    assertTrue(ChatPrompts.withAdvisorGuidance(ChatPrompts.CHAT_SYSTEM, true)
      .contains(ChatPrompts.ADVISOR_AWARE_ADDON));
    assertEquals("", ChatPrompts.withAdvisorGuidance(ChatPrompts.GEMMA_CHAT_SYSTEM, true));
    assertEquals(ChatPrompts.CHAT_SYSTEM,
      ChatPrompts.withAdvisorGuidance(ChatPrompts.CHAT_SYSTEM, false));
    assertTrue(ChatMessages.newConversation(true).isEmpty());
    assertFalse(ChatMessages.newConversation(false).isEmpty());
    assertEquals(1, ChatMessages.newConversation("Be brief.").size());
    assertTrue(ChatMessages.newConversation("").isEmpty());
    assertTrue(ChatMessages.newConversation(null).isEmpty());
    assertEquals(ChatRole.SYSTEM, ChatMessages.newConversation("Be brief.").getFirst().role());
    assertEquals("hi", ChatMessage.user("hi").content());
    assertEquals("user", ChatMessage.user("hi").toMap().get("role"));
    assertEquals(ChatRole.ASSISTANT, ChatRole.fromWire("model"));
  }

  @Test
  void samplingDefaultsPreferGemmaTopKWhenFlagged() {
    SamplingParams plain = SamplingDefaults.forTokenizer(null, 100);
    assertEquals(0, plain.topK());
    assertEquals(100, plain.maxTokens());
    assertEquals(0.95f, plain.topP(), 1e-6f);
  }

  @Test
  void causalLmFactoryDetectsArchFromConfig() throws Exception {
    Path qwenCfg = createTempFile("qwen-cfg", ".json");
    Path gemmaCfg = createTempFile("gemma-cfg", ".json");
    try {
      writeString(qwenCfg, """
          {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":64,
           "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
           "vocab_size":100,"intermediate_size":128,"num_hidden_layers":1,
           "max_position_embeddings":128,"rms_norm_eps":1e-6,"hidden_act":"silu"}
          """);
      writeString(gemmaCfg, """
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
      assertEquals("qwen3", com.igormaznitsa.nanollvm.models.internal.CausalLMFactory.detect(qwen));
      assertEquals("gemma3",
        com.igormaznitsa.nanollvm.models.internal.CausalLMFactory.detect(gemma));
      assertTrue(gemma.isSlidingLayer(0));
      assertFalse(gemma.isSlidingLayer(1));
      assertEquals((float) Math.pow(16, -0.5), gemma.attentionScale(), 1e-6f);
      assertEquals("gelu_pytorch_tanh", gemma.effectiveActivation());
    } finally {
      deleteIfExists(qwenCfg);
      deleteIfExists(gemmaCfg);
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
    Path path = BundledModelAssumptions.requireQwen3();
    var tok = com.igormaznitsa.nanollvm.tokenizer.Tokenizer.fromPretrained(path);
    assertFalse(tok.isGemmaChat());
    assertTrue(tok.invitesThinking());
    assertEquals(ChatPrompts.CHAT_SYSTEM, ChatPrompts.systemFor(tok));
    String chat = tok.applyChatTemplate(
        List.of(Map.of("role", "user", "content", "hi")), true, false);
    assertTrue(chat.contains("<|im_start|>"));
    assertFalse(chat.contains("<start_of_turn>"));
  }

  @Test
  void gemmaChatTemplateBranchingWithoutWeights() throws Exception {
    assertEquals("gemma3",
      com.igormaznitsa.nanollvm.models.internal.CausalLMFactory.detect(
            new Config.HfConfig(
                100, 64, 128, 1, 4, 1, 16, 128, 1e-6f, "gelu", false, false,
                1e6f, null, "float32", "gemma3_text",
                List.of("Gemma3ForCausalLM"), "gelu_pytorch_tanh",
              512, List.of("sliding_attention"), 10_000f, 256f, 0)));

    Path dir = createTempDirectory("gemma-tok");
    try {
      writeString(dir.resolve("config.json"),
          "{\"model_type\":\"gemma3_text\",\"vocab_size\":32}");
      writeString(dir.resolve("tokenizer_config.json"), """
          {"eos_token":"<eos>","pad_token":"<pad>",
           "chat_template":"{% for m in messages %}<start_of_turn>{{ m.role }}\\n{{ m.content }}<end_of_turn>\\n{% endfor %}"}
          """);
      writeString(dir.resolve("tokenizer.json"), """
          {"model":{"type":"BPE","vocab":{"<bos>":0,"<eos>":1,"<pad>":2,"a":3,"▁":4},"merges":[]},
           "added_tokens":[
             {"id":0,"content":"<bos>","special":true},
             {"id":1,"content":"<eos>","special":true},
             {"id":5,"content":"<start_of_turn>","special":true},
             {"id":6,"content":"<end_of_turn>","special":true}
           ],
           "pre_tokenizer":{"type":"Metaspace","replacement":"▁"}}
          """);
      var tok = com.igormaznitsa.nanollvm.tokenizer.Tokenizer.fromPretrained(dir);
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
      try (var walk = walk(dir)) {
        walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
          try {
            deleteIfExists(p);
          } catch (IOException e) {
            throw new UncheckedIOException("failed to delete temp path " + p, e);
          }
        });
      }
    }
  }

  @Test
  void gemmaSmokeWhenWeightsPresent() {
    Path path = BundledModelAssumptions.requireGemma3();
    var tok = com.igormaznitsa.nanollvm.tokenizer.Tokenizer.fromPretrained(path);
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
      hf = Config.HfConfig.load(path.resolve("config.json"));
    } catch (Exception e) {
      throw new AssertionError(e);
    }
    assertTrue(hf.tieWordEmbeddings());
    assertEquals("gemma3", com.igormaznitsa.nanollvm.models.internal.CausalLMFactory.detect(hf));
    assertTrue(com.igormaznitsa.nanollvm.models.internal.WeightSchema.gemma3(hf)
        .expects("model.layers.0.pre_feedforward_layernorm.weight"));
  }

  @Test
  void stripChatMarkupRemovesGemmaTurnTokens() {
    assertEquals("", ChatReply.stripChatMarkup("<end_of_turn>"));
    assertEquals("Hi", ChatReply.stripChatMarkup("Hi<end_of_turn>"));
    assertEquals("model\nplan",
      ChatReply.stripChatMarkup("<start_of_turn>model\nplan<end_of_turn>"));
  }
}
