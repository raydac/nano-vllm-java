package io.nanovllm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.nanovllm.chat.AssistantParts;
import io.nanovllm.chat.FactMemory;
import io.nanovllm.chat.MessageClassifier;
import io.nanovllm.engine.BlockManager;
import io.nanovllm.engine.Sequence;
import io.nanovllm.prompts.ChatPrompts;
import io.nanovllm.prompts.MessageAnalysis;
import io.nanovllm.prompts.MessageIntent;
import io.nanovllm.tensor.FloatKernels;
import io.nanovllm.tensor.FloatKernelsFactory;
import io.nanovllm.tensor.Ops;
import io.nanovllm.tensor.Tensor;
import io.nanovllm.utils.BundledModels;
import io.nanovllm.utils.Json;
import java.nio.file.Path;
import java.util.ArrayList;
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
    assertEquals("Hello there", FactMemory.cleanAssistantText(raw));
  }

  @Test
  void cleanAssistantTextKeepsGreetingReplies() {
    assertEquals("Hello! How can I assist you today?",
        FactMemory.cleanAssistantText("</think>\n\nHello! How can I assist you today?<|im_end|>"));
    assertEquals("hello", FactMemory.cleanAssistantText("hello"));
  }

  @Test
  void cleanAssistantTextUsesThinkBodyWhenAnswerEmpty() {
    assertEquals("Tere hommikust",
        FactMemory.cleanAssistantText("<think>\nTere hommikust\n</think>\n\n"));
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
    assertEquals("Hello", FactMemory.streamDisplayText("<think>secret</think>Hello"));
    assertEquals("still thinking", FactMemory.streamDisplayText("<think>still thinking"));
    assertEquals("ok", FactMemory.streamDisplayText("ok<|im_end|>"));
  }

  @Test
  void assistantPartsSplitsThinkAndAnswer() {
    AssistantParts parts = AssistantParts.parse(
        "<think>\nplan\n</think>\n\nTere hommikust<|im_end|>");
    assertEquals("\nplan\n", parts.thinking());
    assertEquals("Tere hommikust", parts.answer());
    assertEquals(false, parts.thinkOpen());
  }

  @Test
  void knowledgeBaseFromAssistantDirectives() {
    List<String> knowledge = new ArrayList<>();
    FactMemory.applyKnowledgeDirectives(
        knowledge,
        "I will remember that Nora Vale is the club president.\nREMEMBER: Nora Vale is the club president");
    assertEquals(List.of("Nora Vale is the club president"), knowledge);

    String system = ChatPrompts.chatSystemWithKnowledge(knowledge);
    assertTrue(system.contains("Knowledge base"));
    assertTrue(system.contains("Nora Vale is the club president")
        || system.toLowerCase().contains("nora vale"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("You are the Assistant"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("User"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("you") || ChatPrompts.CHAT_SYSTEM.contains("your"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.toLowerCase().contains("do not repeat")
        || ChatPrompts.CHAT_SYSTEM.toLowerCase().contains("do not greet"));
    assertEquals("Got it.", ChatPrompts.FACT_SHARE_ACKNOWLEDGMENT);

    FactMemory.applyKnowledgeDirectives(
        knowledge,
        "I will forget that fact.\nFORGET: Nora Vale is the club president");
    assertTrue(knowledge.isEmpty());

    assertEquals(
        "I will remember that Nora Vale is the club president.",
        FactMemory.stripMemoryDirectives(
            "I will remember that Nora Vale is the club president.\nREMEMBER: Nora Vale is the club president"));
  }

  @Test
  void knowledgeIgnoresNoneAndParsesInlineRemember() {
    List<String> knowledge = new ArrayList<>();
    FactMemory.applyKnowledgeDirectives(knowledge, "I don't know. REMEMBER: (none)<|im_ended|");
    assertTrue(knowledge.isEmpty());

    FactMemory.applyKnowledgeDirectives(
        knowledge,
        "I will remember it. REMEMBER: Mira Quinn is a robot designer<|im_end|>");
    assertEquals(List.of("Mira Quinn is a robot designer"), knowledge);

    assertEquals(
        "I don't have information about Mira Quinn.",
        FactMemory.stripMemoryDirectives(
            "I don't have information about Mira Quinn. REMEMBER: (none)<|im_ended|"));
  }

  @Test
  void parseExtractedFactsKeepsCompactUnknownOnly() {
    List<String> fromCompound = FactMemory.parseExtractedFacts("""
        + Alex Rivera is a cartographer born in 1988
        NONE
        """.strip());
    assertTrue(fromCompound.stream().anyMatch(f -> f.toLowerCase().contains("cartographer")));
    assertTrue(fromCompound.stream().anyMatch(f -> f.toLowerCase().contains("1988")));

    assertTrue(FactMemory.parseExtractedFacts("NONE").isEmpty());
    assertTrue(FactMemory.parseExtractedFacts("FACT: NONE").isEmpty());
    assertTrue(FactMemory.parseExtractedFacts("No facts here.").isEmpty());

    List<String> knowledge = new ArrayList<>();
    knowledge.add("User is a cartographer");
    assertTrue(!FactMemory.isUnknownFact(knowledge, "User is a cartographer"));
    assertTrue(FactMemory.isUnknownFact(knowledge, "Riverdale is a fictional town"));

    String payload = ChatPrompts.factExtractUserPayload(
        "Alex Rivera is a cartographer, born in 1988 and it is me",
        knowledge);
    assertTrue(payload.contains("Message:"));
    assertTrue(payload.toLowerCase().contains("already known")
        || payload.contains("User is a cartographer")
        || payload.contains("Message:"));
  }

  @Test
  void shouldSeekRemainingFactsWhenMessageHasMoreClaimsThanFound() {
    String bio = "Hello my name is Igor maznitsa, I am a computer programmer living in estonia, "
        + "I was relocated to Estonia in 2013 with my family but originally I am from "
        + "Saint-Petersburg in Russia, my mother tongue is russian but I prefer english "
        + "for communication";
    assertTrue(MessageClassifier.approximateClaimHints(bio) >= 4);
    assertTrue(FactMemory.shouldSeekRemainingFacts(
        bio, List.of("User's mother tongue is Russian.")));
    assertFalse(FactMemory.shouldSeekRemainingFacts("Hi", List.of("User said hi")));
    assertFalse(FactMemory.shouldSeekRemainingFacts(bio, List.of(
        "a", "b", "c", "d", "e", "f", "g", "h")));
  }

  @Test
  void factsNormalizeUserRoleAndSplitCompounds() {
    assertEquals("User is Alex Rivera", FactMemory.normalizeFact("I am Alex Rivera"));
    assertEquals("User lives in Riverdale", FactMemory.normalizeFact("I live in Riverdale"));
    assertEquals("User is Alex Rivera", FactMemory.normalizeFact("Alex Rivera is the User"));

    List<String> split = FactMemory.splitAtomicFacts(
        "User is a cartographer born in 1988");
    assertTrue(split.stream().anyMatch(f -> f.equalsIgnoreCase("User is a cartographer")));
    assertTrue(split.stream().anyMatch(f -> f.equalsIgnoreCase("User was born in 1988")));

    List<String> multi = FactMemory.parseExtractedFacts("""
        + User is Alex Rivera
        + User is a cartographer
        + User was born in 1988
        """.strip());
    assertEquals(3, multi.size());

    List<String> jammed = FactMemory.parseExtractedFacts(
        "+ User is Alex Rivera + User is a cartographer + User was born in 1988");
    assertEquals(3, jammed.size());

    List<String> legacy = FactMemory.parseExtractedFacts("""
        FACT: User is Mira Quinn
        F| User lives in Riverdale
        """.strip());
    assertEquals(2, legacy.size());

    List<String> knowledge = new ArrayList<>();
    for (String fact : multi) {
      FactMemory.rememberFact(knowledge, fact);
    }
    assertEquals(3, knowledge.size());
  }

  @Test
  void compactFactsDropsOverlappingCompounds() {
    List<String> compact = FactMemory.compactFacts(List.of(
        "User is a computer programmer living in estonia",
        "User is a computer programmer",
        "User lives in Estonia",
        "User is From Saint-Petersburg In Russia",
        "User relocated to Estonia in 2013",
        "User relocated to Estonia from Russia in 2013"
    ));
    assertTrue(compact.stream().anyMatch(f -> f.equalsIgnoreCase("User is a computer programmer")));
    assertTrue(compact.stream().anyMatch(f -> f.equalsIgnoreCase("User lives in Estonia")));
    assertTrue(compact.stream()
        .anyMatch(f -> f.equalsIgnoreCase("User is from Saint-Petersburg in Russia")));
    assertTrue(compact.stream().anyMatch(f ->
        f.equalsIgnoreCase("User relocated to Estonia from Russia in 2013")));
    assertTrue(
        compact.stream().noneMatch(f -> f.equalsIgnoreCase("User relocated to Estonia in 2013")));
    assertTrue(compact.stream().noneMatch(f -> f.toLowerCase().contains("living in")));
    assertEquals(4, compact.size());
  }

  @Test
  void parseMessageAnalysisReadsClassifierOutput() {
    // Bare labels (Gemma-270M often omits "TYPE:")
    assertEquals(MessageIntent.STORE,
        MessageClassifier.parseMessageAnalysis("STORE").intent());
    assertEquals(MessageIntent.QUESTION,
        MessageClassifier.parseMessageAnalysis("QUESTION").intent());
    assertEquals(MessageIntent.CHAT,
        MessageClassifier.parseMessageAnalysis("CHAT").intent());

    assertEquals(MessageIntent.QUESTION,
        MessageClassifier.parseMessageAnalysis("TYPE: QUESTION").intent());
    assertEquals(MessageIntent.QUESTION,
        MessageClassifier.parseMessageAnalysis("type: question").intent());
    assertEquals(MessageIntent.STORE,
        MessageClassifier.parseMessageAnalysis("TYPE: STORE").intent());
    assertEquals(MessageIntent.SKIP,
        MessageClassifier.parseMessageAnalysis("TYPE: SKIP").intent());
    assertEquals(MessageIntent.CHAT,
        MessageClassifier.parseMessageAnalysis("TYPE: CHAT").intent());
    assertEquals(MessageIntent.CHAT,
        MessageClassifier.parseMessageAnalysis("TYPE: ").intent());
    assertEquals(MessageIntent.CHAT,
        MessageClassifier.parseMessageAnalysis("unparseable gibberish").intent());
    assertEquals(MessageIntent.CHAT,
        MessageClassifier.parseMessageAnalysis("").intent());

    MessageAnalysis forget = MessageClassifier.parseMessageAnalysis("""
        TYPE: FORGET
        PROBE: mother tongue
        """.strip());
    assertEquals(MessageIntent.FORGET, forget.intent());
    assertEquals("mother tongue", forget.forgetProbe());

    // Sample classifier outputs for representative user messages
    assertEquals(MessageIntent.QUESTION,
        MessageClassifier.parseMessageAnalysis("TYPE: QUESTION")
            .intent()); // "so guess where did I live in 2000"
    assertEquals(MessageIntent.STORE,
        MessageClassifier.parseMessageAnalysis("TYPE: STORE").intent()); // "add rule that X"
    assertEquals(MessageIntent.SKIP,
        MessageClassifier.parseMessageAnalysis("TYPE: SKIP").intent()); // "hello"
  }

  @Test
  void invalidFactsRejectInterrogatives() {
    assertFalse(FactMemory.isValidFact("User's mother tongue is What"));
    assertFalse(FactMemory.isValidFact("User's mother tongue is who"));
    assertTrue(FactMemory.isValidFact("User's mother tongue is Russian"));
  }

  @Test
  void invalidFactsRejectBareFragmentsAndPlaceholders() {
    assertFalse(FactMemory.isValidFact("2013"));
    assertFalse(FactMemory.isValidFact("Russian"));
    assertFalse(FactMemory.isValidFact("English"));
    assertFalse(FactMemory.isValidFact("St. Petersburg"));
    assertFalse(FactMemory.isValidFact("igor Maznitsa"));
    assertFalse(FactMemory.isValidFact("User lives in …"));
    assertFalse(FactMemory.isValidFact("User's name is …"));
    assertFalse(FactMemory.isValidFact("User lives in ..."));
    assertTrue(FactMemory.isValidFact("User relocated to Estonia in 2013"));
    assertTrue(FactMemory.isValidFact("User's mother tongue is Russian"));
    assertTrue(FactMemory.isValidFact("User prefers English for communication"));
    assertTrue(FactMemory.isValidFact("User is from Saint-Petersburg in Russia"));
    assertTrue(FactMemory.parseExtractedFacts("+ 2013 + Russian + English").isEmpty());
  }

  @Test
  void storeRuleViaRememberFact() {
    List<String> knowledge = new ArrayList<>();
    knowledge.add("User was born in 1975");
    FactMemory.rememberFact(knowledge, "Rule: the month plays role");
    assertTrue(knowledge.contains("Rule: the month plays role"));
    assertEquals(2, knowledge.size());
  }

  @Test
  void freeFormRulesRelyOnLlmGroundingNotHardcodedCues() {
    assertFalse(FactMemory.isGroundedInUserMessage(
        "User is from Saint Petersburg in Russia",
        "birth month plays role for age calculation"));
    assertTrue(FactMemory.isGroundedInUserMessage(
        "Rule: birth month plays a role in age calculation",
        "birth month plays role for age calculation"));
    assertTrue(FactMemory.isGroundedInUserMessage(
        "User was born in 1975",
        "I was born in 1975 and live in Estonia"));
    assertTrue(FactMemory.parseExtractedFacts(
            "+ Rule: birth month plays a role in age calculation").stream()
        .anyMatch(f -> f.toLowerCase().startsWith("rule:") && f.toLowerCase().contains("birth")));
  }

  @Test
  void promoteStoreWhenFactDenseOverridesChatAndSkip() {
    String bio = "Hello my name is Igor maznitsa, I am a computer programmer living in estonia, "
        + "I was relocated to Estonia in 2013 with my family but originally I am from "
        + "Saint-Petersburg in Russia, my mother tongue is russian but I prefer english "
        + "for communication";
    assertTrue(MessageClassifier.looksLikeFactDenseShare(bio));
    assertEquals(
        MessageIntent.STORE,
        MessageClassifier.promoteStoreWhenFactDense(
            new MessageAnalysis(MessageIntent.CHAT, null), bio).intent());
    assertEquals(
        MessageIntent.STORE,
        MessageClassifier.promoteStoreWhenFactDense(
            new MessageAnalysis(MessageIntent.SKIP, null), bio).intent());
    assertEquals(
        MessageIntent.QUESTION,
        MessageClassifier.promoteStoreWhenFactDense(
            new MessageAnalysis(MessageIntent.QUESTION, null), bio).intent());
    assertFalse(MessageClassifier.looksLikeFactDenseShare("hello"));
    assertEquals(
        MessageIntent.CHAT,
        MessageClassifier.promoteStoreWhenFactDense(
            new MessageAnalysis(MessageIntent.CHAT, null), "hello").intent());
  }

  @Test
  void demoteStoreWhenEphemeralTaskRequest() {
    assertTrue(MessageClassifier.looksLikeEphemeralTaskRequest("print list of months"));
    assertEquals(
        MessageIntent.CHAT,
        MessageClassifier.refineClassifiedIntent(
            new MessageAnalysis(MessageIntent.STORE, null),
            "print list of months").intent());
    assertFalse(MessageClassifier.looksLikeEphemeralTaskRequest(
        "add rule that always list months in English"));
    assertEquals(
        MessageIntent.STORE,
        MessageClassifier.refineClassifiedIntent(
            new MessageAnalysis(MessageIntent.STORE, null),
            "add rule that always list months in English").intent());
    assertFalse(FactMemory.isGroundedInUserMessage(
        "Rule: print list of months", "print list of months"));
  }

  @Test
  void longWriteProgramRequestStaysChatNotFactDenseStore() {
    String ask = "write java program to open some text file and read line by line from it "
        + "and print on the screen, code should be safe and high quality";
    assertTrue(MessageClassifier.looksLikeEphemeralTaskRequest(ask));
    assertFalse(MessageClassifier.looksLikeFactDenseShare(ask));
    assertEquals(
        MessageIntent.CHAT,
        MessageClassifier.refineClassifiedIntent(
            new MessageAnalysis(MessageIntent.CHAT, null), ask).intent());
    assertEquals(
        MessageIntent.CHAT,
        MessageClassifier.refineClassifiedIntent(
            new MessageAnalysis(MessageIntent.STORE, null), ask).intent());
  }

  @Test
  void classifyPromptMentionsAllIntents() {
    assertTrue(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.contains("TYPE: STORE"));
    assertTrue(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.contains("TYPE: QUESTION"));
    assertTrue(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.contains("TYPE: FORGET"));
    assertTrue(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.contains("TYPE: SKIP"));
    assertTrue(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.contains("TYPE: CHAT"));
    assertTrue(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.contains("PROBE:"));
    String payload = ChatPrompts.messageClassifyUserPayload("so guess where did I live in 2000");
    assertTrue(payload.contains("Message:"));
    assertTrue(payload.contains("so guess where did I live in 2000"));
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
    // gemmaStyle: (1+w) * x / rms — with w=[0,0.5] scales are 1 and 1.5
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
      String chat = tok.applyChatTemplate(
          List.of(Map.of("role", "user", "content", "hi")), true, true);
      assertTrue(chat.startsWith("<bos>") || chat.contains("<start_of_turn>user"));
      assertTrue(chat.contains("<start_of_turn>model"));
      assertTrue(chat.contains("<think>"));
      assertTrue(chat.contains("User intent"));
      assertFalse(tok.applyChatTemplate(
          List.of(Map.of("role", "user", "content", "hi")), true, false).contains("<think>"));
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
    String chatThink = tok.applyChatTemplate(
        List.of(Map.of("role", "user", "content", "hello")), true, true);
    assertTrue(chatThink.contains("<start_of_turn>user"));
    assertTrue(chatThink.contains("<start_of_turn>model"));
    assertTrue(chatThink.contains("<think>"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("<think>"));
    assertTrue(ChatPrompts.GEMMA_THINK_SCAFFOLD.contains("Reply plan"));
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
  void needsExtractRepairWhenSparseOrFragmentHeavy() {
    String bio = "Hello my name is Igor maznitsa, I am a computer programmer living in estonia, "
        + "I was relocated to Estonia in 2013 with my family but originally I am from "
        + "Saint-Petersburg in Russia, my mother tongue is russian but I prefer english "
        + "for communication";
    assertTrue(FactMemory.needsExtractRepair(
        "+ 2013 + Russian + English + User prefers English for communication",
        List.of("User prefers English for communication"),
        bio));
    assertTrue(FactMemory.needsExtractRepair("+ 2013 + Russian", List.of(), bio));
    assertFalse(FactMemory.needsExtractRepair(
        "+ User's name is Igor Maznitsa\n+ User is a computer programmer\n"
            + "+ User lives in Estonia\n+ User prefers English for communication",
        List.of(
            "User's name is Igor Maznitsa",
            "User is a computer programmer",
            "User lives in Estonia",
            "User prefers English for communication"),
        bio));
    assertFalse(FactMemory.needsExtractRepair("NONE", List.of(), "hi"));
  }

  @Test
  void sharedPromptsHaveNoFewShotBiographies() {
    assertFalse(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.toLowerCase().contains("ada"));
    assertFalse(ChatPrompts.FACT_EXTRACT_SYSTEM.toLowerCase().contains("london"));
    assertFalse(ChatPrompts.FACT_EXTRACT_SYSTEM.toLowerCase().contains("ada"));
    assertTrue(ChatPrompts.MESSAGE_CLASSIFY_SYSTEM.contains("TYPE: STORE"));
    assertTrue(ChatPrompts.FACT_EXTRACT_SYSTEM.contains("NONE"));
    assertTrue(ChatPrompts.FACT_EXTRACT_REPAIR_SYSTEM.contains("NONE"));
    assertTrue(ChatPrompts.FACT_EXTRACT_SYSTEM.toLowerCase().contains("if/then")
        || ChatPrompts.FACT_EXTRACT_SYSTEM.contains("Rule:"));
    assertTrue(ChatPrompts.FACT_EXTRACT_ASSISTANT_SEED.contains("<think>"));
    assertTrue(ChatPrompts.FACT_EXTRACT_ASSISTANT_SEED.contains("+"));
    assertTrue(ChatPrompts.CHAT_SYSTEM.contains("Assistant"));
  }

  @Test
  void topicOutlineDetectionAndRejection() {
    String outline = "+ 1 Polite greeting + 2 Greetings + 3 Information about the relocation";
    assertTrue(FactMemory.looksLikeTopicOutline(outline));
    assertTrue(FactMemory.parseExtractedFacts(outline).isEmpty());
    assertFalse(FactMemory.isValidFact("Information about the relocation"));
    assertFalse(FactMemory.isValidFact("1 Polite greeting"));

    String numberedFacts = "+ 1 User's name is Igor Maznitsa + 2 User is a computer programmer "
        + "+ 3 User lives in Estonia";
    assertFalse(FactMemory.looksLikeTopicOutline(numberedFacts));
    assertFalse(FactMemory.needsExtractRepair(
        numberedFacts,
        List.of(
            "User's name is Igor Maznitsa",
            "User is a computer programmer",
            "User lives in Estonia"),
        "Hello my name is Igor"));
    assertTrue(FactMemory.parseExtractedFacts(numberedFacts).stream()
        .anyMatch(f -> f.toLowerCase().contains("igor")));
  }

  @Test
  void stripChatMarkupRemovesGemmaTurnTokens() {
    assertEquals("", AssistantParts.stripChatMarkup("<end_of_turn>"));
    assertEquals("Hi", AssistantParts.stripChatMarkup("Hi<end_of_turn>"));
    assertEquals("plan", AssistantParts.stripChatMarkup("<start_of_turn>model\nplan<end_of_turn>"));
  }

  @Test
  void bareGreetingBecomesSkipNotStore() {
    assertTrue(MessageClassifier.looksLikeBareGreeting("hello"));
    assertTrue(MessageClassifier.looksLikeBareGreeting("Hi!"));
    assertTrue(MessageClassifier.looksLikeBareGreeting("привет"));
    assertFalse(MessageClassifier.looksLikeBareGreeting(
        "Hello my name is Igor maznitsa, I am a computer programmer"));
    assertEquals(
        MessageIntent.SKIP,
        MessageClassifier.refineClassifiedIntent(
            new MessageAnalysis(MessageIntent.STORE, null), "hello").intent());
  }

  @Test
  void parseKeepScanReadsLlmGate() {
    assertTrue(FactMemory.parseKeepScan("KEEP: yes"));
    assertTrue(FactMemory.parseKeepScan("yes"));
    assertFalse(FactMemory.parseKeepScan("KEEP: no"));
    assertFalse(FactMemory.parseKeepScan("KEEP: 0"));
    assertFalse(FactMemory.parseKeepScan("no"));
    assertFalse(FactMemory.parseKeepScan(""));
    assertTrue(ChatPrompts.FACT_SCAN_SYSTEM.contains("KEEP:"));
  }

  @Test
  void childhoodPossessionIsLastingFactCue() {
    String msg = "in my childhooh I had zxspectrum";
    assertTrue(MessageClassifier.hasFirstPersonMemoryCue(msg));
    assertTrue(MessageClassifier.looksLikePersonalFactShare(msg));
    assertTrue(FactMemory.hasLastingContentCues(msg));
    assertTrue(FactMemory.isValidFact("User had a ZX Spectrum in childhood"));
    assertTrue(FactMemory.isValidFact("User had zxspectrum"));
    assertTrue(FactMemory.isGroundedInUserMessage(
        "User had zxspectrum", "in my childhood I had zxspectrum"));
  }

  @Test
  void looksLikeLastingRuleDetectsIfThen() {
    assertTrue(FactMemory.looksLikeLastingRule(
        "if the user asks about age then use birth year from memory"));
    assertTrue(FactMemory.looksLikeLastingRule("always answer briefly"));
    assertFalse(FactMemory.looksLikeLastingRule("hello"));
  }

  @Test
  void parseExtractedFactsFromPlusLines() {
    List<String> facts = FactMemory.parseExtractedFacts("""
        + User's name is Igor Maznitsa
        + User is a computer programmer
        NONE
        """.strip());
    assertTrue(facts.stream().anyMatch(f -> f.toLowerCase().contains("igor")));
    assertTrue(facts.stream().anyMatch(f -> f.toLowerCase().contains("programmer")));

    List<String> seeded = FactMemory.parseExtractedFacts("+ User's name is Igor Maznitsa");
    assertTrue(seeded.stream().anyMatch(f -> f.toLowerCase().contains("igor")), seeded.toString());
  }

  @Test
  void spokenUserFactAndIdentityRewrite() {
    assertEquals("Your name is Igor Maznitsa",
        ChatPrompts.toSpokenUserFact("User's name is Igor Maznitsa"));
    assertEquals("You are a computer programmer",
        ChatPrompts.toSpokenUserFact("User is a computer programmer"));
    assertEquals(
        "Your name is Igor Maznitsa.",
        FactMemory.rewriteMistakenFirstPersonIdentity(
            "I am Igor Maznitsa.",
            List.of("User's name is Igor Maznitsa")));
    assertEquals(
        "Your name is Igor Maznitsa.",
        FactMemory.rewriteMistakenFirstPersonIdentity(
            "Hello, my name is Igor Maznitsa.",
            List.of("User's name is Igor Maznitsa")));
    assertEquals(
        "You live in Estonia.",
        FactMemory.rewriteMistakenFirstPersonIdentity(
            "I am located in Estonia.",
            List.of("User lives in Estonia")));
    assertEquals(
        "You live in Estonia.",
        FactMemory.rewriteMistakenFirstPersonIdentity(
            "I live in Estonia.",
            List.of("User lives in Estonia")));
  }

  @Test
  void refinePromotesQuestionsAndIgnoresNonAsciiTypeGarbage() {
    assertTrue(MessageClassifier.looksLikeQuestion("who am i"));
    assertTrue(MessageClassifier.looksLikeQuestion("where do i live"));
    assertEquals(
        MessageIntent.QUESTION,
        MessageClassifier.refineClassifiedIntent(
            new MessageAnalysis(MessageIntent.CHAT, null), "who am i").intent());
    assertEquals(
        MessageIntent.CHAT,
        MessageClassifier.parseMessageAnalysis("TYPE: فِي الْЕНَا").intent());
    assertEquals(
        MessageIntent.QUESTION,
        MessageClassifier.parseMessageAnalysis("TYPE: QUESTION").intent());
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
