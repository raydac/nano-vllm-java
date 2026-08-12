package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.LlmAdvisor;
import com.igormaznitsa.nanollvm.llm.LlmAdvisorMixer;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.DenseRagIndex;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagIndex;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.BundledRag;
import com.igormaznitsa.nanollvm.samples.utils.SampleAdvisorPrompts;
import com.igormaznitsa.nanollvm.samples.utils.SampleChatPrompts;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Consumer;

final class ExampleSessionSupport {

  static final int MAX_NEW_TOKENS = 768;
  static final int COMPACT_DEMO_MAX_NEW_TOKENS = 256;
  static final int RAG_MAX_TOKENS_DEFAULT = 768;
  static final int RAG_MAX_TOKENS_GEMMA = 128;
  static final int RAG_TOP_K_DEFAULT = 4;
  static final int RAG_TOP_K_GEMMA = 2;
  static final int RAG_CONTEXT_CHARS_DEFAULT = 3500;
  static final int RAG_CONTEXT_CHARS_GEMMA = 900;
  static final int EMBED_PREVIEW = 8;

  private ExampleSessionSupport() {
  }

  static List<ModelChoice> catalog() {
    List<ModelChoice> choices = new ArrayList<>();
    choices.add(choice(
      "Qwen3-0.6B (chat, safetensors)",
      BundledModels.QWEN3_0_6B,
      "Run models/download-qwen3-0.6b.sh"));
    choices.add(choice(
      "Gemma3-270M (chat, safetensors)",
      BundledModels.GEMMA3_270M,
      "Run models/download-gemma3-270m.sh (HF license + HF_TOKEN)"));
    choices.add(choice(
      "LFM2.5-2.6B Q4_K_M (chat, gguf, ~16g heap)",
      BundledModels.LFM2_5_2_6B_GGUF,
      "Run models/download-lfm2.5-2.6b-gguf.sh"));
    choices.add(choice(
      "SmolLM2-135M-Instruct-ONNX (chat, onnx)",
      BundledModels.SMOLLM2_135M_INSTRUCT_ONNX,
      "Run models/download-smollm2-135m-instruct-onnx.sh"));
    choices.add(choice(
      "Tiny-LLM-ONNX (base, onnx ~10M)",
      BundledModels.TINY_LLM_ONNX,
      "Run models/download-tiny-llm-onnx.sh"));
    choices.add(choice(
      "gte-small Q2_K (embeddings, gguf)",
      BundledModels.GTE_SMALL_GGUF,
      "Run models/download-gte-small-gguf.sh"));
    return List.copyOf(choices);
  }

  private static ModelChoice choice(
    final String label,
    final String bundledPath,
    final String missingHint
  ) {
    Optional<Path> path = BundledModels.find(bundledPath);
    return new ModelChoice(label, path, missingHint, path.isEmpty());
  }

  static boolean isGemmaPath(final Path path) {
    return path.toString().toLowerCase(Locale.ROOT).contains("gemma");
  }

  static boolean isGgufPath(final Path path) {
    return path.toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
  }

  static boolean isTinyLlmPath(final Path path) {
    return path.toString().toLowerCase(Locale.ROOT).contains("tiny-llm");
  }

  static RagSetup prepareRagSetup(
    final Path ragRoot,
    final RagMode ragMode,
    final Path gtePath,
    final boolean gemmaPath,
    final LlmListener status,
    final Consumer<String> info
  ) {
    info.accept("Preparing RAG corpus from " + ragRoot);
    RagLoadOptions options =
      gemmaPath ? RagLoadOptions.forTinyModels() : RagLoadOptions.defaults();
    PreparedRag lexical = RagFactory.tryMake(ragRoot, options, status).orElse(null);
    if (lexical == null) {
      return null;
    }

    return switch (ragMode) {
      case NONE -> new RagSetup(null, null);
      case BM25 -> {
        info.accept(
          "RAG: BM25 over " + BundledRag.ragRoot() + " (" + lexical.size() + " chunks)");
        yield new RagSetup(lexical, null);
      }
      case DENSE -> {
        Path gte = requireNonNull(gtePath, "gte-small GGUF path");
        info.accept("Loading RAG embedding model from " + gte);
        LlmModel embed = LlmModelFactory.make(gte, status);
        try {
          DenseRagIndex dense = DenseRagIndex.of(lexical, embed);
          info.accept(
            "RAG: dense embeddings over " + BundledRag.ragRoot()
              + " (" + dense.size() + " chunks; encoder " + embed.architectureName() + ")");
          yield new RagSetup(dense, embed);
        } catch (RuntimeException | Error failed) {
          embed.close();
          throw failed;
        }
      }
      case HYBRID -> {
        Path gte = requireNonNull(gtePath, "gte-small GGUF path");
        info.accept("Loading RAG embedding model from " + gte);
        LlmModel embed = LlmModelFactory.make(gte, status);
        try {
          RagIndex hybrid = RagFactory.withEmbeddings(lexical, embed);
          info.accept(
            "RAG: hybrid BM25+dense over " + BundledRag.ragRoot()
              + " (" + hybrid.size() + " chunks; encoder " + embed.architectureName() + ")");
          yield new RagSetup(hybrid, embed);
        } catch (RuntimeException | Error failed) {
          embed.close();
          throw failed;
        }
      }
    };
  }

  static CausalBundle openCausal(
    final LlmModel model,
    final RagSetup ragSetup,
    final LlmListener status,
    final Consumer<String> info
  ) {
    LLM.Builder builder = LLM.builder(model)
      .maxNumSeqs(4)
      .maxModelLen(2048)
      .listen(status);

    String arch = model.architectureName();
    String system = SampleChatPrompts.forDemo(arch, model.tokenizer());
    boolean advisorsOn = false;
    switch (arch) {
      case ARCH_GEMMA3 -> {
        builder.advisors(
          LlmAdvisorMixer.defaults(),
          LlmAdvisor.builder().name("Practical").prompt(SampleAdvisorPrompts.ROLE_PRACTICAL)
            .build(),
          LlmAdvisor.builder().name("Abstract").prompt(SampleAdvisorPrompts.ROLE_ABSTRACT)
            .build(),
          LlmAdvisor.builder().name("Consequence").prompt(SampleAdvisorPrompts.ROLE_CONSEQUENCE)
            .build());
        advisorsOn = true;
        info.accept("Advisors: Practical, Abstract, Consequence for Gemma.");
      }
      case ARCH_QWEN3 -> {
        system = SampleAdvisorPrompts.withAdvisorAddon(system);
        builder.advisors(
          LlmAdvisorMixer.defaults(),
          LlmAdvisor.builder().name("Practical").prompt(SampleAdvisorPrompts.ROLE_PRACTICAL)
            .build(),
          LlmAdvisor.builder().name("Abstract").prompt(SampleAdvisorPrompts.ROLE_ABSTRACT)
            .build());
        advisorsOn = true;
        info.accept("Advisors: Practical, Abstract for Qwen.");
      }
      case ARCH_LFM2 -> info.accept("Advisors: off for LFM.");
      case null, default -> info.accept("Advisors: off (architecture " + arch + ").");
    }
    if (advisorsOn) {
      builder.advisorNoteFilter(note -> !SampleChatPrompts.isSetupBoilerplate(note));
    }
    builder.systemPrompt(system);

    LLM llm = builder.build();
    boolean turnBased = llm.tokenizer().isTurnBasedChat();
    int modelLen = llm.config().maxModelLen();
    int maxNew = Math.clamp(modelLen / 2, 32, MAX_NEW_TOKENS);
    if (isCompactDemoModel(llm)) {
      maxNew = Math.min(maxNew, COMPACT_DEMO_MAX_NEW_TOKENS);
    }

    RagIndex ragIndex = ragSetup.index();
    if (ragIndex != null) {
      int maxTokens = Math.clamp(modelLen / 4, 32,
        turnBased ? RAG_MAX_TOKENS_GEMMA : RAG_MAX_TOKENS_DEFAULT);
      if (isCompactDemoModel(llm)) {
        maxTokens = Math.min(maxTokens, COMPACT_DEMO_MAX_NEW_TOKENS);
      }
      int maxNoHits =
        Math.clamp(maxTokens, isCompactDemoModel(llm) ? maxTokens : modelLen / 2, maxNew);
      int maxContextChars = Math.clamp((modelLen - maxTokens - 64) * 3, 256,
        turnBased ? RAG_CONTEXT_CHARS_GEMMA : RAG_CONTEXT_CHARS_DEFAULT);
      if (isCompactDemoModel(llm)) {
        maxContextChars = Math.min(maxContextChars, 1200);
      }
      RagSession rag = llm.rag(ragIndex, maxTokens)
        .maxTokensWhenNoHits(maxNoHits)
        .topK(turnBased ? RAG_TOP_K_GEMMA : RAG_TOP_K_DEFAULT)
        .maxContextChars(maxContextChars)
        .isolateGeneration(turnBased)
        .enableThinking(llm.tokenizer().invitesThinking())
        .sampling(new SamplingParams(
          turnBased ? 0.1f : 0.4f,
          maxTokens,
          false,
          turnBased ? 30 : 0,
          turnBased ? 0.8f : 0.85f));
      if (turnBased) {
        rag.chat().recoverUnusableAnswers(true)
          .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
          .unusableAnswerFallback("What would you like to explore?");
      }
      return new CausalBundle(llm, rag, null, true);
    }

    ChatSession chat = llm.chat(SampleChatPrompts.samplingForDemo(llm.tokenizer(), maxNew));
    if (turnBased) {
      chat.recoverUnusableAnswers(true)
        .unusableAnswer(SampleChatPrompts::isSetupBoilerplate)
        .unusableAnswerFallback("What would you like to explore?");
    }
    return new CausalBundle(llm, null, chat, false);
  }

  static boolean isCompactDemoModel(final LLM llm) {
    var hf = llm.config().hfConfig();
    return hf.numHiddenLayers() <= 32 && hf.hiddenSize() <= 768;
  }

  static double l2Norm(final float[] vector) {
    double sum = 0.0;
    for (float v : vector) {
      sum += (double) v * v;
    }
    return Math.sqrt(sum);
  }

  static float cosine(final float[] a, final float[] b) {
    float dot = 0f;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  static String preview(final float[] vector) {
    int n = Math.min(EMBED_PREVIEW, vector.length);
    StringBuilder sb = new StringBuilder("[");
    for (int i = 0; i < n; i++) {
      if (i > 0) {
        sb.append(", ");
      }
      sb.append(String.format(Locale.ROOT, "%.4f", vector[i]));
    }
    if (vector.length > n) {
      sb.append(", …");
    }
    return sb.append(']').toString();
  }

  enum RagMode {
    NONE("None (plain chat)"),
    BM25("BM25 lexical"),
    DENSE("Dense embeddings (gte-small)"),
    HYBRID("Hybrid BM25 + dense");

    private final String label;

    RagMode(final String label) {
      this.label = label;
    }

    String label() {
      return this.label;
    }
  }

  record ModelChoice(String label, Optional<Path> path, String missingHint, boolean missing) {
    String display() {
      return this.missing ? this.label + "  [not downloaded]" : this.label;
    }

    Path requirePath() {
      return this.path.orElseThrow(() -> new IllegalStateException(this.missingHint));
    }

    @Override
    public String toString() {
      return this.label;
    }
  }

  record RagSetup(RagIndex index, LlmModel embeddingModel) {
  }

  record CausalBundle(LLM llm, RagSession rag, ChatSession chat, boolean ragMode)
    implements AutoCloseable {
    @Override
    public void close() {
      this.llm.close();
    }
  }
}
