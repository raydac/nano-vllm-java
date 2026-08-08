package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL_LEGACY;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_COLOR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_COLOR_LEGACY;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL_LEGACY;

import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.LlmAdvisor;
import com.igormaznitsa.nanollvm.llm.LlmAdvisorMixer;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.prompts.AdvisorPrompts;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.BundledRag;
import com.igormaznitsa.nanollvm.samples.utils.OrderedConsole;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Interactive chat using {@link RagSession} when {@code ./rag} (or override) has indexable
 * documents, otherwise plain {@link ChatSession}.
 */
public final class Example {

  private static final int MAX_NEW_TOKENS = 768;
  private static final int RAG_MAX_TOKENS_DEFAULT = 768;
  private static final int RAG_MAX_TOKENS_GEMMA = 128;
  private static final int RAG_TOP_K_DEFAULT = 4;
  private static final int RAG_TOP_K_GEMMA = 2;
  private static final int RAG_CONTEXT_CHARS_DEFAULT = 3500;
  private static final int RAG_CONTEXT_CHARS_GEMMA = 900;

  private Example() {
  }

  public static void main(final String[] args) throws Exception {
    OrderedConsole console = new OrderedConsole(System.out, System.err);
    try (BufferedReader in = new BufferedReader(
        new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
      Path path = resolveModel(args, in, console);
      if (path == null) {
        return;
      }

      boolean gemmaPath = path.toString().toLowerCase(Locale.ROOT).contains("gemma");
      boolean ggufPath = path.toString().toLowerCase(Locale.ROOT).endsWith(".gguf")
        || path.toString().toLowerCase(Locale.ROOT).contains("lfm2");
      Optional<PreparedRag> preparedRag = loadPreparedRag(gemmaPath, console);

      console.printlnInfo("Loading model from " + path);
      console.printlnInfo(
        "Architecture auto-detects from config.json / GGUF metadata "
          + "(override: -Dnanovllm.arch=qwen3|gemma3|lfm2).");
      console.printlnInfo(
        "CPU matmul: " + Runtime.getRuntime().availableProcessors()
          + " threads from Runtime (override: -Dnanovllm.cpu.threads=N).");
      if (ggufPath) {
        console.printlnInfo(
          "GGUF/LFM2: weights stay packed (dequant on matmul). For float32 at load: "
            + "LlmModelFactory.make(path, io, true).");
      }
      if (preparedRag.isPresent()) {
        PreparedRag rag = preparedRag.get();
        console.printlnInfo("RAG: prepared BM25 over " + BundledRag.ragRoot()
            + " (" + rag.size() + " chunks, shared index)");
        console.printlnInfo("Ask about the docs in rag/ (engine, models, Nile, capitals, …).");
      } else {
        console.printlnInfo("RAG: no usable corpus at " + BundledRag.ragRoot() + " — plain chat.");
      }
      console.println("Type a message and press Enter. Commands: /exit  /quit  /clear");
      console.println(
        "Answer/prompts on stdout; thinking, debug, and load/status on stderr (red in many IDEs).");
      console.println(
        "After each turn: engine tok/s from GenerationStats (main generate; excludes advisors / RAG prep).");
      console.println();

      boolean color = useColor();
      LlmListener status = statusTo(console);
      LlmModel model = LlmModelFactory.make(path, status);
      LLM.Builder builder = LLM.builder(model)
        .enforceEager(true)
        .maxNumSeqs(4)
        .maxModelLen(2048)
        .listen(status);
      configureDemoAdvisors(builder, model.architectureName(), console);
      try (LLM llm = builder.build()) {
        if (preparedRag.isPresent()) {
          runRagChat(in, llm, preparedRag.get(), color, console);
        } else {
          runPlainChat(in, llm, color, console);
        }
      }
    }
  }

  private static LlmListener statusTo(final OrderedConsole console) {
    return (source, event) -> {
      switch (event.kind()) {
        case STATUS_INFO -> console.printInfo(event.text());
        case STATUS_PROGRESS -> console.print(event.text());
        default -> {
        }
      }
    };
  }

  private static void configureDemoAdvisors(
    final LLM.Builder builder,
    final String arch,
    final OrderedConsole console
  ) {
    if (ARCH_GEMMA3.equals(arch)) {
      builder.advisors(
        LlmAdvisorMixer.defaults(),
        LlmAdvisor.builder().name("Practical").prompt(AdvisorPrompts.ROLE_PRACTICAL).build(),
        LlmAdvisor.builder().name("Abstract").prompt(AdvisorPrompts.ROLE_ABSTRACT).build(),
        LlmAdvisor.builder().name("Consequence").prompt(AdvisorPrompts.ROLE_CONSEQUENCE).build());
      console.printlnInfo("Advisors: Practical, Abstract, Consequence for Gemma.");
      return;
    }
    if (ARCH_QWEN3.equals(arch)) {
      builder.advisors(
        LlmAdvisorMixer.defaults(),
        LlmAdvisor.builder().name("Practical").prompt(AdvisorPrompts.ROLE_PRACTICAL).build(),
        LlmAdvisor.builder().name("Abstract").prompt(AdvisorPrompts.ROLE_ABSTRACT).build());
      console.printlnInfo("Advisors: Practical, Abstract for Qwen.");
      return;
    }
    if (ARCH_LFM2.equals(arch)) {
      console.printlnInfo("Advisors: off for LFM.");
      return;
    }
    console.printlnInfo("Advisors: off (architecture " + arch + ").");
  }

  private static Optional<PreparedRag> loadPreparedRag(
    final boolean tinyModel,
    final OrderedConsole console
  ) {
    Optional<Path> root = BundledRag.find();
    if (root.isEmpty()) {
      return Optional.empty();
    }
    console.printlnInfo("Preparing RAG corpus from " + root.get());
    RagLoadOptions options =
      tinyModel ? RagLoadOptions.forTinyModels() : RagLoadOptions.defaults();
    Optional<PreparedRag> prepared =
      RagFactory.tryMake(root.get(), options, statusTo(console));
    if (prepared.isEmpty()) {
      console.printlnInfo("RAG: no documents in " + root.get() + " — plain chat.");
    }
    return prepared;
  }

  private static void runRagChat(
      final BufferedReader in,
      final LLM llm,
      final PreparedRag prepared,
      final boolean color,
      final OrderedConsole console
  ) throws Exception {
    boolean gemma = llm.tokenizer().isGemmaChat();
    int maxTokens = gemma ? RAG_MAX_TOKENS_GEMMA : RAG_MAX_TOKENS_DEFAULT;
    PrintStream answerOut = console.stream();
    PrintStream thinkOut = console.infoStream();
    RagSession rag = llm.rag(prepared, maxTokens)
        .maxTokensWhenNoHits(gemma ? MAX_NEW_TOKENS : RAG_MAX_TOKENS_DEFAULT)
        .topK(gemma ? RAG_TOP_K_GEMMA : RAG_TOP_K_DEFAULT)
        .maxContextChars(gemma ? RAG_CONTEXT_CHARS_GEMMA : RAG_CONTEXT_CHARS_DEFAULT)
      .enableThinking(llm.tokenizer().invitesThinking())
      .sampling(new SamplingParams(gemma ? 0.1f : 0.4f, maxTokens, false, gemma ? 30 : 0,
        gemma ? 0.8f : 0.85f))
      .streamTo(thinkOut, answerOut, color);
    TurnSpeedTracker speed = new TurnSpeedTracker(console);

    while (true) {
      console.print("rag?> ");
      String line = in.readLine();
      if (line == null) {
        console.println();
        speed.printSessionAverage();
        break;
      }
      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isExit(user)) {
        speed.printSessionAverage();
        break;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        rag.clear();
        console.println("(conversation cleared; RAG index kept)");
        continue;
      }

      ChatReply reply = rag.send(user);
      speed.recordAndPrint(reply);
      printRetrievalSummary(rag.lastHits(), console);
      console.println();
    }
  }

  private static void printRetrievalSummary(
    final java.util.List<RagHit> hits,
    final OrderedConsole console
  ) {
    if (hits.isEmpty()) {
      console.println("(no RAG hits)");
      return;
    }
    String sources = hits.stream()
        .map(hit -> Path.of(hit.chunk().source()).getFileName().toString())
        .distinct()
        .collect(Collectors.joining(", "));
    console.println("(retrieved " + hits.size() + " chunk(s): " + sources + ")");
  }

  private static void runPlainChat(
    final BufferedReader in,
    final LLM llm,
    final boolean color,
    final OrderedConsole console
  ) throws Exception {
    PrintStream answerOut = console.stream();
    PrintStream thinkOut = console.infoStream();
    ChatSession chat = llm.chat(MAX_NEW_TOKENS)
      .streamTo(thinkOut, answerOut, color);
    TurnSpeedTracker speed = new TurnSpeedTracker(console);

    while (true) {
      console.print("?> ");
      String line = in.readLine();
      if (line == null) {
        console.println();
        speed.printSessionAverage();
        break;
      }
      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isExit(user)) {
        speed.printSessionAverage();
        break;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        chat.clear();
        console.println("(conversation cleared)");
        continue;
      }

      ChatReply reply = chat.send(user);
      speed.recordAndPrint(reply);
      console.println();
    }
  }

  static Path resolveModel(final String[] args, final BufferedReader in) throws Exception {
    return resolveModel(args, in, new OrderedConsole(System.out, System.err));
  }

  private static Path resolveModel(
    final String[] args,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    if (hasExplicitModel(args)) {
      return BundledModels.resolveDefault(args);
    }
    return promptModelChoice(in, console);
  }

  private static Path promptModelChoice(
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    var qwen = BundledModels.find(BundledModels.QWEN3_0_6B);
    var gemma = BundledModels.find(BundledModels.GEMMA3_270M);
    var lfm2 = BundledModels.find(BundledModels.LFM2_5_2_6B_GGUF);

    while (true) {
      console.println("Select model to load:");
      console.println("  1) Qwen3-0.6B" + (qwen.isPresent() ? "" : "  [not downloaded]"));
      console.println("  2) Gemma3-270M" + (gemma.isPresent() ? "" : "  [not downloaded]"));
      console.println(
        "  3) LFM2.5-2.6B GGUF Q4_K_M" + (lfm2.isPresent() ? "" : "  [not downloaded]")
          + "  (~16g heap)");
      console.println("  4) Exit");
      console.print("Choice [1-4]: ");
      String line = in.readLine();
      if (line == null) {
        return null;
      }
      switch (line.strip()) {
        case "1" -> {
          return qwen.orElseThrow(() -> new IllegalStateException(
              "Qwen3-0.6B not found. Run models/download-qwen3-0.6b.sh"));
        }
        case "2" -> {
          return gemma.orElseThrow(() -> new IllegalStateException(
              "Gemma3-270M not found. Run models/download-gemma3-270m.sh (HF license + HF_TOKEN)"));
        }
        case "3" -> {
          return lfm2.orElseThrow(() -> new IllegalStateException(
            "LFM2.5 GGUF not found. Run models/download-lfm2.5-2.6b-gguf.sh "
              + "(heap: .mvn/jvm.config -Xmx16g)"));
        }
        case "4", "q", "quit", "exit" -> {
          console.println("Bye.");
          return null;
        }
        default -> console.println("Enter 1, 2, 3, or 4.");
      }
    }
  }

  private static boolean hasExplicitModel(final String[] args) {
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      return true;
    }
    String prop = NanoLlvmProps.systemProperty(PROP_MODEL, PROP_MODEL_LEGACY);
    if (prop != null && !prop.isBlank()) {
      return true;
    }
    String env = NanoLlvmProps.environment(ENV_MODEL, ENV_MODEL_LEGACY);
    return env != null && !env.isBlank();
  }

  private static final class TurnSpeedTracker {
    private final OrderedConsole console;
    private long totalTokens;
    private long totalNanos;
    private int turns;

    TurnSpeedTracker(final OrderedConsole console) {
      this.console = console;
    }

    void recordAndPrint(final ChatReply reply) {
      var stats = reply.stats();
      int tokens = stats.completionTokens();
      long nanos = Math.max(1L, stats.elapsedNanos());
      this.totalTokens += tokens;
      this.totalNanos += nanos;
      this.turns++;

      double seconds = nanos / 1e9;
      this.console.printf(
        Locale.ROOT,
        "(turn %d: %d tok in %.2fs → %.1f tok/s; session avg %.1f tok/s)%n",
        this.turns,
        tokens,
        seconds,
        stats.completionTokensPerSecond(),
        this.sessionTokPerSec());
    }

    void printSessionAverage() {
      if (this.turns == 0) {
        return;
      }
      this.console.printf(
        Locale.ROOT,
        "(session: %d turn(s), %d tok, %.2fs → avg %.1f tok/s)%n",
        this.turns,
        this.totalTokens,
        this.totalNanos / 1e9,
        this.sessionTokPerSec());
    }

    private double sessionTokPerSec() {
      return this.totalTokens / (this.totalNanos / 1e9);
    }
  }

  private static boolean isExit(final String user) {
    String t = user.toLowerCase(Locale.ROOT);
    return t.equals("/exit") || t.equals("/quit") || t.equals("exit") || t.equals("quit");
  }

  private static boolean useColor() {
    if (System.getenv("NO_COLOR") != null) {
      return false;
    }
    String color = NanoLlvmProps.systemProperty(PROP_COLOR, PROP_COLOR_LEGACY);
    return !"false".equalsIgnoreCase(color);
  }
}
