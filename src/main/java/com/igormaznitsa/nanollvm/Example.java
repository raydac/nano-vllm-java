package com.igormaznitsa.nanollvm;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_COLOR;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_MODEL;

import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.llm.EngineIo;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.utils.BundledModels;
import com.igormaznitsa.nanollvm.utils.BundledRag;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Interactive chat using {@link RagSession} when {@code ./rag} (or override) is present,
 * otherwise plain {@link ChatSession}.
 */
public final class Example {

  private static final int MAX_NEW_TOKENS = 768;
  private static final int RAG_MAX_TOKENS_DEFAULT = 768;
  private static final int RAG_MAX_TOKENS_GEMMA = 96;
  private static final int RAG_TOP_K_DEFAULT = 4;
  private static final int RAG_TOP_K_GEMMA = 2;
  private static final int RAG_CONTEXT_CHARS_DEFAULT = 3500;
  private static final int RAG_CONTEXT_CHARS_GEMMA = 700;

  private Example() {
  }

  public static void main(final String[] args) throws Exception {
    try (BufferedReader in = new BufferedReader(
        new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
      Path path = resolveModel(args, in);
      if (path == null) {
        return;
      }

      boolean gemmaPath = path.toString().toLowerCase(Locale.ROOT).contains("gemma");
      boolean ggufPath = path.toString().toLowerCase(Locale.ROOT).endsWith(".gguf")
        || path.toString().toLowerCase(Locale.ROOT).contains("lfm2");
      Optional<PreparedRag> preparedRag = loadPreparedRag(gemmaPath);

      System.out.println("Loading model from " + path);
      System.out.println(
        "Architecture auto-detects from config.json / GGUF metadata "
          + "(override: -Dnanovllm.arch=qwen3|gemma3|lfm2).");
      System.out.println(
        "CPU matmul: " + Runtime.getRuntime().availableProcessors()
          + " threads from Runtime (override: -Dnanovllm.cpu.threads=N).");
      if (ggufPath) {
        System.out.println(
          "GGUF/LFM2: weights expand to float32 — default heap is -Xmx16g via .mvn/jvm.config.");
      }
      if (preparedRag.isPresent()) {
        PreparedRag rag = preparedRag.get();
        System.out.println("RAG: prepared BM25 over " + BundledRag.ragRoot()
            + " (" + rag.size() + " chunks, shared index)");
        System.out.println("Ask about the docs in rag/ (engine, models, Nile, capitals, …).");
      } else {
        System.out.println("RAG: no corpus at " + BundledRag.ragRoot() + " — plain chat.");
      }
      System.out.println("Type a message and press Enter. Commands: /exit  /quit  /clear");
      System.out.println("Thinking → stderr (dim cyan); reply → stdout.");
      System.out.println();

      boolean color = useColor();
      try (LLM llm = LLM.builder(path)
          .enforceEager(true)
          .tensorParallelSize(1)
          .maxNumSeqs(4)
          .maxModelLen(2048)
          .withSystemIo()
          .build()) {
        if (preparedRag.isPresent()) {
          runRagChat(in, llm, preparedRag.get(), color);
        } else {
          runPlainChat(in, llm, color);
        }
      }
    }
  }

  private static Optional<PreparedRag> loadPreparedRag(final boolean tinyModel) {
    return BundledRag.find().map(root -> {
      System.out.println("Preparing RAG corpus from " + root);
      RagLoadOptions options =
          tinyModel ? RagLoadOptions.forTinyModels() : RagLoadOptions.defaults();
      return RagFactory.make(root, options, EngineIo.system());
    });
  }

  private static void runRagChat(
      final BufferedReader in,
      final LLM llm,
      final PreparedRag prepared,
      final boolean color
  ) throws Exception {
    boolean gemma = llm.tokenizer().isGemmaChat();
    int maxTokens = gemma ? RAG_MAX_TOKENS_GEMMA : RAG_MAX_TOKENS_DEFAULT;
    RagSession rag = llm.rag(prepared, maxTokens)
        .maxTokensWhenNoHits(gemma ? MAX_NEW_TOKENS : RAG_MAX_TOKENS_DEFAULT)
        .topK(gemma ? RAG_TOP_K_GEMMA : RAG_TOP_K_DEFAULT)
        .maxContextChars(gemma ? RAG_CONTEXT_CHARS_GEMMA : RAG_CONTEXT_CHARS_DEFAULT)
      .enableThinking(llm.tokenizer().invitesThinking())
      .sampling(new SamplingParams(gemma ? 0.1f : 0.4f, maxTokens, false, gemma ? 30 : 0,
        gemma ? 0.8f : 0.85f))
        .streamTo(System.err, System.out, color)
        .diagnostics(System.err::println);

    while (true) {
      System.out.print("rag?> ");
      System.out.flush();
      String line = in.readLine();
      if (line == null) {
        System.out.println();
        break;
      }
      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isExit(user)) {
        break;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        rag.clear();
        System.out.println("(conversation cleared; RAG index kept)");
        continue;
      }

      rag.send(user);
      printRetrievalSummary(rag.lastHits());
      System.out.println();
      System.out.flush();
    }
  }

  private static void printRetrievalSummary(final java.util.List<RagHit> hits) {
    if (hits.isEmpty()) {
      System.out.println("(no RAG hits)");
      System.out.flush();
      return;
    }
    String sources = hits.stream()
        .map(hit -> Path.of(hit.chunk().source()).getFileName().toString())
        .distinct()
        .collect(Collectors.joining(", "));
    System.out.println("(retrieved " + hits.size() + " chunk(s): " + sources + ")");
    System.out.flush();
  }

  private static void runPlainChat(final BufferedReader in, final LLM llm, final boolean color)
      throws Exception {
    ChatSession chat = llm.chat(MAX_NEW_TOKENS)
        .streamTo(System.err, System.out, color)
        .diagnostics(System.err::println);

    while (true) {
      System.out.print("?> ");
      System.out.flush();
      String line = in.readLine();
      if (line == null) {
        System.out.println();
        break;
      }
      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      if (isExit(user)) {
        break;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        chat.clear();
        System.out.println("(conversation cleared)");
        continue;
      }

      chat.send(user);
      System.out.println();
    }
  }

  /**
   * CLI / {@code -Dnanovllm.model} / {@code NANOVLLM_MODEL} skip the menu.
   */
  static Path resolveModel(final String[] args, final BufferedReader in) throws Exception {
    if (hasExplicitModel(args)) {
      return BundledModels.resolveDefault(args);
    }
    return promptModelChoice(in);
  }

  private static boolean hasExplicitModel(final String[] args) {
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      return true;
    }
    String prop = System.getProperty(PROP_MODEL);
    if (prop != null && !prop.isBlank()) {
      return true;
    }
    String env = System.getenv(ENV_MODEL);
    return env != null && !env.isBlank();
  }

  private static Path promptModelChoice(final BufferedReader in) throws Exception {
    var qwen = BundledModels.find(BundledModels.QWEN3_0_6B);
    var gemma = BundledModels.find(BundledModels.GEMMA3_270M);
    var lfm2 = BundledModels.find(BundledModels.LFM2_5_2_6B_GGUF);

    while (true) {
      System.out.println("Select model to load:");
      System.out.println("  1) Qwen3-0.6B" + (qwen.isPresent() ? "" : "  [not downloaded]"));
      System.out.println("  2) Gemma3-270M" + (gemma.isPresent() ? "" : "  [not downloaded]"));
      System.out.println(
        "  3) LFM2.5-2.6B GGUF Q4_K_M" + (lfm2.isPresent() ? "" : "  [not downloaded]")
          + "  (~16g heap)");
      System.out.println("  4) Exit");
      System.out.print("Choice [1-4]: ");
      System.out.flush();
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
          System.out.println("Bye.");
          return null;
        }
        default -> System.out.println("Enter 1, 2, 3, or 4.");
      }
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
    return !"false".equalsIgnoreCase(System.getProperty(PROP_COLOR, "true"));
  }
}
