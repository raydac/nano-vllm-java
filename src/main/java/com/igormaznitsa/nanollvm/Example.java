package com.igormaznitsa.nanollvm;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_COLOR;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_MODEL;

import com.igormaznitsa.nanollvm.chat.ChatSession;
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
  private static final int RAG_MAX_TOKENS_DEFAULT = 384;
  private static final int RAG_MAX_TOKENS_GEMMA = 96;
  private static final int RAG_TOP_K_DEFAULT = 4;
  private static final int RAG_TOP_K_GEMMA = 2;
  private static final int RAG_CONTEXT_CHARS_DEFAULT = 3500;
  private static final int RAG_CONTEXT_CHARS_GEMMA = 700;

  private Example() {
  }

  public static void main(String[] args) throws Exception {
    try (BufferedReader in = new BufferedReader(
        new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
      Path path = resolveModel(args, in);
      if (path == null) {
        return;
      }

      boolean gemmaPath = path.toString().toLowerCase(Locale.ROOT).contains("gemma");
      Optional<PreparedRag> preparedRag = loadPreparedRag(gemmaPath);

      System.out.println("Loading model from " + path);
      System.out.println(
          "Architecture auto-detects from config.json (override: -Dnanovllm.arch=qwen3|gemma3).");
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

  private static Optional<PreparedRag> loadPreparedRag(boolean tinyModel) {
    return BundledRag.find().map(root -> {
      System.out.println("Preparing RAG corpus from " + root);
      RagLoadOptions options =
          tinyModel ? RagLoadOptions.forTinyModels() : RagLoadOptions.defaults();
      return RagFactory.make(root, options);
    });
  }

  private static void runRagChat(
      BufferedReader in,
      LLM llm,
      PreparedRag prepared,
      boolean color
  ) throws Exception {
    boolean gemma = llm.tokenizer().isGemmaChat();
    int maxTokens = gemma ? RAG_MAX_TOKENS_GEMMA : RAG_MAX_TOKENS_DEFAULT;
    RagSession rag = llm.rag(prepared, maxTokens)
        .maxTokensWhenNoHits(gemma ? MAX_NEW_TOKENS : RAG_MAX_TOKENS_DEFAULT)
        .topK(gemma ? RAG_TOP_K_GEMMA : RAG_TOP_K_DEFAULT)
        .maxContextChars(gemma ? RAG_CONTEXT_CHARS_GEMMA : RAG_CONTEXT_CHARS_DEFAULT)
        .sampling(new SamplingParams(gemma ? 0.2f : 0.6f, maxTokens, false, gemma ? 40 : 0,
            gemma ? 0.85f : 0.9f))
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

  private static void printRetrievalSummary(java.util.List<RagHit> hits) {
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

  private static void runPlainChat(BufferedReader in, LLM llm, boolean color) throws Exception {
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
  static Path resolveModel(String[] args, BufferedReader in) throws Exception {
    if (hasExplicitModel(args)) {
      return BundledModels.resolveDefault(args);
    }
    return promptModelChoice(in);
  }

  private static boolean hasExplicitModel(String[] args) {
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

  private static Path promptModelChoice(BufferedReader in) throws Exception {
    var qwen = BundledModels.find(BundledModels.QWEN3_0_6B);
    var gemma = BundledModels.find(BundledModels.GEMMA3_270M);

    while (true) {
      System.out.println("Select model to load:");
      System.out.println("  1) Qwen3-0.6B" + (qwen.isPresent() ? "" : "  [not downloaded]"));
      System.out.println("  2) Gemma3-270M" + (gemma.isPresent() ? "" : "  [not downloaded]"));
      System.out.println("  3) Exit");
      System.out.print("Choice [1-3]: ");
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
        case "3", "q", "quit", "exit" -> {
          System.out.println("Bye.");
          return null;
        }
        default -> System.out.println("Enter 1, 2, or 3.");
      }
    }
  }

  private static boolean isExit(String user) {
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
