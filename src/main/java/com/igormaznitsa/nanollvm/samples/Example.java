package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_COLOR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL;

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
import com.igormaznitsa.nanollvm.rag.DenseRagIndex;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagIndex;
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
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Interactive sample: causal chat/{@link RagSession} for LMs, or an embedding REPL for BERT GGUF.
 * After choosing a chat model, a second menu selects RAG mode (none / BM25 / dense / hybrid).
 */
public final class Example {

  private static final int MAX_NEW_TOKENS = 768;
  private static final int RAG_MAX_TOKENS_DEFAULT = 768;
  private static final int RAG_MAX_TOKENS_GEMMA = 128;
  private static final int RAG_TOP_K_DEFAULT = 4;
  private static final int RAG_TOP_K_GEMMA = 2;
  private static final int RAG_CONTEXT_CHARS_DEFAULT = 3500;
  private static final int RAG_CONTEXT_CHARS_GEMMA = 900;
  private static final int EMBED_PREVIEW = 8;

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

      String pathLower = path.toString().toLowerCase(Locale.ROOT);
      boolean gemmaPath = pathLower.contains("gemma");
      boolean ggufPath = pathLower.endsWith(".gguf");

      LlmListener status = (source, event) -> {
        switch (event.kind()) {
          case STATUS_INFO -> console.printInfo(event.text());
          case STATUS_PROGRESS -> console.print(event.text());
          default -> {
          }
        }
      };

      console.printlnInfo("Loading model from " + path);
      console.printlnInfo(
        "Architecture auto-detects from config.json / GGUF metadata "
          + "(override: -Dnanollvm.arch=qwen3|gemma3|lfm2|bert).");
      console.printlnInfo(
        "CPU matmul: " + Runtime.getRuntime().availableProcessors()
          + " threads from Runtime (override: -Dnanollvm.cpu.threads=N).");
      if (ggufPath) {
        console.printlnInfo(
          "GGUF: weights stay packed (dequant on matmul). For float32 at load: "
            + "LlmModelFactory.make(path, io, true).");
      }

      try (LlmModel model = LlmModelFactory.make(path, status)) {
        if (model.isEmbeddingModel()) {
          runEmbeddingRepl(model, in, console);
          return;
        }
        runCausalSession(model, gemmaPath, status, in, console);
      }
    }
  }

  private static void runEmbeddingRepl(
    final LlmModel model,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    console.printlnInfo(
      "Embedding model (" + model.architectureName() + ") — each line → L2-normalized vector.");
    console.println("Type text and press Enter. Commands: /exit  /quit  /clear");
    console.println();

    float[] previous = null;
    while (true) {
      console.print("embed?> ");
      String line = in.readLine();
      if (line == null) {
        console.println();
        break;
      }
      String user = line.strip();
      if (user.isEmpty()) {
        continue;
      }
      String command = user.toLowerCase(Locale.ROOT);
      if (command.equals("/exit") || command.equals("/quit")
        || command.equals("exit") || command.equals("quit")) {
        break;
      }
      if ("/clear".equalsIgnoreCase(user)) {
        previous = null;
        console.println("(previous embedding cleared)");
        continue;
      }

      long started = System.nanoTime();
      float[] vector = model.embed(user);
      double elapsedSec = (System.nanoTime() - started) / 1e9;
      console.printf(
        Locale.ROOT,
        "dim=%d  L2=%.4f  %.3fs  preview=%s%n",
        vector.length,
        l2Norm(vector),
        elapsedSec,
        preview(vector));
      if (previous != null) {
        console.printf(Locale.ROOT, "cos(prev)=%.4f%n", cosine(previous, vector));
      }
      previous = vector;
      console.println();
    }
  }

  private static void runCausalSession(
    final LlmModel model,
    final boolean gemmaPath,
    final LlmListener status,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    Optional<Path> ragRoot = BundledRag.find();
    Optional<Path> gtePath = BundledModels.find(BundledModels.GTE_SMALL_GGUF);

    RagMode ragMode;
    if (ragRoot.isEmpty()) {
      console.printlnInfo("RAG: no usable corpus at " + BundledRag.ragRoot() + " — plain chat.");
      ragMode = RagMode.NONE;
    } else {
      ragMode = chooseRagMode(in, console, gtePath.isPresent());
      if (ragMode == null) {
        return;
      }
    }

    RagSetup ragSetup = ragMode == RagMode.NONE
      ? new RagSetup(null, null)
      : prepareRagSetup(ragRoot.get(), ragMode, gtePath, gemmaPath, status, console);
    if (ragSetup == null) {
      console.printlnInfo("RAG: no documents in " + ragRoot.get() + " — plain chat.");
      ragSetup = new RagSetup(null, null);
    }

    try {
      if (ragSetup.index() != null) {
        console.printlnInfo("Ask about the docs in rag/ (engine, models, Nile, capitals, …).");
      }

      console.println("Type a message and press Enter. Commands: /exit  /quit  /clear");
      console.println(
        "Answer/prompts on stdout; thinking, debug, and load/status on stderr (red in many IDEs).");
      console.println(
        "After each turn: engine tok/s from GenerationStats (main generate; excludes advisors / RAG prep).");
      console.println();

      boolean color = System.getenv("NO_COLOR") == null
        && !"false".equalsIgnoreCase(NanoLlvmProps.systemProperty(PROP_COLOR));
      PrintStream answerOut = console.stream();
      PrintStream thinkOut = console.infoStream();

      LLM.Builder builder = LLM.builder(model)
        .maxNumSeqs(4)
        .maxModelLen(2048)
        .listen(status);

      String arch = model.architectureName();
      switch (arch) {
        case ARCH_GEMMA3 -> {
          builder.advisors(
            LlmAdvisorMixer.defaults(),
            LlmAdvisor.builder().name("Practical").prompt(AdvisorPrompts.ROLE_PRACTICAL).build(),
            LlmAdvisor.builder().name("Abstract").prompt(AdvisorPrompts.ROLE_ABSTRACT).build(),
            LlmAdvisor.builder().name("Consequence").prompt(AdvisorPrompts.ROLE_CONSEQUENCE)
              .build());
          console.printlnInfo("Advisors: Practical, Abstract, Consequence for Gemma.");
        }
        case ARCH_QWEN3 -> {
          builder.advisors(
            LlmAdvisorMixer.defaults(),
            LlmAdvisor.builder().name("Practical").prompt(AdvisorPrompts.ROLE_PRACTICAL).build(),
            LlmAdvisor.builder().name("Abstract").prompt(AdvisorPrompts.ROLE_ABSTRACT).build());
          console.printlnInfo("Advisors: Practical, Abstract for Qwen.");
        }
        case ARCH_LFM2 -> console.printlnInfo("Advisors: off for LFM.");
        case null, default -> console.printlnInfo("Advisors: off (architecture " + arch + ").");
      }

      try (LLM llm = builder.build()) {
        boolean gemma = llm.tokenizer().isGemmaChat();
        RagSession rag = null;
        ChatSession chat = null;
        String promptLabel;
        RagIndex ragIndex = ragSetup.index();

        if (ragIndex != null) {
          int maxTokens = gemma ? RAG_MAX_TOKENS_GEMMA : RAG_MAX_TOKENS_DEFAULT;
          rag = llm.rag(ragIndex, maxTokens)
            .maxTokensWhenNoHits(gemma ? MAX_NEW_TOKENS : RAG_MAX_TOKENS_DEFAULT)
            .topK(gemma ? RAG_TOP_K_GEMMA : RAG_TOP_K_DEFAULT)
            .maxContextChars(gemma ? RAG_CONTEXT_CHARS_GEMMA : RAG_CONTEXT_CHARS_DEFAULT)
            .enableThinking(llm.tokenizer().invitesThinking())
            .sampling(new SamplingParams(
              gemma ? 0.1f : 0.4f,
              maxTokens,
              false,
              gemma ? 30 : 0,
              gemma ? 0.8f : 0.85f))
            .streamTo(thinkOut, answerOut, color);
          promptLabel = "rag?> ";
        } else {
          chat = llm.chat(MAX_NEW_TOKENS).streamTo(thinkOut, answerOut, color);
          promptLabel = "?> ";
        }

        long totalTokens = 0;
        long totalNanos = 0;
        int turns = 0;

        while (true) {
          console.print(promptLabel);
          String line = in.readLine();
          if (line == null) {
            console.println();
            break;
          }

          String user = line.strip();
          if (user.isEmpty()) {
            continue;
          }

          String command = user.toLowerCase(Locale.ROOT);
          if (command.equals("/exit") || command.equals("/quit")
            || command.equals("exit") || command.equals("quit")) {
            break;
          }
          if ("/clear".equalsIgnoreCase(user)) {
            if (rag != null) {
              rag.clear();
              console.println("(conversation cleared; RAG index kept)");
            } else {
              chat.clear();
              console.println("(conversation cleared)");
            }
            continue;
          }

          ChatReply reply = rag != null ? rag.send(user) : chat.send(user);
          int tokens = reply.stats().completionTokens();
          long nanos = Math.max(1L, reply.stats().elapsedNanos());
          totalTokens += tokens;
          totalNanos += nanos;
          turns++;
          console.printf(
            Locale.ROOT,
            "(turn %d: %d tok in %.2fs → %.1f tok/s; session avg %.1f tok/s)%n",
            turns,
            tokens,
            nanos / 1e9,
            reply.stats().completionTokensPerSecond(),
            totalTokens / (totalNanos / 1e9));

          if (rag != null) {
            List<RagHit> hits = rag.lastHits();
            if (hits.isEmpty()) {
              console.println("(no RAG hits)");
            } else {
              String sources = hits.stream()
                .map(hit -> Path.of(hit.chunk().source()).getFileName().toString())
                .distinct()
                .collect(Collectors.joining(", "));
              console.println("(retrieved " + hits.size() + " chunk(s): " + sources + ")");
            }
          }
          console.println();
        }

        if (turns > 0) {
          console.printf(
            Locale.ROOT,
            "(session: %d turn(s), %d tok, %.2fs → avg %.1f tok/s)%n",
            turns,
            totalTokens,
            totalNanos / 1e9,
            totalTokens / (totalNanos / 1e9));
        }
      }
    } finally {
      if (ragSetup.embeddingModel() != null) {
        ragSetup.embeddingModel().close();
      }
    }
  }

  private static RagMode chooseRagMode(
    final BufferedReader in,
    final OrderedConsole console,
    final boolean gtePresent
  ) throws Exception {
    String denseMark = gtePresent ? "" : "  [not downloaded]";
    while (true) {
      console.println("Select RAG mode:");
      console.println("  1) None (plain chat)");
      console.println("  2) BM25 lexical");
      console.println("  3) Dense embeddings (gte-small)" + denseMark);
      console.println("  4) Hybrid BM25 + dense" + denseMark);
      console.println("  5) Back / exit");
      console.print("Choice [1-5]: ");
      String line = in.readLine();
      if (line == null) {
        return null;
      }
      switch (line.strip()) {
        case "1" -> {
          console.printlnInfo("RAG: off — plain chat.");
          return RagMode.NONE;
        }
        case "2" -> {
          return RagMode.BM25;
        }
        case "3" -> {
          if (!gtePresent) {
            console.println(
              "gte-small GGUF not found. Run models/download-gte-small-gguf.sh");
            continue;
          }
          return RagMode.DENSE;
        }
        case "4" -> {
          if (!gtePresent) {
            console.println(
              "gte-small GGUF not found. Run models/download-gte-small-gguf.sh");
            continue;
          }
          return RagMode.HYBRID;
        }
        case "5", "q", "quit", "exit" -> {
          console.println("Bye.");
          return null;
        }
        default -> console.println("Enter 1, 2, 3, 4, or 5.");
      }
    }
  }

  private static RagSetup prepareRagSetup(
    final Path ragRoot,
    final RagMode ragMode,
    final Optional<Path> gtePath,
    final boolean gemmaPath,
    final LlmListener status,
    final OrderedConsole console
  ) {
    console.printlnInfo("Preparing RAG corpus from " + ragRoot);
    RagLoadOptions options =
      gemmaPath ? RagLoadOptions.forTinyModels() : RagLoadOptions.defaults();
    PreparedRag lexical = RagFactory.tryMake(ragRoot, options, status).orElse(null);
    if (lexical == null) {
      return null;
    }

    return switch (ragMode) {
      case NONE -> new RagSetup(null, null);
      case BM25 -> {
        console.printlnInfo(
          "RAG: BM25 over " + BundledRag.ragRoot()
            + " (" + lexical.size() + " chunks)");
        yield new RagSetup(lexical, null);
      }
      case DENSE -> {
        Path gte = gtePath.orElseThrow();
        console.printlnInfo("Loading RAG embedding model from " + gte);
        LlmModel embed = LlmModelFactory.make(gte, status);
        try {
          DenseRagIndex dense = DenseRagIndex.of(lexical, embed);
          console.printlnInfo(
            "RAG: dense embeddings over " + BundledRag.ragRoot()
              + " (" + dense.size() + " chunks; encoder " + embed.architectureName() + ")");
          yield new RagSetup(dense, embed);
        } catch (RuntimeException | Error failed) {
          embed.close();
          throw failed;
        }
      }
      case HYBRID -> {
        Path gte = gtePath.orElseThrow();
        console.printlnInfo("Loading RAG embedding model from " + gte);
        LlmModel embed = LlmModelFactory.make(gte, status);
        try {
          RagIndex hybrid = RagFactory.withEmbeddings(lexical, embed);
          console.printlnInfo(
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

  private enum RagMode {
    NONE,
    BM25,
    DENSE,
    HYBRID
  }

  private record RagSetup(RagIndex index, LlmModel embeddingModel) {
  }

  private static double l2Norm(final float[] vector) {
    double sum = 0.0;
    for (float v : vector) {
      sum += (double) v * v;
    }
    return Math.sqrt(sum);
  }

  private static float cosine(final float[] a, final float[] b) {
    float dot = 0f;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  private static String preview(final float[] vector) {
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

  static Path resolveModel(final String[] args, final BufferedReader in) throws Exception {
    return resolveModel(args, in, new OrderedConsole(System.out, System.err));
  }

  private static Path resolveModel(
    final String[] args,
    final BufferedReader in,
    final OrderedConsole console
  ) throws Exception {
    boolean explicit = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      || NanoLlvmProps.systemProperty(PROP_MODEL) != null
      || NanoLlvmProps.environment(ENV_MODEL) != null;
    if (explicit) {
      return BundledModels.resolveDefault(args);
    }

    var qwen = BundledModels.find(BundledModels.QWEN3_0_6B);
    var gemma = BundledModels.find(BundledModels.GEMMA3_270M);
    var lfm2 = BundledModels.find(BundledModels.LFM2_5_2_6B_GGUF);
    var gte = BundledModels.find(BundledModels.GTE_SMALL_GGUF);

    while (true) {
      console.println("Select model to load:");
      console.println("  1) Qwen3-0.6B" + (qwen.isPresent() ? "" : "  [not downloaded]"));
      console.println("  2) Gemma3-270M" + (gemma.isPresent() ? "" : "  [not downloaded]"));
      console.println(
        "  3) LFM2.5-2.6B GGUF Q4_K_M" + (lfm2.isPresent() ? "" : "  [not downloaded]")
          + "  (~16g heap)");
      console.println(
        "  4) gte-small GGUF Q2_K" + (gte.isPresent() ? "" : "  [not downloaded]")
          + "  (embeddings)");
      console.println("  5) Exit");
      console.print("Choice [1-5]: ");
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
        case "4" -> {
          return gte.orElseThrow(() -> new IllegalStateException(
            "gte-small GGUF not found. Run models/download-gte-small-gguf.sh"));
        }
        case "5", "q", "quit", "exit" -> {
          console.println("Bye.");
          return null;
        }
        default -> console.println("Enter 1, 2, 3, 4, or 5.");
      }
    }
  }
}
