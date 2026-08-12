package com.igormaznitsa.nanollvm.samples;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_COLOR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL;

import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.RagHit;
import com.igormaznitsa.nanollvm.rag.RagSession;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.CausalBundle;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.ModelChoice;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.RagMode;
import com.igormaznitsa.nanollvm.samples.ExampleSessionSupport.RagSetup;
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

final class ExampleCli {

  private ExampleCli() {
  }

  static void run(final String[] args) throws Exception {
    OrderedConsole console = new OrderedConsole(System.out, System.err);
    try (BufferedReader in = new BufferedReader(
      new InputStreamReader(System.in, StandardCharsets.UTF_8))) {

      Path path = resolveModel(args, in, console);
      if (path == null) {
        return;
      }

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
          + "(override: -Dnanollvm.arch=qwen3|gemma3|llama|lfm2|bert).");
      console.printlnInfo(
        "CPU matmul: " + Runtime.getRuntime().availableProcessors()
          + " threads from Runtime (override: -Dnanollvm.cpu.threads=N).");
      if (ExampleSessionSupport.isTinyLlmPath(path)) {
        console.printlnInfo(
          "This checkpoint is a base/completion toy model (~10M), not chat-tuned — "
            + "expect odd replies under the chat/RAG UI.");
      }
      if (ExampleSessionSupport.isGgufPath(path)) {
        console.printlnInfo(
          "GGUF: weights stay packed (dequant on matmul). For float32 at load: "
            + "LlmModelFactory.make(path, io, true).");
      }

      try (LlmModel model = LlmModelFactory.make(path, status)) {
        if (model.isEmbeddingModel()) {
          runEmbeddingRepl(model, in, console);
          return;
        }
        runCausalSession(model, ExampleSessionSupport.isGemmaPath(path), status, in, console);
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
        ExampleSessionSupport.l2Norm(vector),
        elapsedSec,
        ExampleSessionSupport.preview(vector));
      if (previous != null) {
        console.printf(Locale.ROOT, "cos(prev)=%.4f%n",
          ExampleSessionSupport.cosine(previous, vector));
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
      : ExampleSessionSupport.prepareRagSetup(
      ragRoot.get(),
      ragMode,
      gtePath.orElse(null),
      gemmaPath,
      status,
      console::printlnInfo);
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

      try (CausalBundle causal = ExampleSessionSupport.openCausal(
        model, ragSetup, status, console::printlnInfo)) {
        RagSession rag = causal.rag();
        ChatSession chat = causal.chat();
        String promptLabel = causal.ragMode() ? "rag?> " : "?> ";
        if (rag != null) {
          rag.streamTo(thinkOut, answerOut, color);
        } else {
          chat.streamTo(thinkOut, answerOut, color);
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

  static Path resolveModel(
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

    List<ModelChoice> catalog = ExampleSessionSupport.catalog();
    while (true) {
      console.println("Select model to load:");
      console.println(
        "  Kind: chat = instruct Q&A · base = plain completion (not chat-tuned)"
          + " · embeddings = vectors");
      for (int i = 0; i < catalog.size(); i++) {
        console.println("  " + (i + 1) + ") " + catalog.get(i).display());
      }
      console.println("  " + (catalog.size() + 1) + ") Exit");
      console.print("Choice [1-" + (catalog.size() + 1) + "]: ");
      String line = in.readLine();
      if (line == null) {
        return null;
      }
      String choice = line.strip();
      if (choice.equals(Integer.toString(catalog.size() + 1))
        || choice.equalsIgnoreCase("q")
        || choice.equalsIgnoreCase("quit")
        || choice.equalsIgnoreCase("exit")) {
        console.println("Bye.");
        return null;
      }
      try {
        int index = Integer.parseInt(choice) - 1;
        if (index >= 0 && index < catalog.size()) {
          return catalog.get(index).requirePath();
        }
      } catch (NumberFormatException ignored) {
        // fall through
      } catch (IllegalStateException missing) {
        throw missing;
      }
      console.println("Enter 1 … " + (catalog.size() + 1) + ".");
    }
  }
}
