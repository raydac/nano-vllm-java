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
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Interactive chat using {@link RagSession} when {@code ./rag} has indexable documents,
 * otherwise plain {@link ChatSession}.
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

      String pathLower = path.toString().toLowerCase(Locale.ROOT);
      boolean gemmaPath = pathLower.contains("gemma");
      boolean ggufPath = pathLower.endsWith(".gguf") || pathLower.contains("lfm2");

      LlmListener status = (source, event) -> {
        switch (event.kind()) {
          case STATUS_INFO -> console.printInfo(event.text());
          case STATUS_PROGRESS -> console.print(event.text());
          default -> {
          }
        }
      };

      PreparedRag preparedRag = null;
      Optional<Path> ragRoot = BundledRag.find();
      if (ragRoot.isPresent()) {
        console.printlnInfo("Preparing RAG corpus from " + ragRoot.get());
        RagLoadOptions options =
          gemmaPath ? RagLoadOptions.forTinyModels() : RagLoadOptions.defaults();
        preparedRag = RagFactory.tryMake(ragRoot.get(), options, status).orElse(null);
        if (preparedRag == null) {
          console.printlnInfo("RAG: no documents in " + ragRoot.get() + " — plain chat.");
        }
      }

      console.printlnInfo("Loading model from " + path);
      console.printlnInfo(
        "Architecture auto-detects from config.json / GGUF metadata "
          + "(override: -Dnanollvm.arch=qwen3|gemma3|lfm2).");
      console.printlnInfo(
        "CPU matmul: " + Runtime.getRuntime().availableProcessors()
          + " threads from Runtime (override: -Dnanollvm.cpu.threads=N).");
      if (ggufPath) {
        console.printlnInfo(
          "GGUF/LFM2: weights stay packed (dequant on matmul). For float32 at load: "
            + "LlmModelFactory.make(path, io, true).");
      }
      if (preparedRag != null) {
        console.printlnInfo(
          "RAG: prepared BM25 over " + BundledRag.ragRoot()
            + " (" + preparedRag.size() + " chunks, shared index)");
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

      boolean color = System.getenv("NO_COLOR") == null
        && !"false".equalsIgnoreCase(NanoLlvmProps.systemProperty(PROP_COLOR));
      PrintStream answerOut = console.stream();
      PrintStream thinkOut = console.infoStream();

      try (LlmModel model = LlmModelFactory.make(path, status)) {
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

          if (preparedRag != null) {
            int maxTokens = gemma ? RAG_MAX_TOKENS_GEMMA : RAG_MAX_TOKENS_DEFAULT;
            rag = llm.rag(preparedRag, maxTokens)
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
      }
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
    boolean explicit = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      || NanoLlvmProps.systemProperty(PROP_MODEL) != null
      || NanoLlvmProps.environment(ENV_MODEL) != null;
    if (explicit) {
      return BundledModels.resolveDefault(args);
    }

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
}
