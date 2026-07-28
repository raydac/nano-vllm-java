package io.nanovllm;

import io.nanovllm.chat.AssistantParts;
import io.nanovllm.chat.ChatMessages;
import io.nanovllm.chat.StreamPrinter;
import io.nanovllm.tokenizer.Tokenizer;
import io.nanovllm.utils.BundledModels;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Interactive chat using the system prompt plus rolling dialog history as context.
 */
public final class Example {

  private static final int MAX_NEW_TOKENS = 768;

  private Example() {
  }

  public static void main(String[] args) throws Exception {
    try (BufferedReader in = new BufferedReader(
        new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
      Path path = resolveModel(args, in);
      if (path == null) {
        return;
      }

      System.out.println("Loading model from " + path);
      System.out.println(
          "Architecture auto-detects from config.json (override: -Dnanovllm.arch=qwen3|gemma3).");
      System.out.println("Type a message and press Enter. Commands: /exit  /quit  /clear");
      System.out.println("Thinking → stderr (dim cyan); reply → stdout.");
      System.out.println("Context is the system prompt plus this session's dialog history.");
      System.out.println();

      boolean color = useColor();
      try (LLM llm = new LLM(path, Map.of(
          "enforce_eager", true,
          "tensor_parallel_size", 1,
          "max_num_seqs", 4,
          "max_model_len", 2048
      ))) {
        Tokenizer tokenizer = llm.tokenizer();
        boolean gemmaChat = tokenizer.isGemmaChat();
        int maxModelLen = llm.config().maxModelLen();
        SamplingParams samplingParams = new SamplingParams(0.6f, MAX_NEW_TOKENS, false, 0, 0.95f);
        List<Map<String, String>> history = ChatMessages.newConversation(gemmaChat);

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
            history = ChatMessages.newConversation(gemmaChat);
            System.out.println("(conversation cleared)");
            continue;
          }

          history.add(ChatMessages.message("user", user));
          ChatMessages.truncateHistory(history, tokenizer, maxModelLen, samplingParams.maxTokens());
          runChatTurn(llm, tokenizer, history, samplingParams, color);
        }
      }
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
    String prop = System.getProperty("nanovllm.model");
    if (prop != null && !prop.isBlank()) {
      return true;
    }
    String env = System.getenv("NANOVLLM_MODEL");
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

  private static void runChatTurn(
      LLM llm,
      Tokenizer tokenizer,
      List<Map<String, String>> history,
      SamplingParams samplingParams,
      boolean color
  ) {
    boolean gemmaChat = tokenizer.isGemmaChat();
    boolean enableThinking = !gemmaChat;
    String prompt = tokenizer.applyChatTemplate(history, true, enableThinking);
    List<Integer> streamedIds = new ArrayList<>();
    StreamPrinter printer = new StreamPrinter(System.err, System.out, color);
    List<LLM.GenerationOutput> outputs = llm.generate(
        List.of(prompt),
        samplingParams,
        false,
        tokenId -> {
          streamedIds.add(tokenId);
          printer.update(AssistantParts.parse(tokenizer.decode(streamedIds, gemmaChat)));
        }
    );

    AssistantParts parts =
        AssistantParts.parse(tokenizer.decode(outputs.getFirst().tokenIds(), gemmaChat));
    finishChatTurn(parts, history, printer);
  }

  private static void finishChatTurn(
      AssistantParts parts,
      List<Map<String, String>> history,
      StreamPrinter printer
  ) {
    String answer = parts.answer().strip();
    if (answer.isBlank() && !parts.thinking().isBlank()) {
      answer = AssistantParts.salvageFromThinking(parts.thinking());
      if (parts.thinkOpen()) {
        System.err.println("(reply recovered from unclosed thinking)");
      } else {
        System.err.println("(reply recovered from thinking; model omitted visible answer)");
      }
    }
    if (answer.isBlank()) {
      answer = "Sorry — I couldn't form a reply. Please try again.";
      System.err.println("(empty reply — used fallback)");
    }
    parts = new AssistantParts(parts.thinking(), answer, false);
    printer.update(parts);
    printer.closeTurn();

    System.out.println();
    history.add(ChatMessages.message("assistant", parts.answer()));
  }

  private static boolean isExit(String user) {
    String t = user.toLowerCase(Locale.ROOT);
    return t.equals("/exit") || t.equals("/quit") || t.equals("exit") || t.equals("quit");
  }

  private static boolean useColor() {
    if (System.getenv("NO_COLOR") != null) {
      return false;
    }
    return !"false".equalsIgnoreCase(System.getProperty("nanovllm.color", "true"));
  }
}
