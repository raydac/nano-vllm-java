package io.nanovllm;

import io.nanovllm.chat.AssistantParts;
import io.nanovllm.chat.ChatMessages;
import io.nanovllm.chat.FactMemory;
import io.nanovllm.chat.MessageClassifier;
import io.nanovllm.chat.StreamPrinter;
import io.nanovllm.prompts.ChatPrompts;
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
 * Interactive chat: every user turn updates the knowledge base (extract facts/rules),
 * then the assistant always replies in chat with that KB in the system prompt.
 */
public final class Example {

  private static final int MAX_NEW_TOKENS = 768;
  private static final int EXTRACT_MAX_TOKENS = 512;
  private static final int GEMMA_THINK_MAX_TOKENS = 96;

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
      System.out.println("Type a message and press Enter. Commands: /exit  /quit  /clear  /memory");
      System.out.println("Thinking → stderr (dim cyan); reply → stdout.");
      System.out.println("Every turn: extract lasting facts/rules into the session KB, then chat.");
      System.out.println("Say \"forget …\" to remove a memory. Use /memory to list.");
      System.out.println();

      boolean color = useColor();
      try (LLM llm = new LLM(path, Map.of(
          "enforce_eager", true,
          "tensor_parallel_size", 1,
          "max_num_seqs", 4,
          "max_model_len", 2048
      ))) {
        Tokenizer tokenizer = llm.tokenizer();
        int maxModelLen = llm.config().maxModelLen();
        SamplingParams samplingParams = new SamplingParams(0.6f, MAX_NEW_TOKENS, false, 0, 0.95f);
        SamplingParams extractParams =
            new SamplingParams(0.05f, EXTRACT_MAX_TOKENS, false, 0, 1.0f);
        List<String> knowledge = new ArrayList<>();
        List<Map<String, String>> history = ChatMessages.newConversation(knowledge);

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
            knowledge.clear();
            history = ChatMessages.newConversation(knowledge);
            System.out.println("(conversation and knowledge base cleared)");
            continue;
          }
          if ("/memory".equalsIgnoreCase(user)) {
            printMemory(knowledge);
            continue;
          }

          if (looksLikeForgetRequest(user)) {
            applyForget(knowledge, user);
            ChatMessages.syncSystemMessage(history, knowledge);
          } else if (!MessageClassifier.looksLikeBareGreeting(user)) {
            FactMemory.extractAndStore(llm, tokenizer, knowledge, user, extractParams);
            ChatMessages.syncSystemMessage(history, knowledge);
          }

          history.add(ChatMessages.message("user", user));
          ChatMessages.truncateHistory(history, tokenizer, maxModelLen, samplingParams.maxTokens());
          runChatTurn(llm, tokenizer, history, knowledge, samplingParams, color);
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
      List<String> knowledge,
      SamplingParams samplingParams,
      boolean color
  ) {
    if (tokenizer.isGemmaChat()) {
      runGemmaChatTurn(llm, tokenizer, history, knowledge, samplingParams, color);
      return;
    }

    String prompt = tokenizer.applyChatTemplate(history, true, true);
    List<Integer> streamedIds = new ArrayList<>();
    StreamPrinter printer = new StreamPrinter(System.err, System.out, color);
    List<LLM.GenerationOutput> outputs = llm.generate(
        List.of(prompt),
        samplingParams,
        false,
        tokenId -> {
          streamedIds.add(tokenId);
          printer.update(AssistantParts.parse(tokenizer.decode(streamedIds, false)));
        }
    );

    AssistantParts parts =
        AssistantParts.parse(tokenizer.decode(outputs.getFirst().tokenIds(), false));
    finishChatTurn(parts, knowledge, history, printer);
  }

  /**
   * Gemma has no native reasoning mode. Two passes: fill a think scaffold, then answer
   * with </think> already closed so the visible reply cannot be swallowed by thinking.
   */
  private static void runGemmaChatTurn(
      LLM llm,
      Tokenizer tokenizer,
      List<Map<String, String>> history,
      List<String> knowledge,
      SamplingParams samplingParams,
      boolean color
  ) {
    String userText = lastUserMessage(history);
    if (MessageClassifier.looksLikeBareGreeting(userText)) {
      runGemmaSimpleChatTurn(llm, tokenizer, history, knowledge, samplingParams, color);
      return;
    }

    SamplingParams thinkParams = new SamplingParams(
        Math.min(0.4f, samplingParams.temperature()),
        GEMMA_THINK_MAX_TOKENS,
        false,
        samplingParams.topK(),
        samplingParams.topP()
    );
    String thinkPrompt = tokenizer.applyChatTemplate(history, true, true);
    List<Integer> thinkIds = new ArrayList<>();
    StreamPrinter thinkPrinter = new StreamPrinter(System.err, System.out, color);
    llm.generate(
        List.of(thinkPrompt),
        thinkParams,
        false,
        tokenId -> {
          thinkIds.add(tokenId);
          // Prefixed scaffold is not in token ids; parse generated fragment as open think.
          String gen = AssistantParts.stripChatMarkup(tokenizer.decode(thinkIds, true));
          thinkPrinter.update(new AssistantParts(stripThinkTags(gen), "", true));
        }
    );

    String generatedThink = AssistantParts.stripChatMarkup(tokenizer.decode(thinkIds, true));
    if (generatedThink.isBlank()) {
      runGemmaSimpleChatTurn(llm, tokenizer, history, knowledge, samplingParams, color);
      return;
    }
    AssistantParts first = AssistantParts.parse(
        ChatPrompts.GEMMA_THINK_SCAFFOLD + generatedThink);
    if (!first.thinkOpen() && !first.answer().isBlank()) {
      // Model closed think and answered in one shot — keep it.
      finishChatTurn(first, knowledge, history, thinkPrinter);
      return;
    }

    String thinking = cleanGemmaThinking(first.thinking());
    thinkPrinter.update(new AssistantParts(thinking, "", false));
    thinkPrinter.closeTurn();

    String answerPrompt = tokenizer.applyChatTemplate(history, true, false)
        + "<think>\n" + thinking + "\n</think>\n";
    List<Integer> answerIds = new ArrayList<>();
    StreamPrinter answerPrinter = new StreamPrinter(System.err, System.out, color);
    // Show closed think once, then stream the answer.
    answerPrinter.update(new AssistantParts(thinking, "", false));
    llm.generate(
        List.of(answerPrompt),
        samplingParams,
        false,
        tokenId -> {
          answerIds.add(tokenId);
          String answerSoFar = AssistantParts.stripChatMarkup(tokenizer.decode(answerIds, true));
          answerPrinter.update(new AssistantParts(thinking, answerSoFar, false));
        }
    );

    String answerRaw = tokenizer.decode(answerIds, true);
    AssistantParts parts = new AssistantParts(
        thinking,
        AssistantParts.parse(answerRaw).answer(),
        false
    );
    finishChatTurn(parts, knowledge, history, answerPrinter);
  }

  /**
   * Single-pass Gemma chat (no think scaffold) — reliable for greetings and when think pass hits EOS.
   */
  private static void runGemmaSimpleChatTurn(
      LLM llm,
      Tokenizer tokenizer,
      List<Map<String, String>> history,
      List<String> knowledge,
      SamplingParams samplingParams,
      boolean color
  ) {
    String prompt = tokenizer.applyChatTemplate(history, true, false);
    List<Integer> streamedIds = new ArrayList<>();
    StreamPrinter printer = new StreamPrinter(System.err, System.out, color);
    List<LLM.GenerationOutput> outputs = llm.generate(
        List.of(prompt),
        samplingParams,
        false,
        tokenId -> {
          streamedIds.add(tokenId);
          printer.update(AssistantParts.parse(tokenizer.decode(streamedIds, true)));
        }
    );
    AssistantParts parts =
        AssistantParts.parse(tokenizer.decode(outputs.getFirst().tokenIds(), true));
    finishChatTurn(parts, knowledge, history, printer);
  }

  private static String lastUserMessage(List<Map<String, String>> history) {
    for (int i = history.size() - 1; i >= 0; i--) {
      Map<String, String> msg = history.get(i);
      if ("user".equals(msg.getOrDefault("role", ""))) {
        return msg.getOrDefault("content", "");
      }
    }
    return "";
  }

  private static void finishChatTurn(
      AssistantParts parts,
      List<String> knowledge,
      List<Map<String, String>> history,
      StreamPrinter printer
  ) {
    String answer = FactMemory.stripMemoryDirectives(parts.answer());
    answer = FactMemory.rewriteMistakenFirstPersonIdentity(answer, knowledge);
    if (answer.isBlank() && !parts.thinking().isBlank()) {
      answer = salvageVisibleReply(parts.thinking());
      System.err.println("(reply recovered from unclosed thinking)");
    }
    if (answer.isBlank()) {
      answer = "Sorry — I couldn't form a reply. Please try again.";
      System.err.println("(empty reply — used fallback)");
    }
    parts = new AssistantParts(parts.thinking(), answer, false);
    printer.update(parts);
    printer.closeTurn();

    FactMemory.applyKnowledgeDirectives(knowledge, parts.answer());
    ChatMessages.syncSystemMessage(history, knowledge);
    System.out.println();

    history.add(ChatMessages.message("assistant", parts.answer()));
  }

  private static String stripThinkTags(String text) {
    return AssistantParts.stripChatMarkup(text);
  }

  private static String cleanGemmaThinking(String thinking) {
    if (thinking == null || thinking.isBlank()) {
      return "(no notes)";
    }
    String t = stripThinkTags(thinking);
    // Drop trailing half-answer leakage into think.
    int cut = t.indexOf("assistant>");
    if (cut >= 0) {
      t = t.substring(0, cut).strip();
    }
    if (t.length() > 600) {
      t = t.substring(0, 600).strip() + "…";
    }
    return t.isBlank() ? "(no notes)" : t;
  }

  private static String salvageVisibleReply(String thinking) {
    if (thinking == null || thinking.isBlank()) {
      return "";
    }
    String[] lines = thinking.strip().split("\\R");
    for (int i = lines.length - 1; i >= 0; i--) {
      String line = lines[i].strip();
      if (line.isEmpty() || line.startsWith("+")) {
        continue;
      }
      if (line.length() >= 8) {
        return line;
      }
    }
    String one = thinking.replace('\n', ' ').strip();
    return one.length() > 200 ? one.substring(0, 200).strip() + "…" : one;
  }

  static boolean looksLikeForgetRequest(String user) {
    if (user == null || user.isBlank()) {
      return false;
    }
    String text = user.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    return text.equals("forget")
        || text.startsWith("forget ")
        || text.startsWith("please forget")
        || text.contains("forget that")
        || text.contains("forget about")
        || text.contains("forget my ");
  }

  private static void applyForget(List<String> knowledge, String user) {
    String probe = FactMemory.stripUserMemoryCues(user);
    String removed = FactMemory.forgetBestMatch(knowledge, probe);
    if (removed != null) {
      System.err.println("(knowledge-) " + removed);
    } else {
      System.err.println("(knowledge? nothing matched forget) " + probe);
    }
  }

  private static void printMemory(List<String> knowledge) {
    if (knowledge.isEmpty()) {
      System.out.println("(knowledge base empty)");
      return;
    }
    System.out.println("Knowledge base:");
    for (String fact : knowledge) {
      System.out.println(" - " + fact);
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
    return !"false".equalsIgnoreCase(System.getProperty("nanovllm.color", "true"));
  }
}
