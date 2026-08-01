package io.nanovllm;

import io.nanovllm.chat.ChatSession;
import io.nanovllm.utils.BundledModels;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Interactive chat using {@link ChatSession} over a loaded {@link LLM}.
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
      try (LLM llm = LLM.builder(path)
          .enforceEager(true)
          .tensorParallelSize(1)
          .maxNumSeqs(4)
          .maxModelLen(2048)
          .withSystemIo()
          .build()) {
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
