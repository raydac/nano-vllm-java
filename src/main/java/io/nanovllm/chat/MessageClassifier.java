package io.nanovllm.chat;

import static io.nanovllm.prompts.ChatPrompts.MESSAGE_CLASSIFY_SYSTEM;

import io.nanovllm.LLM;
import io.nanovllm.SamplingParams;
import io.nanovllm.prompts.ChatPrompts;
import io.nanovllm.prompts.MessageAnalysis;
import io.nanovllm.prompts.MessageIntent;
import io.nanovllm.tokenizer.Tokenizer;
import java.util.Locale;

public final class MessageClassifier {

  private MessageClassifier() {
  }

  public static MessageAnalysis classify(
      LLM llm,
      Tokenizer tokenizer,
      String user,
      SamplingParams classifyParams
  ) {
    System.err.println("(classify…)");
    // Do not seed "TYPE: " — tiny models often continue with random non-English tokens.
    String answer = ChatMessages.complete(
        llm,
        tokenizer,
        MESSAGE_CLASSIFY_SYSTEM,
        ChatPrompts.messageClassifyUserPayload(user),
        classifyParams
    );
    MessageAnalysis analysis = parseMessageAnalysis(answer);
    MessageAnalysis resolved = refineClassifiedIntent(analysis, user);
    if (resolved.intent() != analysis.intent()) {
      System.err.println("(classify) override " + analysis.intent() + " → " + resolved.intent()
          + " (" + overrideReason(analysis, resolved, user) + ")");
    }
    System.err.println("(classify) " + resolved.intent()
        + (resolved.forgetProbe() == null || resolved.forgetProbe().isBlank()
        ? ""
        : " probe=" + resolved.forgetProbe())
        + " ← " + ChatMessages.oneLineSummary(answer, 120));
    return resolved;
  }

  public static MessageAnalysis refineClassifiedIntent(MessageAnalysis analysis, String user) {
    return demoteStoreWhenNoLastingContent(
        demoteStoreWhenEphemeralTask(
            promoteStoreWhenFactDense(
                promoteQuestionWhenInterrogative(
                    promoteSkipWhenGreeting(analysis, user),
                    user),
                user),
            user),
        user);
  }

  private static String overrideReason(MessageAnalysis before, MessageAnalysis after, String user) {
    if (after.intent() == MessageIntent.SKIP && looksLikeBareGreeting(user)) {
      return "bare greeting";
    }
    if (after.intent() == MessageIntent.QUESTION && looksLikeQuestion(user)) {
      return "interrogative message";
    }
    if (after.intent() == MessageIntent.STORE && looksLikeFactDenseShare(user)) {
      return "fact-dense message";
    }
    if (before.intent() == MessageIntent.STORE && after.intent() == MessageIntent.CHAT) {
      if (looksLikeEphemeralTaskRequest(user)) {
        return "ephemeral task request";
      }
      return "no lasting fact/rule cues";
    }
    return "refined";
  }

  /**
   * Tiny models often mark flirt/chat as STORE. Keep STORE only when the message
   * actually looks like lasting personal facts or future rules.
   */
  public static MessageAnalysis demoteStoreWhenNoLastingContent(MessageAnalysis analysis,
                                                                String user) {
    if (analysis == null) {
      return new MessageAnalysis(MessageIntent.CHAT, null);
    }
    if (analysis.intent() != MessageIntent.STORE) {
      return analysis;
    }
    if (looksLikeFactDenseShare(user)
        || FactMemory.looksLikeLastingRule(user)
        || looksLikePersonalFactShare(user)) {
      return analysis;
    }
    return new MessageAnalysis(MessageIntent.CHAT, null);
  }

  /**
   * Lightweight cues that a message states lasting personal info (not a full bio).
   */
  public static boolean looksLikePersonalFactShare(String user) {
    if (user == null || user.isBlank() || looksLikeEphemeralTaskRequest(user)) {
      return false;
    }
    String text = user.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    if (text.endsWith("?") && !hasFirstPersonMemoryCue(text)) {
      return false;
    }
    return text.contains("my name is")
        || text.contains("i am a ")
        || text.contains("i'm a ")
        || text.contains("i am an ")
        || text.contains("i'm an ")
        || text.contains("i live")
        || text.contains("i work")
        || text.contains("i was born")
        || text.contains("i relocated")
        || text.contains("mother tongue")
        || text.contains("i prefer")
        || text.contains("i speak")
        || text.contains("i had ")
        || text.contains("i've had ")
        || text.contains("i have a ")
        || text.contains("i have an ")
        || text.contains("i owned ")
        || text.contains("i used to ")
        || text.contains("i played ")
        || text.contains("in my childhood")
        || text.contains("in my youth")
        || text.contains("as a child")
        || text.contains("when i was ")
        || text.contains("growing up")
        || text.contains("remember that")
        || text.contains("please remember")
        || text.startsWith("always ")
        || text.startsWith("never ")
        || text.contains(" from now on");
  }

  /**
   * I / my / me style cues that often carry storeable autobiography.
   */
  public static boolean hasFirstPersonMemoryCue(String user) {
    if (user == null || user.isBlank()) {
      return false;
    }
    String text = " " + user.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ") + " ";
    return text.contains(" i ")
        || text.contains(" i'm ")
        || text.contains(" i've ")
        || text.contains(" i'd ")
        || text.contains(" my ")
        || text.contains(" me ")
        || text.startsWith("i ")
        || text.startsWith("i'm ")
        || text.startsWith("my ");
  }

  public static MessageAnalysis promoteSkipWhenGreeting(MessageAnalysis analysis, String user) {
    if (analysis == null) {
      return new MessageAnalysis(MessageIntent.CHAT, null);
    }
    if (analysis.intent() == MessageIntent.FORGET) {
      return analysis;
    }
    if (looksLikeBareGreeting(user)) {
      return new MessageAnalysis(MessageIntent.SKIP, null);
    }
    return analysis;
  }

  /**
   * True for standalone hi/hello/thanks/bye with no other content.
   */
  public static boolean looksLikeBareGreeting(String user) {
    if (user == null || user.isBlank()) {
      return false;
    }
    String text = user.strip().toLowerCase(Locale.ROOT)
        .replaceAll("[!?.,]+$", "")
        .replaceAll("\\s+", " ")
        .strip();
    return text.equals("hi")
        || text.equals("hello")
        || text.equals("hey")
        || text.equals("yo")
        || text.equals("hiya")
        || text.equals("thanks")
        || text.equals("thank you")
        || text.equals("thx")
        || text.equals("bye")
        || text.equals("goodbye")
        || text.equals("good bye")
        || text.equals("good morning")
        || text.equals("good evening")
        || text.equals("good night")
        || text.equals("good afternoon")
        || text.equals("привет")
        || text.equals("здравствуй")
        || text.equals("здравствуйте")
        || text.equals("салют")
        || text.equals("hola")
        || text.equals("salut")
        || text.equals("bonjour")
        || text.equals("ciao");
  }

  public static MessageAnalysis promoteQuestionWhenInterrogative(MessageAnalysis analysis,
                                                                 String user) {
    if (analysis == null) {
      return new MessageAnalysis(MessageIntent.CHAT, null);
    }
    // Keep STORE (fact+ask) and FORGET; only upgrade CHAT/SKIP noise.
    if (analysis.intent() == MessageIntent.STORE
        || analysis.intent() == MessageIntent.FORGET
        || analysis.intent() == MessageIntent.QUESTION) {
      return analysis;
    }
    if (looksLikeQuestion(user) && !looksLikeFactDenseShare(user)) {
      return new MessageAnalysis(MessageIntent.QUESTION, null);
    }
    return analysis;
  }

  public static boolean looksLikeQuestion(String user) {
    if (user == null || user.isBlank()) {
      return false;
    }
    String text = user.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    if (text.endsWith("?")) {
      return true;
    }
    return startsWithAny(
        text,
        "who ", "what ", "where ", "when ", "why ", "how ",
        "who's ", "what's ", "where's ", "when's ", "why's ", "how's ",
        "do i ", "do you ", "did i ", "did you ",
        "am i ", "are you ", "is my ", "is your ",
        "can you ", "could you ", "would you "
    );
  }

  public static MessageAnalysis promoteStoreWhenFactDense(MessageAnalysis analysis, String user) {
    if (analysis == null) {
      return new MessageAnalysis(MessageIntent.CHAT, null);
    }
    MessageIntent intent = analysis.intent();
    if (intent == MessageIntent.STORE
        || intent == MessageIntent.QUESTION
        || intent == MessageIntent.FORGET) {
      return analysis;
    }
    if (looksLikeBareGreeting(user) || looksLikeEphemeralTaskRequest(user)) {
      return analysis;
    }
    if (looksLikeFactDenseShare(user)) {
      return new MessageAnalysis(MessageIntent.STORE, null);
    }
    return analysis;
  }

  public static MessageAnalysis demoteStoreWhenEphemeralTask(MessageAnalysis analysis,
                                                             String user) {
    if (analysis == null) {
      return new MessageAnalysis(MessageIntent.CHAT, null);
    }
    if (analysis.intent() != MessageIntent.STORE) {
      return analysis;
    }
    if (looksLikeEphemeralTaskRequest(user)) {
      return new MessageAnalysis(MessageIntent.CHAT, null);
    }
    return analysis;
  }

  public static boolean looksLikeFactDenseShare(String user) {
    if (user == null || user.isBlank() || looksLikeEphemeralTaskRequest(user)) {
      return false;
    }
    int hints = approximateClaimHints(user);
    return hints >= 2 || (user.strip().length() >= 80 && hints >= 1);
  }

  /**
   * One-shot task / coding request for the assistant to do now — not lasting memory.
   * Checked before fact-dense promotion so long "write a program…" messages stay CHAT.
   */
  public static boolean looksLikeEphemeralTaskRequest(String user) {
    if (user == null || user.isBlank()) {
      return false;
    }
    String text = user.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    if (text.endsWith("?")) {
      return false;
    }
    if (hasLastingPolicyCue(text)) {
      return false;
    }
    return startsWithAny(
        text,
        "print ", "list ", "show ", "write ", "generate ", "draw ", "output ", "display ",
        "make a ", "make an ", "give me ", "tell me ", "create a ", "create an ",
        "implement ", "code ", "please print", "please list", "please show",
        "please write", "please generate", "please create", "please implement"
    );
  }

  public static int approximateClaimHints(String user) {
    int hints = 0;
    for (String part : user.split("[,.;]+")) {
      if (part.strip().length() >= 12) {
        hints++;
      }
    }
    return Math.max(hints, 1);
  }

  private static boolean hasLastingPolicyCue(String lowerText) {
    return lowerText.contains("add rule")
        || lowerText.contains("remember that")
        || lowerText.contains("from now")
        || lowerText.contains("whenever")
        || lowerText.contains("when you")
        || lowerText.startsWith("always ")
        || lowerText.contains(" always ")
        || lowerText.startsWith("never ")
        || lowerText.contains(" never ");
  }

  private static boolean startsWithAny(String text, String... prefixes) {
    for (String prefix : prefixes) {
      if (text.startsWith(prefix)) {
        return true;
      }
    }
    return false;
  }

  public static MessageAnalysis parseMessageAnalysis(String raw) {
    if (raw == null || raw.isBlank()) {
      return new MessageAnalysis(MessageIntent.CHAT, null);
    }
    MessageIntent intent = MessageIntent.CHAT;
    String probe = null;
    boolean sawTypedIntent = false;
    for (String line : raw.strip().split("\\R")) {
      String text = line.strip();
      if (text.isEmpty()) {
        continue;
      }
      if (startsWithIgnoreCase(text, "TYPE:")) {
        intent = parseIntentToken(text.substring("TYPE:".length()).strip());
        sawTypedIntent = true;
      } else if (startsWithIgnoreCase(text, "PROBE:")) {
        String value = text.substring("PROBE:".length()).strip();
        if (!value.isBlank()) {
          probe = value;
        }
      } else if (!sawTypedIntent) {
        // Gemma often answers with a bare label: "STORE" / "QUESTION"
        MessageIntent bare = parseIntentToken(text);
        if (bare != MessageIntent.CHAT || equalsIgnoreCase(firstWord(text), "CHAT")) {
          intent = bare;
          sawTypedIntent = true;
        }
      }
    }
    if (intent != MessageIntent.FORGET) {
      probe = null;
    }
    return new MessageAnalysis(intent, probe);
  }

  private static String firstWord(String text) {
    if (text == null || text.isBlank()) {
      return "";
    }
    int end = 0;
    while (end < text.length() && isAsciiLetter(text.charAt(end))) {
      end++;
    }
    return text.substring(0, end);
  }

  private static MessageIntent parseIntentToken(String token) {
    if (token == null || token.isBlank()) {
      return MessageIntent.CHAT;
    }
    String word = token.strip();
    int end = 0;
    while (end < word.length() && isAsciiLetter(word.charAt(end))) {
      end++;
    }
    if (end == 0) {
      // Non-Latin garbage after "TYPE:" (common on Gemma-270M)
      return MessageIntent.CHAT;
    }
    String name = word.substring(0, end);
    if (equalsIgnoreCase(name, "STORE")) {
      return MessageIntent.STORE;
    }
    if (equalsIgnoreCase(name, "QUESTION")) {
      return MessageIntent.QUESTION;
    }
    if (equalsIgnoreCase(name, "FORGET")) {
      return MessageIntent.FORGET;
    }
    if (equalsIgnoreCase(name, "SKIP")) {
      return MessageIntent.SKIP;
    }
    if (equalsIgnoreCase(name, "CHAT")) {
      return MessageIntent.CHAT;
    }
    return MessageIntent.CHAT;
  }

  private static boolean isAsciiLetter(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
  }

  private static boolean startsWithIgnoreCase(String text, String prefix) {
    return text.length() >= prefix.length()
        && text.regionMatches(true, 0, prefix, 0, prefix.length());
  }

  private static boolean equalsIgnoreCase(String a, String b) {
    return a != null && a.equalsIgnoreCase(b);
  }
}
