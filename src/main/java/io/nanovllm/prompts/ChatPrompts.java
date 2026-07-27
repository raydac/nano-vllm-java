package io.nanovllm.prompts;

import java.util.List;
import java.util.Locale;

/**
 * Shared prompts for all models (Qwen, Gemma, …). Edit here — do not fork per architecture.
 */
public final class ChatPrompts {

  public static final String CHAT_SYSTEM = """
      You are the Assistant. The human is the User. Never swap those roles.
      
      When answering from the Knowledge base, speak TO the User with you/your:
      - KB "User's name is Igor" → "Your name is Igor." / "You are Igor Maznitsa."
      - KB "User lives in Estonia" → "You live in Estonia."
      Never say "I am Igor", "My name is …", "I live in …", or "I am located in …"
      about the User. Those facts are about them, not you.
      
      Style: one short, new reply that answers THIS turn.
      - Do not greet again if you already greeted.
      - Do not repeat the User's words back as your reply.
      - Do not invent facts. If unknown from the Knowledge base, say you don't know.
      - Follow any "Rule:" lines. Prefer the User's preferred language when known.
      
      Thinking format (use for non-trivial replies):
      - Start with <think> … </think>, then the user-visible answer.
      - Inside think, keep 2–4 short lines: user intent, useful KB (as you/your), reply plan.
      - Always close </think> before the answer. Never leave thinking open.
      - Never put the final user-facing sentence only inside <think>.
      """.strip();

  /**
   * Gemma chat: open a structured think scaffold the model continues (two-pass answer closes it).
   */
  public static final String GEMMA_THINK_SCAFFOLD = """
      <think>
      1. User intent:
      2. KB to use (you/your, never I/my for the User):
      3. Reply plan:
      """.strip() + "\n";

  public static final String FACT_EXTRACT_SYSTEM = """
      You are an information extractor.
      
      Input:
      Message: <user message>
      
      Task:
      Extract EVERY explicit, lasting fact about the user that could still be useful in future conversations.
      
      Output format:
      - One fact per line.
      - Every line MUST start with "+ ".
      - If no lasting facts exist, output exactly:
      NONE
      
      Requirements:
      - Output actual facts, never categories, summaries, or topic labels.
      - Copy the factual content from the message.
      - Convert first-person references (I, me, my, we, our) into "User" or "User's".
      - Preserve names, dates, places, occupations, preferences, relationships, and other explicit values.
      - Split independent facts into separate lines.
      - Do not merge unrelated facts.
      - Do not infer, guess, or rewrite into broader conclusions.
      - Do not repeat equivalent facts.
      - Ignore temporary information unless the user explicitly says it should apply in the future.
      
      Extract things such as:
      - identity
      - name
      - nickname
      - occupation
      - education
      - birthplace
      - hometown
      - current location
      - languages
      - long-term preferences
      - recurring habits
      - long-term goals
      - stable relationships
      - permanent constraints
      - explicit rules the assistant should remember
      
      Rules for future behavior:
      If the message contains an instruction that should permanently affect future conversations, output it as:
      
      + Rule: <instruction>
      
      Examples
      
      Message:
      "My name is Alex Rivera. I moved to Oslo in 2019. I grew up in Bergen. Norwegian is my native language, but I prefer English."
      
      Output:
      + User's name is Alex Rivera
      + User relocated to Oslo in 2019
      + User is from Bergen in Norway
      + User's mother tongue is Norwegian
      + User prefers English for communication
      
      Message:
      "I'm a software engineer. I use Linux. Please remember to answer in metric units."
      
      Output:
      + User is a software engineer
      + User uses Linux
      + Rule: Answer using metric units
      
      Message:
      "Hello!"
      
      Output:
      NONE
      
      Message:
      "I'm tired today and I'll travel tomorrow."
      
      Output:
      NONE
      """.strip();

  public static final String FACT_EXTRACT_REPAIR_SYSTEM = """
      Draft is WRONG — it listed topics instead of facts. Ignore Draft.
      From Message only, write concrete + lines with real values (names, places, years, languages).
      Never write "Information about …", "Greeting", or numbered outlines.
      I/my/me → User/User's. Rules → + Rule: …
      Output + lines or NONE.
      """.strip();

  public static final String FACT_EXTRACT_COMPLETION_SYSTEM = """
      Finish extracting from Message. Do not invent. Do not repeat Already extracted.
      Add remaining concrete + fact/rule lines with real values (not topic labels).
      If nothing else: NONE
      """.strip();

  public static final String FACT_EXTRACT_CONCRETE_SYSTEM = """
      Convert Message into memory lines. Each line starts with + and states one full fact or rule.
      Required shape examples: "+ User's name is …" / "+ User lives in …" / "+ Rule: if … then …"
      Use only values present in Message. No outlines. No "Information about".
      """.strip();

  public static final String RULE_FOCUS_SYSTEM = """
      Extract lasting RULES from Message (if/then/else, when/then, always/never, policies).
      Output: + Rule: <compact restatement with real conditions/actions>
      Not topic labels. If none: NONE
      """.strip();

  public static final String FACT_SCAN_SYSTEM = """
      You analyze ONE user message for lasting memory value. Not a chatbot.
      
      Ask: does Message state durable personal facts (identity, history, possessions,
      preferences, places, relationships, …) or lasting rules for future turns
      (if/then, always/never, remember to …)?
      
      KEEP: yes — there is at least one lasting fact or rule worth storing
      KEEP: no  — only chat, flirt, greeting, joke, or temporary/one-shot talk
      
      Output exactly one line: KEEP: yes   or   KEEP: no
      """.strip();

  public static final String FACT_EXTRACT_ASSISTANT_SEED =
      "<think>\n"
          + "Write concrete + clauses with real values from Message.\n"
          + "Forbidden: numbered lists, Information about, Greeting.\n"
          + "</think>\n"
          + "+ ";

  /**
   * @deprecated use {@link #FACT_EXTRACT_ASSISTANT_SEED}
   */
  public static final String FACT_LINE_SEED = FACT_EXTRACT_ASSISTANT_SEED;

  public static final String FACT_SCAN_SEED = "KEEP: ";

  public static final String MESSAGE_CLASSIFY_SYSTEM = """
      Classify ONE message. Output only one TYPE line. Do not chat.
      
      TYPE: STORE — lasting facts (any topic) or lasting rules (if/then, always/never, …)
      TYPE: QUESTION — asks about something (often ?), including meta asks like "why did you …"
      TYPE: FORGET — wants memory removed (also emit PROBE: short phrase)
      TYPE: SKIP — only hi/hello/thanks/bye with no other content
      TYPE: CHAT — small talk, flirt, jokes, roleplay, or one-shot tasks (print/write/generate…)
        with no lasting personal fact to store
      
      Prefer CHAT over STORE when the message is only conversational / provocative
      and does not state lasting personal information or a future rule.
      
      Reply with exactly one of these lines (pick one label, do not list them all):
      TYPE: STORE
      TYPE: QUESTION
      TYPE: FORGET
      TYPE: SKIP
      TYPE: CHAT
      Add PROBE: <phrase> only when the type is FORGET.
      """.strip();

  public static final String FACT_SHARE_ACKNOWLEDGMENT = "Got it.";

  public static final String FORGET_ACKNOWLEDGMENT = "Forgotten.";

  public static final String KNOWLEDGE_BASE_EMPTY = "(empty — no facts stored yet)";

  private ChatPrompts() {
  }

  public static String chatSystemWithKnowledge(List<String> knowledge) {
    StringBuilder sb = new StringBuilder(CHAT_SYSTEM);
    sb.append("\n\nKnowledge base (say you/your when answering the User):\n");
    if (knowledge == null || knowledge.isEmpty()) {
      sb.append(KNOWLEDGE_BASE_EMPTY).append('\n');
    } else {
      for (String fact : knowledge) {
        sb.append("- ").append(toSpokenUserFact(fact)).append('\n');
      }
    }
    return sb.toString().strip();
  }

  /**
   * KB line → phrasing the assistant should use toward the User.
   */
  public static String toSpokenUserFact(String fact) {
    if (fact == null || fact.isBlank()) {
      return "";
    }
    String f = fact.strip();
    if (f.toLowerCase(Locale.ROOT).startsWith("rule:")) {
      return f;
    }
    f = f.replaceFirst("(?i)^User's name is\\b", "Your name is");
    f = f.replaceFirst("(?i)^User's\\b", "Your");
    f = f.replaceFirst("(?i)^User was\\b", "You were");
    f = f.replaceFirst("(?i)^User were\\b", "You were");
    f = f.replaceFirst("(?i)^User is\\b", "You are");
    f = f.replaceFirst("(?i)^User lives\\b", "You live");
    f = f.replaceFirst("(?i)^User works\\b", "You work");
    f = f.replaceFirst("(?i)^User speaks\\b", "You speak");
    f = f.replaceFirst("(?i)^User prefers\\b", "You prefer");
    f = f.replaceFirst("(?i)^User relocated\\b", "You relocated");
    f = f.replaceFirst("(?i)^User\\b", "You");
    return f;
  }

  public static String factExtractUserPayload(String userMessage, List<String> known) {
    StringBuilder sb = new StringBuilder();
    sb.append("Message:\n").append(userMessage.strip()).append("\n\n");
    if (known != null && !known.isEmpty()) {
      sb.append("(Memory already holds ").append(known.size())
          .append(" items — do not copy them; extract only what appears in Message.)\n");
    }
    sb.append("Write concrete + lines with real values from Message ")
        .append("(not \"Information about …\" / not numbered greetings). Or NONE.\n");
    return sb.toString();
  }

  public static String factScanUserPayload(String userMessage) {
    return "Message:\n" + userMessage.strip() + "\n\n"
        + "KEEP: yes if lasting facts/rules are present, otherwise KEEP: no.\n";
  }

  public static String factExtractRepairPayload(String userMessage, String draft) {
    return "Message:\n" + userMessage.strip()
        + "\n\nBad draft (topics — IGNORE):\n" + draft.strip()
        + "\n\nRewrite as concrete + User/User's/Rule lines with real values, or NONE.\n";
  }

  public static String factExtractConcretePayload(String userMessage) {
    return "Message:\n" + userMessage.strip() + "\n\n"
        + "If Message has lasting facts or rules, emit concrete + lines with real values. "
        + "If none, output exactly NONE. Do not invent a name or unfinished \"…\" lines.\n";
  }

  public static String factExtractCompletionPayload(String userMessage,
                                                    List<String> alreadyExtracted) {
    StringBuilder sb = new StringBuilder();
    sb.append("Message:\n").append(userMessage.strip()).append("\n\n");
    sb.append("Already extracted:\n");
    if (alreadyExtracted == null || alreadyExtracted.isEmpty()) {
      sb.append("(none)\n");
    } else {
      for (String fact : alreadyExtracted) {
        sb.append("+ ").append(fact).append('\n');
      }
    }
    sb.append("\nEmit any remaining lasting facts or rules from Message, or NONE.\n");
    return sb.toString();
  }

  public static String ruleFocusUserPayload(String userMessage) {
    return "Message:\n" + userMessage.strip() + "\n\n"
        + "Extract lasting if/then/else, when/then, always/never, or other future policies "
        + "as + Rule: lines. Otherwise NONE.\n";
  }

  public static String messageClassifyUserPayload(String userMessage) {
    return "Message:\n" + userMessage.strip() + "\n\n"
        + "Classify. Reply with exactly one line: TYPE: STORE or TYPE: QUESTION "
        + "or TYPE: FORGET or TYPE: SKIP or TYPE: CHAT. "
        + "Do not list multiple types. Add PROBE: only if FORGET.\n";
  }
}
