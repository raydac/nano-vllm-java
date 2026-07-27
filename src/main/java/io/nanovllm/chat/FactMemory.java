package io.nanovllm.chat;

import static io.nanovllm.prompts.ChatPrompts.FACT_EXTRACT_ASSISTANT_SEED;
import static io.nanovllm.prompts.ChatPrompts.FACT_EXTRACT_COMPLETION_SYSTEM;
import static io.nanovllm.prompts.ChatPrompts.FACT_EXTRACT_CONCRETE_SYSTEM;
import static io.nanovllm.prompts.ChatPrompts.FACT_EXTRACT_REPAIR_SYSTEM;
import static io.nanovllm.prompts.ChatPrompts.FACT_EXTRACT_SYSTEM;
import static io.nanovllm.prompts.ChatPrompts.FACT_SCAN_SEED;
import static io.nanovllm.prompts.ChatPrompts.FACT_SCAN_SYSTEM;
import static io.nanovllm.prompts.ChatPrompts.RULE_FOCUS_SYSTEM;

import io.nanovllm.LLM;
import io.nanovllm.SamplingParams;
import io.nanovllm.prompts.ChatPrompts;
import io.nanovllm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class FactMemory {

  private static final int SCAN_MAX_TOKENS = 8;

  private static final Pattern REMEMBER_ANY = Pattern.compile("(?i)\\bREMEMBER:\\s*([^\\n]+)");
  private static final Pattern FORGET_ANY = Pattern.compile("(?i)\\bFORGET:\\s*([^\\n]+)");
  private static final Pattern FACT_PLUS_SEGMENTS = Pattern.compile("(?i)\\+\\s*([^+\\n]+)");
  private static final Pattern FACT_LINE = Pattern.compile("(?im)^\\s*FACT:\\s*(.+?)\\s*$");
  private static final Pattern FACT_PIPE_LINE = Pattern.compile("(?im)^\\s*F\\|\\s*(.+?)\\s*$");
  private static final Pattern FACT_INLINE = Pattern.compile("(?i)(?:\\bFACT:|\\+)\\s*([^\\n]+)");
  private static final Pattern BLANK_LINES = Pattern.compile("\\n{3,}");
  private static final Pattern ONLY_SPACES_LINE = Pattern.compile("(?m)^[ \\t]+$");
  private static final Pattern FACT_GROUNDING_STOPWORD = Pattern.compile(
      "(?i)^(user|user's|was|were|is|are|a|an|the|in|to|from|and|or|of|for|with|rule|that|this|than|into)$"
  );
  private static final Pattern LIVING_IN_COMPOUND =
      Pattern.compile("(?i)^(.+?)\\s+living\\s+in\\s+(.+)$");
  private static final Pattern SMALL_PLACE_WORD = Pattern.compile("(?i)^(in|of|the|and|from|to)$");
  private static final Pattern RELOCATE_FACT = Pattern.compile(
      "(?i)^user relocated to (.+?)(?: from (.+))? in (\\d{4})$"
  );
  private static final Pattern BORN_IN_YEAR = Pattern.compile(
      "(?i)^(.+?)(?:,)?\\s+(?:who\\s+)?(?:was\\s+)?born\\s+in\\s+(\\d{4})\\s*$"
  );

  private static final Pattern[] STRIP_USER_MEMORY_CUES = {
      Pattern.compile("(?i)^\\s*(please\\s+)?forget(\\s+that|\\s+about)?\\s+"),
      Pattern.compile("(?i)[,.]?\\s*(and\\s+)?it\\s+is\\s+me\\s*[.!?]*\\s*$"),
  };

  private static final Pattern[] FIRST_PERSON_PATTERNS = {
      Pattern.compile("(?i)^\\s*(i am|i'm)\\b"),
      Pattern.compile("(?i)^\\s*i was\\b"),
      Pattern.compile("(?i)^\\s*i were\\b"),
      Pattern.compile("(?i)^\\s*i had\\b"),
      Pattern.compile("(?i)^\\s*i live\\b"),
      Pattern.compile("(?i)^\\s*i work\\b"),
      Pattern.compile("(?i)^\\s*i have\\b"),
      Pattern.compile("(?i)^\\s*my name is\\b"),
      Pattern.compile("(?i)^\\s*i\\b"),
  };
  private static final String[] FIRST_PERSON_REPLACEMENTS = {
      "User is", "User was", "User was", "User had", "User lives",
      "User works", "User has", "User's name is", "User",
  };

  private FactMemory() {
  }

  public static int extractAndStore(
      LLM llm,
      Tokenizer tokenizer,
      List<String> knowledge,
      String user,
      SamplingParams extractParams
  ) {
    if (user == null || user.isBlank()) {
      return 0;
    }

    if (!scanHasLastingContent(llm, tokenizer, user, extractParams)) {
      System.err.println("(fact-scan) KEEP: no — skip extract");
      return 0;
    }

    System.err.println("(fact-extract LLM…)");
    String raw = ChatMessages.complete(
        llm,
        tokenizer,
        FACT_EXTRACT_SYSTEM,
        ChatPrompts.factExtractUserPayload(user, knowledge),
        extractParams,
        FACT_EXTRACT_ASSISTANT_SEED
    );
    System.err.println("(fact-extract) raw → " + ChatMessages.oneLineSummary(raw, 160));
    List<String> fromModel = groundedFacts(parseExtractedFacts(raw), user);

    if (fromModel.isEmpty() && isExplicitNone(raw)) {
      return 0;
    }

    if (needsExtractRepair(raw, fromModel, user)) {
      System.err.println("(fact-extract repair…)");
      String repaired = ChatMessages.complete(
          llm,
          tokenizer,
          FACT_EXTRACT_REPAIR_SYSTEM,
          ChatPrompts.factExtractRepairPayload(user, raw),
          extractParams,
          FACT_EXTRACT_ASSISTANT_SEED
      );
      System.err.println("(fact-extract) repair → " + ChatMessages.oneLineSummary(repaired, 160));
      fromModel = preferRicherExtraction(
          fromModel, groundedFacts(parseExtractedFacts(repaired), user));
    }

    if (fromModel.isEmpty()) {
      System.err.println("(fact-extract concrete…)");
      String concrete = ChatMessages.complete(
          llm,
          tokenizer,
          FACT_EXTRACT_CONCRETE_SYSTEM,
          ChatPrompts.factExtractConcretePayload(user),
          extractParams,
          FACT_EXTRACT_ASSISTANT_SEED
      );
      System.err.println("(fact-extract) concrete → " + ChatMessages.oneLineSummary(concrete, 160));
      fromModel = groundedFacts(parseExtractedFacts(concrete), user);
    }

    if (shouldSeekRemainingFacts(user, fromModel)) {
      for (int pass = 0; pass < 2 && shouldSeekRemainingFacts(user, fromModel); pass++) {
        System.err.println("(fact-extract completion…)");
        List<String> more = groundedFacts(
            extractWithModel(llm, tokenizer, FACT_EXTRACT_COMPLETION_SYSTEM,
                ChatPrompts.factExtractCompletionPayload(user, fromModel),
                extractParams,
                FACT_EXTRACT_ASSISTANT_SEED),
            user
        );
        if (more.isEmpty()) {
          break;
        }
        fromModel = mergeUnique(fromModel, more);
      }
    }

    if (looksLikeLastingRule(user)) {
      System.err.println("(fact-extract rule-focus…)");
      List<String> rules = groundedFacts(
          extractWithModel(llm, tokenizer, RULE_FOCUS_SYSTEM,
              ChatPrompts.ruleFocusUserPayload(user), extractParams, "+ Rule: "),
          user
      );
      fromModel = mergeUnique(fromModel, rules);
    }

    return storeUnknownCompactFacts(knowledge, fromModel);
  }

  /**
   * LLM gate: lasting facts/rules worth extracting? Replaces heuristic cue lists.
   */
  public static boolean scanHasLastingContent(
      LLM llm,
      Tokenizer tokenizer,
      String user,
      SamplingParams extractParams
  ) {
    System.err.println("(fact-scan…)");
    SamplingParams scanParams = new SamplingParams(
        extractParams.temperature(),
        SCAN_MAX_TOKENS,
        extractParams.ignoreEos(),
        extractParams.topK(),
        extractParams.topP()
    );
    String answer = ChatMessages.complete(
        llm,
        tokenizer,
        FACT_SCAN_SYSTEM,
        ChatPrompts.factScanUserPayload(user),
        scanParams,
        FACT_SCAN_SEED
    );
    boolean keep = parseKeepScan(answer);
    System.err.println("(fact-scan) " + (keep ? "KEEP: yes" : "KEEP: no")
        + " ← " + ChatMessages.oneLineSummary(answer, 80));
    return keep;
  }

  public static boolean parseKeepScan(String raw) {
    if (raw == null || raw.isBlank()) {
      return false;
    }
    String text = raw.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    Matcher keep = Pattern.compile("keep:\\s*(yes|no)\\b").matcher(text);
    if (keep.find()) {
      return "yes".equals(keep.group(1));
    }
    if (text.startsWith("yes")) {
      return true;
    }
    if (text.startsWith("no")) {
      return false;
    }
    return text.contains("yes") && !text.contains("no");
  }

  /**
   * @deprecated prefer {@link #scanHasLastingContent}; kept for unit tests of cue helpers.
   */
  public static boolean hasLastingContentCues(String user) {
    return MessageClassifier.looksLikePersonalFactShare(user)
        || MessageClassifier.looksLikeFactDenseShare(user)
        || looksLikeLastingRule(user);
  }

  public static boolean needsExtractRepair(String raw, List<String> grounded, String user) {
    if (isExplicitNone(raw)) {
      return false;
    }
    int found = grounded == null ? 0 : grounded.size();
    // Already have usable facts — do not spend another full LLM pass.
    if (found >= 2) {
      return false;
    }
    if (looksLikeTopicOutline(raw) && found == 0) {
      return true;
    }
    if (found == 0) {
      return true;
    }
    int hints = MessageClassifier.approximateClaimHints(user);
    int plusSegments = countPlusSegments(raw);
    if (plusSegments >= found + 3 && found == 0) {
      return true;
    }
    return hints >= 3 && found < Math.min(hints, 4);
  }

  /**
   * True for topic outlines ("+ 1 Greeting + 2 Information about …").
   * Numbered <em>real</em> facts ("+ 1 User's name is …") are NOT outlines.
   */
  public static boolean looksLikeTopicOutline(String raw) {
    if (raw == null || raw.isBlank()) {
      return false;
    }
    String lower = raw.toLowerCase(Locale.ROOT);
    if (lower.contains("information about")
        || lower.contains("polite greeting")) {
      return true;
    }
    int topicLike = 0;
    int factLike = 0;
    for (String part : raw.split("\\+")) {
      String rawPart = part.strip();
      if (rawPart.isEmpty()) {
        continue;
      }
      String p = stripLeadingListIndex(rawPart);
      if (p.isEmpty()) {
        continue;
      }
      String pl = p.toLowerCase(Locale.ROOT);
      if (pl.equals("greetings")
          || pl.equals("greeting")
          || pl.startsWith("information about")
          || pl.equals("relocation")
          || pl.equals("mother tongue")) {
        topicLike++;
        continue;
      }
      if (pl.startsWith("user ")
          || pl.startsWith("user's ")
          || pl.startsWith("rule:")
          || isValidFact(p)) {
        factLike++;
      } else if (Character.isDigit(rawPart.charAt(0))) {
        topicLike++;
      }
    }
    return topicLike >= 2 && factLike == 0;
  }

  static String stripLeadingListIndex(String text) {
    if (text == null || text.isBlank()) {
      return "";
    }
    return text.strip().replaceFirst("^\\d+[.)]?\\s+", "").strip();
  }

  static int countPlusSegments(String raw) {
    if (raw == null || raw.isBlank()) {
      return 0;
    }
    int count = 0;
    for (String part : raw.split("\\+")) {
      if (!part.isBlank()) {
        count++;
      }
    }
    return count;
  }

  private static List<String> preferRicherExtraction(List<String> first, List<String> repaired) {
    if (repaired == null || repaired.isEmpty()) {
      return first == null ? List.of() : first;
    }
    if (first == null || first.isEmpty() || repaired.size() > first.size()) {
      return repaired;
    }
    return mergeUnique(first, repaired);
  }

  public static boolean looksLikeLastingRule(String user) {
    if (user == null || user.isBlank() || MessageClassifier.looksLikeEphemeralTaskRequest(user)) {
      return false;
    }
    String text = user.strip().toLowerCase(Locale.ROOT);
    return text.contains("add rule")
        || text.contains("remember that")
        || text.contains("from now")
        || text.contains("whenever")
        || text.contains(" if ")
        || text.startsWith("if ")
        || text.contains(" then ")
        || text.contains(" else ")
        || text.contains("when you")
        || text.contains("when i")
        || text.startsWith("always ")
        || text.contains(" always ")
        || text.startsWith("never ")
        || text.contains(" never ");
  }

  private static boolean isExplicitNone(String raw) {
    if (raw == null || raw.isBlank()) {
      return true;
    }
    String r = raw.strip().replaceFirst("^\\+\\s*", "");
    return r.equalsIgnoreCase("NONE") || r.equals("+") || r.isEmpty();
  }

  private static List<String> extractWithModel(
      LLM llm,
      Tokenizer tokenizer,
      String systemPrompt,
      String userPayload,
      SamplingParams extractParams,
      String assistantPrefix
  ) {
    String answer = ChatMessages.complete(
        llm, tokenizer, systemPrompt, userPayload, extractParams, assistantPrefix);
    List<String> parsed = parseExtractedFacts(answer);
    System.err.println("(fact-extract) raw → " + ChatMessages.oneLineSummary(answer, 160));
    return parsed;
  }

  public static boolean shouldSeekRemainingFacts(String user, List<String> found) {
    if (user == null || user.isBlank() || found == null || found.isEmpty()) {
      return false;
    }
    return MessageClassifier.approximateClaimHints(user) > found.size();
  }

  private static List<String> mergeUnique(List<String> first, List<String> second) {
    List<String> merged = new ArrayList<>(first == null ? List.of() : first);
    if (second == null) {
      return merged;
    }
    for (String fact : second) {
      if (merged.stream().noneMatch(f -> f.equalsIgnoreCase(fact))) {
        merged.add(fact);
      }
    }
    return merged;
  }

  private static List<String> groundedFacts(List<String> facts, String user) {
    List<String> grounded = new ArrayList<>();
    for (String fact : facts) {
      if (isGroundedInUserMessage(fact, user)) {
        grounded.add(fact);
      }
    }
    return grounded;
  }

  public static boolean isGroundedInUserMessage(String fact, String user) {
    if (!isValidFact(fact) || user == null || user.isBlank()) {
      return false;
    }
    if (fact.strip().toLowerCase(Locale.ROOT).startsWith("rule:")) {
      return !MessageClassifier.looksLikeEphemeralTaskRequest(user)
          && enoughOverlap(significantContentWords(user),
          normalizeHaystack(fact.replaceFirst("(?i)^rule:\\s*", "")));
    }
    return enoughOverlap(significantContentWords(fact), normalizeHaystack(user));
  }

  private static boolean enoughOverlap(List<String> needles, String haystack) {
    int significant = 0;
    int hits = 0;
    for (String word : needles) {
      significant++;
      if (haystack.contains(word)) {
        hits++;
      }
    }
    return significant > 0 && hits * 2 >= significant;
  }

  private static String normalizeHaystack(String text) {
    return text.toLowerCase(Locale.ROOT).replace('-', ' ').replaceAll("\\s+", " ");
  }

  private static List<String> significantContentWords(String text) {
    List<String> words = new ArrayList<>();
    for (String word : text.toLowerCase(Locale.ROOT)
        .replaceAll("[^\\p{L}\\p{N}\\s]", " ")
        .replace('-', ' ')
        .split("\\s+")) {
      if (word.length() < 4 || FACT_GROUNDING_STOPWORD.matcher(word).matches()) {
        continue;
      }
      words.add(word);
    }
    return words;
  }

  private static int storeUnknownCompactFacts(List<String> knowledge, List<String> extracted) {
    int added = 0;
    for (String fact : compactFacts(extracted)) {
      if (!isUnknownFact(knowledge, fact)) {
        continue;
      }
      knowledge.removeIf(existing -> isRedundantAgainst(existing, List.of(fact)));
      rememberFact(knowledge, fact);
      System.err.println("(knowledge+) " + fact);
      added++;
    }
    if (added == 0) {
      System.err.println("(fact-extract) no new facts");
    }
    return added;
  }

  public static List<String> parseExtractedFacts(String extractorAnswer) {
    List<String> facts = new ArrayList<>();
    if (extractorAnswer == null || extractorAnswer.isBlank()) {
      return facts;
    }
    String text = extractorAnswer.strip();
    if (text.equalsIgnoreCase("NONE") || text.equalsIgnoreCase("FACT: NONE")) {
      return facts;
    }

    boolean any = collectFactMatches(FACT_PLUS_SEGMENTS, text, facts);
    any |= collectFactMatches(FACT_LINE, text, facts);
    any |= collectFactMatches(FACT_PIPE_LINE, text, facts);
    if (!any) {
      Matcher inline = FACT_INLINE.matcher(text);
      while (inline.find()) {
        addParsedFact(facts, inline.group(1));
      }
    }
    // Models sometimes omit "+" but still write "User's name is …"
    if (facts.isEmpty()) {
      for (String line : text.split("\\R")) {
        String cleaned = line.strip().replaceAll("^[`*]+|[`*]+$", "").strip();
        cleaned = cleaned.replaceFirst("^\\+\\s*", "");
        if (cleaned.toLowerCase(Locale.ROOT).startsWith("user's ")
            || cleaned.toLowerCase(Locale.ROOT).startsWith("user ")
            || cleaned.toLowerCase(Locale.ROOT).startsWith("rule:")) {
          addParsedFact(facts, cleaned);
        }
      }
    }
    if (facts.isEmpty() && text.toLowerCase(Locale.ROOT).startsWith("rule:")) {
      addParsedFact(facts, text);
    }
    return expandNormalizedFacts(facts);
  }

  private static boolean collectFactMatches(Pattern pattern, String text, List<String> facts) {
    Matcher m = pattern.matcher(text);
    boolean any = false;
    while (m.find()) {
      any = true;
      addParsedFact(facts, m.group(1));
    }
    return any;
  }

  private static void addParsedFact(List<String> facts, String raw) {
    String cleaned = stripLeadingListIndex(cleanDirectiveValue(raw));
    if (cleaned.equalsIgnoreCase("NONE") || !isValidFact(cleaned)) {
      return;
    }
    facts.add(cleaned);
  }

  public static List<String> expandNormalizedFacts(List<String> rawFacts) {
    List<String> out = new ArrayList<>();
    if (rawFacts == null) {
      return out;
    }
    for (String raw : rawFacts) {
      for (String piece : splitAtomicFacts(normalizeFact(raw))) {
        if (isValidFact(piece) && out.stream().noneMatch(f -> f.equalsIgnoreCase(piece))) {
          out.add(piece);
        }
      }
    }
    return out;
  }

  public static List<String> compactFacts(List<String> rawFacts) {
    List<String> pieces = expandNormalizedFacts(rawFacts);
    pieces.sort(java.util.Comparator.comparingInt(String::length));
    List<String> kept = new ArrayList<>();
    for (String candidate : pieces) {
      if (kept.stream().anyMatch(f -> f.equalsIgnoreCase(candidate))) {
        continue;
      }
      if (isRedundantAgainst(candidate, kept)) {
        continue;
      }
      kept.removeIf(existing -> isRedundantAgainst(existing, List.of(candidate)));
      kept.add(candidate);
    }
    List<String> compact = new ArrayList<>();
    for (String fact : kept) {
      List<String> others = kept.stream().filter(f -> !f.equalsIgnoreCase(fact)).toList();
      if (!isRedundantAgainst(fact, others)) {
        compact.add(fact);
      }
    }
    return compact;
  }

  public static boolean isRedundantAgainst(String fact, List<String> others) {
    if (!isValidFact(fact) || others == null || others.isEmpty()) {
      return false;
    }
    String fl = fact.strip().toLowerCase(Locale.ROOT);
    for (String other : others) {
      if (!isValidFact(other)) {
        continue;
      }
      String ol = other.strip().toLowerCase(Locale.ROOT);
      if (fl.equals(ol) || isSubstantialNearDuplicate(fl, ol)
          || isLongerCompoundCoveredByAtomic(fl, ol, others)
          || isWeakerRelocateFact(fl, ol)) {
        return true;
      }
    }
    return isCoveredByMultipleAtomics(fl, others);
  }

  private static boolean isSubstantialNearDuplicate(String a, String b) {
    if (!(b.contains(a) || a.contains(b))) {
      return false;
    }
    int shorter = Math.min(a.length(), b.length());
    int longer = Math.max(a.length(), b.length());
    return shorter >= 12 && shorter * 10 >= longer * 7;
  }

  private static boolean isLongerCompoundCoveredByAtomic(String factLower, String atomicLower,
                                                         List<String> others) {
    if (factLower.length() <= atomicLower.length() + 3 || atomicLower.length() < 10) {
      return false;
    }
    return factLower.contains(atomicLower) &&
        compoundRemainderCovered(factLower, atomicLower, others);
  }

  private static boolean isCoveredByMultipleAtomics(String factLower, List<String> others) {
    int hits = 0;
    for (String other : others) {
      String ol = other.strip().toLowerCase(Locale.ROOT);
      if (ol.length() >= 10 && factLower.length() > ol.length() + 3 && factLower.contains(ol)) {
        hits++;
      }
    }
    return hits >= 2;
  }

  private static boolean isWeakerRelocateFact(String factLower, String otherLower) {
    Matcher fact = RELOCATE_FACT.matcher(factLower);
    Matcher other = RELOCATE_FACT.matcher(otherLower);
    if (!fact.matches() || !other.matches()) {
      return false;
    }
    if (!fact.group(1).equalsIgnoreCase(other.group(1)) || !fact.group(3).equals(other.group(3))) {
      return false;
    }
    return fact.group(2) == null && other.group(2) != null;
  }

  private static boolean compoundRemainderCovered(String factLower, String atomicLower,
                                                  List<String> others) {
    String remainder = factLower.replace(atomicLower, " ").replaceAll("\\s+", " ").strip();
    remainder = remainder.replaceAll("(?i)^user\\s+", "").strip();
    remainder = remainder.replaceAll("(?i)^(is\\s+a\\s+|is\\s+an\\s+|is\\s+)", "").strip();
    if (remainder.isEmpty()) {
      return true;
    }
    Matcher living = Pattern.compile("(?i)^living\\s+in\\s+(.+)$").matcher(remainder);
    if (living.matches()) {
      String place = living.group(1).strip();
      return others.stream().anyMatch(o -> {
        String ol = o.toLowerCase(Locale.ROOT);
        return ol.contains("lives in " + place) || ol.contains("living in " + place);
      });
    }
    for (String other : others) {
      String ol = other.strip().toLowerCase(Locale.ROOT);
      if (ol.equals(atomicLower) || ol.length() < 8) {
        continue;
      }
      if (remainder.contains(ol) || ol.contains(remainder)) {
        return true;
      }
    }
    return false;
  }

  public static String normalizeFact(String fact) {
    if (fact == null) {
      return "";
    }
    String f = stripLeadingListIndex(cleanDirectiveValue(fact));
    f = f.replaceFirst("(?i)^hello[,!]?\\s+", "");
    f = f.replaceAll("(?i)\\bit is me\\b", "is the User");
    f = f.replaceAll("(?i)\\bthis is me\\b", "is the User");
    f = f.replaceAll("(?i)\\bthat is me\\b", "is the User");
    f = f.replaceAll("(?i)\\bmyself\\b", "User");
    f = f.replaceAll("(?i)\\bmy\\b", "User's");
    f = f.replaceAll("(?i)\\bme\\b", "User");

    for (int i = 0; i < FIRST_PERSON_PATTERNS.length; i++) {
      if (FIRST_PERSON_PATTERNS[i].matcher(f).find()) {
        f = FIRST_PERSON_PATTERNS[i].matcher(f).replaceFirst(FIRST_PERSON_REPLACEMENTS[i]);
        break;
      }
    }

    f = f.replaceAll("(?i)^(.+?)\\s+is\\s+the\\s+User\\s*$", "User is $1");
    f = f.replaceFirst("(?i)^User's preferred language is\\b", "User prefers");
    if (f.toLowerCase(Locale.ROOT).matches("^user prefers \\S+$")) {
      f = f + " for communication";
    }
    f = f.replaceAll("\\s+", " ").strip();
    if (f.toLowerCase(Locale.ROOT).startsWith("assistant is ")
        && f.toLowerCase(Locale.ROOT).contains(" user")) {
      f = f.replaceFirst("(?i)^assistant\\b", "User");
    }
    return titleCaseKnownPlaceFacts(f);
  }

  private static String titleCaseKnownPlaceFacts(String fact) {
    Matcher fromPlace = Pattern.compile("(?i)^User is from (.+)$").matcher(fact);
    if (fromPlace.matches()) {
      return "User is from " + titleCasePlace(fromPlace.group(1));
    }
    Matcher livesIn = Pattern.compile("(?i)^User lives in (.+)$").matcher(fact);
    if (livesIn.matches()) {
      return "User lives in " + titleCasePlace(livesIn.group(1));
    }
    return fact;
  }

  private static String titleCasePlace(String text) {
    StringBuilder sb = new StringBuilder();
    for (String part : text.strip().split("\\s+")) {
      if (part.isEmpty()) {
        continue;
      }
      if (sb.length() > 0) {
        sb.append(' ');
      }
      if (SMALL_PLACE_WORD.matcher(part).matches()) {
        sb.append(part.toLowerCase(Locale.ROOT));
      } else if (part.contains("-")) {
        sb.append(titleCaseHyphenated(part));
      } else {
        sb.append(Character.toUpperCase(part.charAt(0)));
        if (part.length() > 1) {
          sb.append(part.substring(1).toLowerCase(Locale.ROOT));
        }
      }
    }
    return sb.toString();
  }

  private static String titleCaseHyphenated(String part) {
    StringBuilder sb = new StringBuilder();
    for (String bit : part.split("-")) {
      if (bit.isEmpty()) {
        continue;
      }
      if (sb.length() > 0) {
        sb.append('-');
      }
      sb.append(Character.toUpperCase(bit.charAt(0)));
      if (bit.length() > 1) {
        sb.append(bit.substring(1).toLowerCase(Locale.ROOT));
      }
    }
    return sb.toString();
  }

  public static List<String> splitAtomicFacts(String fact) {
    List<String> parts = new ArrayList<>();
    if (!isValidFact(fact)) {
      return parts;
    }
    String f = fact.strip();

    Matcher living = LIVING_IN_COMPOUND.matcher(f);
    if (living.matches()) {
      String head = living.group(1).strip();
      String place = living.group(2).strip();
      parts.add(normalizeFact(head));
      parts.add(normalizeFact(subjectOf(head) + " lives in " + place));
      return dedupeIgnoreCase(parts);
    }

    List<String> bornSplit = splitBornInYearCompound(f);
    if (bornSplit != null) {
      return bornSplit;
    }

    List<String> andSplit = splitAndJoinedClauses(f);
    if (andSplit != null) {
      return andSplit;
    }

    parts.add(f);
    return parts;
  }

  private static List<String> splitBornInYearCompound(String fact) {
    Matcher born = BORN_IN_YEAR.matcher(fact);
    if (!born.matches()) {
      return null;
    }
    String head = born.group(1).strip().replaceAll("(?i),\\s*$", "").strip();
    String year = born.group(2);
    List<String> parts = new ArrayList<>();
    if (head.toLowerCase(Locale.ROOT).matches(".*\\bis\\b.*")) {
      parts.add(normalizeFact(head));
      parts.add(normalizeFact(subjectOf(head) + " was born in " + year));
      return dedupeIgnoreCase(parts);
    }
    parts.add(normalizeFact(head + " was born in " + year));
    return dedupeIgnoreCase(parts);
  }

  private static List<String> splitAndJoinedClauses(String fact) {
    if (!fact.toLowerCase(Locale.ROOT).contains(" and ")) {
      return null;
    }
    String[] andParts = fact.split("(?i)\\s+and\\s+");
    if (andParts.length == 2
        && andParts[0].toLowerCase(Locale.ROOT).contains(" is ")
        && !andParts[1].toLowerCase(Locale.ROOT).contains(" is ")) {
      String left = andParts[0].strip();
      String right = andParts[1].strip();
      List<String> parts = new ArrayList<>();
      parts.add(normalizeFact(left));
      parts.add(normalizeFact(subjectOf(left) + " is " + right));
      return dedupeIgnoreCase(parts);
    }
    if (andParts.length >= 2) {
      boolean allClauses = true;
      for (String p : andParts) {
        if (!p.toLowerCase(Locale.ROOT).matches(".*\\b(is|was|are|were|has|lives|works)\\b.*")) {
          allClauses = false;
          break;
        }
      }
      if (allClauses) {
        List<String> parts = new ArrayList<>();
        for (String p : andParts) {
          parts.add(normalizeFact(p.strip()));
        }
        return dedupeIgnoreCase(parts);
      }
    }
    return null;
  }

  private static String subjectOf(String clause) {
    Matcher m = Pattern.compile("(?i)^(.+?)\\s+is\\b").matcher(clause.strip());
    if (m.find()) {
      return m.group(1).strip();
    }
    m = Pattern.compile("(?i)^(.+?)\\s+was\\b").matcher(clause.strip());
    if (m.find()) {
      return m.group(1).strip();
    }
    return "User";
  }

  private static List<String> dedupeIgnoreCase(List<String> facts) {
    List<String> out = new ArrayList<>();
    for (String fact : facts) {
      if (!isValidFact(fact)) {
        continue;
      }
      if (out.stream().noneMatch(f -> f.equalsIgnoreCase(fact))) {
        out.add(fact.strip());
      }
    }
    return out;
  }

  public static boolean isUnknownFact(List<String> knowledge, String fact) {
    return isValidFact(fact) && !isRedundantAgainst(fact, knowledge);
  }

  public static String stripUserMemoryCues(String user) {
    String fact = user.strip();
    boolean changed;
    do {
      changed = false;
      for (Pattern cue : STRIP_USER_MEMORY_CUES) {
        String next = cue.matcher(fact).replaceFirst("").strip();
        if (!next.equals(fact)) {
          fact = next;
          changed = true;
        }
      }
    } while (changed);
    return fact;
  }

  public static String forgetBestMatch(List<String> knowledge, String probe) {
    if (probe == null || probe.isBlank() || knowledge.isEmpty()) {
      return null;
    }
    if (forgetFact(knowledge, probe)) {
      return probe.strip();
    }
    String key = probe.toLowerCase(Locale.ROOT);
    String best = null;
    for (String fact : knowledge) {
      String f = fact.toLowerCase(Locale.ROOT);
      if (f.contains(key) || key.contains(f)) {
        if (best == null || fact.length() < best.length()) {
          best = fact;
        }
      }
    }
    if (best == null) {
      return null;
    }
    knowledge.remove(best);
    return best;
  }

  public static void applyKnowledgeDirectives(List<String> knowledge, String assistantAnswer) {
    if (assistantAnswer == null || assistantAnswer.isBlank()) {
      return;
    }
    Matcher remember = REMEMBER_ANY.matcher(assistantAnswer);
    while (remember.find()) {
      String fact = cleanDirectiveValue(remember.group(1));
      if (!isValidFact(fact) || !isUnknownFact(knowledge, fact)) {
        continue;
      }
      rememberFact(knowledge, fact);
      System.err.println("(knowledge+) " + fact);
    }
    Matcher forget = FORGET_ANY.matcher(assistantAnswer);
    while (forget.find()) {
      String requested = cleanDirectiveValue(forget.group(1));
      if (!isValidFact(requested)) {
        continue;
      }
      String removed = forgetBestMatch(knowledge, requested);
      if (removed != null) {
        System.err.println("(knowledge-) " + removed);
      }
    }
  }

  public static String cleanDirectiveValue(String raw) {
    return AssistantParts.stripSpecialTokens(raw);
  }

  public static boolean isValidFact(String fact) {
    if (fact == null) {
      return false;
    }
    String f = fact.strip();
    if (f.length() < 8) {
      return false;
    }
    String lower = f.toLowerCase(Locale.ROOT);
    if (lower.equals("(none)")
        || lower.equals("none")
        || lower.equals("n/a")
        || lower.equals("empty")
        || lower.equals("(empty)")
        || lower.equals("-")
        || lower.startsWith("(empty)")
        || lower.endsWith(" none")
        || lower.endsWith(" is none")
        || lower.equals("rule: line")
        || isTopicOutlineFact(lower)) {
      return false;
    }
    if (hasIncompletePlaceholder(f) || lower.endsWith("?")) {
      return false;
    }
    if (endsWithInterrogativeValue(f)) {
      return false;
    }

    String normalized = normalizeFact(f);
    String nLower = normalized.toLowerCase(Locale.ROOT);
    if (hasIncompletePlaceholder(normalized)) {
      return false;
    }

    String[] words = normalized.split("\\s+");
    boolean userScoped = nLower.startsWith("user ")
        || nLower.startsWith("user's ")
        || nLower.startsWith("rule:");
    // LLM decides fact shape; we only require a scoped subject and enough words.
    // Grounding (isGroundedInUserMessage) still blocks invented content.
    if (nLower.startsWith("rule:")) {
      return words.length >= 3 && nLower.length() >= 12;
    }
    if (userScoped) {
      return words.length >= 3;
    }
    return words.length >= 4;
  }

  private static boolean isTopicOutlineFact(String lowerFact) {
    String body = lowerFact.replaceFirst("^\\d+\\s*", "").strip();
    return body.startsWith("information about")
        || body.equals("greetings")
        || body.equals("greeting")
        || body.equals("polite greeting")
        || body.equals("relocation")
        || body.equals("mother tongue")
        || body.startsWith("information about");
  }

  private static boolean hasIncompletePlaceholder(String fact) {
    return fact.contains("…")
        || fact.contains("...")
        || fact.contains("___")
        || fact.matches("(?i).*\\bin\\s*$")
        || fact.matches("(?i).*\\bis\\s*$")
        || fact.matches("(?i).*\\bfrom\\s*$");
  }

  private static boolean endsWithInterrogativeValue(String fact) {
    String lower = fact.strip().toLowerCase(Locale.ROOT);
    int isAt = lower.lastIndexOf(" is ");
    if (isAt < 0) {
      return false;
    }
    String value = lower.substring(isAt + 4).strip();
    return value.equals("what")
        || value.equals("who")
        || value.equals("whom")
        || value.equals("which")
        || value.equals("where")
        || value.equals("when")
        || value.equals("why")
        || value.equals("how");
  }

  public static void rememberFact(List<String> knowledge, String fact) {
    String normalized = normalizeFact(fact);
    if (!isValidFact(normalized)) {
      return;
    }
    String key = normalized.toLowerCase(Locale.ROOT);
    knowledge.removeIf(f -> f.toLowerCase(Locale.ROOT).equals(key));
    knowledge.add(normalized);
  }

  public static boolean forgetFact(List<String> knowledge, String requested) {
    String want = requested.strip();
    if (want.isEmpty()) {
      return false;
    }
    for (int i = 0; i < knowledge.size(); i++) {
      if (knowledge.get(i).equals(want) || knowledge.get(i).equalsIgnoreCase(want)) {
        knowledge.remove(i);
        return true;
      }
    }
    return false;
  }

  public static String stripMemoryDirectives(String answer) {
    if (answer == null || answer.isBlank()) {
      return "";
    }
    String s = REMEMBER_ANY.matcher(answer).replaceAll("");
    s = FORGET_ANY.matcher(s).replaceAll("");
    s = ONLY_SPACES_LINE.matcher(s).replaceAll("");
    s = BLANK_LINES.matcher(s).replaceAll("\n\n");
    return s.strip();
  }

  public static String cleanAssistantText(String raw) {
    AssistantParts parts = AssistantParts.parse(raw);
    String answer = stripMemoryDirectives(parts.answer());
    return answer.isEmpty() ? parts.thinking().strip() : answer;
  }

  public static String streamDisplayText(String raw) {
    return cleanAssistantText(raw);
  }

  /**
   * Tiny models often answer with the User's identity in first person
   * ("I am X", "I live in X", "I am located in X"). Rewrite using the KB.
   */
  public static String rewriteMistakenFirstPersonIdentity(String answer, List<String> knowledge) {
    if (answer == null || answer.isBlank()) {
      return "";
    }
    String text = answer.strip();
    if (knowledge == null || knowledge.isEmpty()) {
      return text;
    }

    Matcher myName =
        Pattern.compile("(?i)^(?:hello[,!]?\\s+)?(?:my name is|i am)\\s+(.+?)\\.?\\s*$")
            .matcher(text);
    if (myName.matches()) {
      String claimed = myName.group(1).strip();
      // "I am located in …" is a location claim, not a name/role.
      if (!claimed.toLowerCase(Locale.ROOT).startsWith("located ")
          && !claimed.toLowerCase(Locale.ROOT).startsWith("in ")) {
        for (String fact : knowledge) {
          Matcher name = Pattern.compile("(?i)^User's name is\\s+(.+)$").matcher(fact.strip());
          if (name.matches() && name.group(1).strip().equalsIgnoreCase(claimed)) {
            return "Your name is " + name.group(1).strip() + ".";
          }
          Matcher is = Pattern.compile("(?i)^User is\\s+(.+)$").matcher(fact.strip());
          if (is.matches() && is.group(1).strip().equalsIgnoreCase(claimed)) {
            return "You are " + is.group(1).strip() + ".";
          }
        }
      }
    }

    Matcher live = Pattern.compile(
        "(?i)^(?:i am located in|i'm located in|i live in|i'm in|i am in)\\s+(.+?)\\.?\\s*$"
    ).matcher(text);
    if (live.matches()) {
      String claimedPlace = live.group(1).strip();
      for (String fact : knowledge) {
        Matcher lives = Pattern.compile("(?i)^User lives in\\s+(.+)$").matcher(fact.strip());
        if (lives.matches()) {
          String place = lives.group(1).strip();
          if (place.equalsIgnoreCase(claimedPlace)
              || claimedPlace.toLowerCase(Locale.ROOT).contains(place.toLowerCase(Locale.ROOT))
              || place.toLowerCase(Locale.ROOT).contains(claimedPlace.toLowerCase(Locale.ROOT))) {
            return "You live in " + place + ".";
          }
        }
      }
      // Model used first person for a home question — still flip pronoun if KB has a home.
      for (String fact : knowledge) {
        Matcher lives = Pattern.compile("(?i)^User lives in\\s+(.+)$").matcher(fact.strip());
        if (lives.matches()) {
          return "You live in " + lives.group(1).strip() + ".";
        }
      }
    }
    return text;
  }
}
