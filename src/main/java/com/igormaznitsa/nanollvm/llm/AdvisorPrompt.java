package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.prompts.AdvisorPrompts;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * Builds isolated advisor chat prompts and mixes advisor answers into the main user text.
 * Wording lives in {@link AdvisorPrompts} / {@link RagPrompts}.
 */
public final class AdvisorPrompt {

  private static final Pattern TOKEN = Pattern.compile("[\\p{L}\\p{N}]+");
  private static final int CONTENTFUL_MIN_LEN = 4;
  private static final double GROUNDING_COVERAGE = 0.6;

  private AdvisorPrompt() {
  }

  /**
   * Role prompt plus shared grounded-extraction rules used for every advisor turn.
   */
  public static String groundedRole(final String rolePrompt) {
    return AdvisorPrompts.groundedRole(rolePrompt);
  }

  /**
   * Isolated turn: grounded role as system (Gemma folds it into the first user turn) plus the
   * same prepared user text the main model will see — no conversation history.
   */
  public static String isolated(
    final Tokenizer tokenizer,
    final String rolePrompt,
    final String modelUserText
  ) {
    requireNonNull(tokenizer, "tokenizer");
    requireNonNull(modelUserText, "modelUserText");
    String user = modelUserText.strip();
    if (user.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }

    boolean ragNoHits = ragTurnWithoutHits(user);
    List<ChatMessage> turn = new ArrayList<>(ChatMessages.newConversation(
      AdvisorPrompts.groundedRole(rolePrompt, ragNoHits)));
    turn.add(ChatMessage.user(user));
    return tokenizer.applyChatTemplate(ChatMessages.toTemplateMaps(turn), true, false);
  }

  /**
   * Chooses which advisor notes may enter the main prompt.
   *
   * <ul>
   *   <li>RAG no-hit — none (notes stay on the thinking stream only)</li>
   *   <li>RAG with Context — drop abstentions and notes that are not lexically grounded in
   *       Context</li>
   *   <li>Plain chat — drop abstentions only</li>
   * </ul>
   */
  public static List<String> selectNotesForMix(
    final String modelUserText,
    final List<String> notes
  ) {
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(notes, "notes");
    if (ragTurnWithoutHits(modelUserText)) {
      return List.of();
    }

    boolean withContext = hasContextSection(modelUserText);
    Set<String> contextTerms = withContext
      ? contentfulTerms(extractContextBlock(modelUserText))
      : Set.of();

    return notes.stream()
      .map(note -> note == null ? "" : note.strip())
      .filter(note -> !note.isEmpty())
      .filter(note -> !isAbstention(note))
      .filter(note -> !withContext || isGroundedInContext(note, contextTerms))
      .toList();
  }

  /**
   * Mixes non-blank advisor notes with {@code modelUserText}. Empty answers leave the text
   * unchanged.
   */
  public static String mix(final String modelUserText, final List<String> answers) {
    return mix(modelUserText, answers, false);
  }

  /**
   * Mixes advisor notes so the main model does not treat them as ranked candidate answers.
   *
   * <p>For {@code compact} (tiny models), notes are <em>prepended</em> as unverified claims so the
   * original Context / answer instruction stays last — otherwise models copy {@code [1]}.
   */
  public static String mix(
    final String modelUserText,
    final List<String> answers,
    final boolean compact
  ) {
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(answers, "answers");
    String base = modelUserText.strip();
    if (base.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }

    List<String> claimLines = IntStream.range(0, answers.size())
      .mapToObj(i -> {
        String note = answers.get(i) == null ? "" : answers.get(i).strip();
        return note.isEmpty() ? "" : AdvisorPrompts.claimLine(claimLabel(i), note);
      })
      .filter(line -> !line.isEmpty())
      .toList();
    if (claimLines.isEmpty()) {
      return base;
    }

    String claims = String.join("\n", claimLines);
    return compact
      ? AdvisorPrompts.mixCompact(claims, base)
      : AdvisorPrompts.mixFull(base, claims);
  }

  static String claimLabel(final int index) {
    if (index < 0) {
      throw new IllegalArgumentException("index must be >= 0");
    }
    if (index < 26) {
      return String.valueOf((char) ('A' + index));
    }
    return Integer.toString(index + 1);
  }

  static boolean ragTurnWithoutHits(final String modelUserText) {
    if (modelUserText == null || modelUserText.isBlank()) {
      return false;
    }
    return modelUserText.contains(RagPrompts.NO_CONTEXT_DOCUMENTS);
  }

  static boolean isAbstention(final String note) {
    if (note == null) {
      return false;
    }
    String stripped = note.strip();
    if (AdvisorPrompts.ABSTAIN_REPLY.matcher(stripped).matches()) {
      return true;
    }
    String lower = stripped.toLowerCase(Locale.ROOT);
    String abstain = RagPrompts.ABSTAIN_REPLY.toLowerCase(Locale.ROOT);
    if (!lower.startsWith(abstain)) {
      return false;
    }
    if (stripped.length() == abstain.length()) {
      return true;
    }
    char next = stripped.charAt(abstain.length());
    return next == '.' || next == '!' || next == '?' || Character.isWhitespace(next);
  }

  static boolean hasContextSection(final String modelUserText) {
    if (ragTurnWithoutHits(modelUserText)) {
      return false;
    }
    String lower = modelUserText.toLowerCase(Locale.ROOT);
    return lower.contains(RagPrompts.CONTEXT_HEADING.toLowerCase(Locale.ROOT));
  }

  static String extractContextBlock(final String modelUserText) {
    if (modelUserText == null || modelUserText.isBlank()) {
      return "";
    }
    String heading = RagPrompts.CONTEXT_HEADING;
    int start = indexOfIgnoreCase(modelUserText, heading);
    if (start < 0) {
      return "";
    }
    int bodyStart = start + heading.length();
    int end = modelUserText.length();
    for (String marker : AdvisorPrompts.CONTEXT_BLOCK_END_MARKERS) {
      int at = modelUserText.indexOf(marker, bodyStart);
      if (at >= 0 && at < end) {
        end = at;
      }
    }
    int questionAt = indexOfIgnoreCase(modelUserText, RagPrompts.QUESTION_HEADING, bodyStart);
    if (questionAt >= 0 && questionAt < end) {
      end = questionAt;
    }
    return modelUserText.substring(bodyStart, end).strip();
  }

  static boolean isGroundedInContext(final String note, final Set<String> contextTerms) {
    if (contextTerms.isEmpty()) {
      return false;
    }
    Set<String> noteTerms = contentfulTerms(note);
    if (noteTerms.isEmpty()) {
      return false;
    }
    long hit = noteTerms.stream().filter(contextTerms::contains).count();
    return hit >= Math.max(1, (int) Math.ceil(noteTerms.size() * GROUNDING_COVERAGE));
  }

  static Set<String> contentfulTerms(final String text) {
    if (text == null || text.isBlank()) {
      return Set.of();
    }
    Set<String> terms = new HashSet<>();
    var matcher = TOKEN.matcher(text.toLowerCase(Locale.ROOT));
    while (matcher.find()) {
      String token = matcher.group();
      if (token.length() >= CONTENTFUL_MIN_LEN) {
        terms.add(token);
      }
    }
    return Set.copyOf(terms);
  }

  private static int indexOfIgnoreCase(final String haystack, final String needle) {
    return indexOfIgnoreCase(haystack, needle, 0);
  }

  private static int indexOfIgnoreCase(
    final String haystack,
    final String needle,
    final int fromIndex
  ) {
    return haystack.toLowerCase(Locale.ROOT)
      .indexOf(needle.toLowerCase(Locale.ROOT), fromIndex);
  }
}
