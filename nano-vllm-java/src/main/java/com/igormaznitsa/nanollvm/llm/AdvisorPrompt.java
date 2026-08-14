package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.chat.ChatRole;
import com.igormaznitsa.nanollvm.prompts.AdvisorPrompts;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;

/**
 * Builds isolated advisor chat prompts and mixes advisor answers into the main user text.
 * Role wording is entirely caller-supplied ({@link LlmAdvisor#prompt()}).
 */
final class AdvisorPrompt {

  private static final Pattern TOKEN = Pattern.compile("[\\p{L}\\p{N}]+");
  private static final int CONTENTFUL_MIN_LEN = 4;
  private static final double GROUNDING_COVERAGE = 0.6;

  private AdvisorPrompt() {
  }

  /**
   * Advisor turn: caller role prompt as system (turn-based templates fold into first user), prior user turns,
   * then advisor user payload (question + Context when RAG hit).
   */
  public static String withDialog(
    final Tokenizer tokenizer,
    final String rolePrompt,
    final List<ChatMessage> priorDialog,
    final String modelUserText
  ) {
    requireNonNull(tokenizer, "tokenizer");
    requireNonNull(priorDialog, "priorDialog");
    List<ChatMessage> turn = dialogTurn(rolePrompt, priorDialog, modelUserText);
    return tokenizer.applyChatTemplate(ChatMessages.toTemplateMaps(turn), true, false);
  }

  static List<ChatMessage> dialogTurn(
    final String rolePrompt,
    final List<ChatMessage> priorDialog,
    final String modelUserText
  ) {
    requireNonNull(priorDialog, "priorDialog");
    requireNonNull(modelUserText, "modelUserText");
    String system = requireNonNull(rolePrompt, "rolePrompt").strip();
    if (system.isEmpty()) {
      throw new IllegalArgumentException("rolePrompt must not be blank");
    }
    String user = advisorFacingUserText(modelUserText);
    if (user.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }

    List<ChatMessage> turn = new ArrayList<>(ChatMessages.newConversation(system));
    priorDialog.stream()
      .filter(message -> message.role() == ChatRole.USER)
      .forEach(turn::add);
    turn.add(ChatMessage.user(user));
    return turn;
  }

  /**
   * User text advisors generate on: question + Context when present (no main-answer footer).
   */
  static String advisorFacingUserText(final String modelUserText) {
    requireNonNull(modelUserText, "modelUserText");
    String base = modelUserText.strip();
    if (base.isEmpty()) {
      return "";
    }
    if (hasContextSection(base)) {
      String question = extractUserQuestion(base);
      String context = extractContextBlock(base);
      if (!question.isEmpty() && !context.isEmpty()) {
        return AdvisorPrompts.advisorUser(question, context);
      }
    }
    return extractUserQuestion(base);
  }

  static String extractUserQuestion(final String modelUserText) {
    if (modelUserText == null || modelUserText.isBlank()) {
      return "";
    }
    return RagPrompts.question(modelUserText);
  }

  /**
   * Notes usable for salvage / diagnostics: drop empty/boilerplate; when facts are present require
   * lexical grounding; dedupe identical notes (first advisor slot wins).
   */
  public static List<String> selectNotesForMix(
    final String modelUserText,
    final List<String> notes
  ) {
    return selectIndexedNotesForMix(modelUserText, notes).stream()
      .map(IndexedNote::note)
      .toList();
  }

  private static List<IndexedNote> selectIndexedNotesForMix(
    final String modelUserText,
    final List<String> notes
  ) {
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(notes, "notes");

    boolean withContext = hasContextSection(modelUserText);
    Set<String> contextTerms = withContext
      ? contentfulTerms(extractContextBlock(modelUserText))
      : Set.of();

    LinkedHashSet<String> seen = new LinkedHashSet<>();
    List<IndexedNote> selected = new ArrayList<>();
    for (int i = 0; i < notes.size(); i++) {
      String note = usableNote(notes.get(i));
      if (note.isEmpty()) {
        continue;
      }
      if (withContext && !isGroundedInContext(note, contextTerms)) {
        continue;
      }
      if (!seen.add(note)) {
        continue;
      }
      selected.add(new IndexedNote(i, note));
    }
    return selected;
  }

  /**
   * Returns the stripped note, or empty when blank.
   *
   * @return stripped note, or empty when blank
   */
  static String usableNote(final String note) {
    return usableNote(note, body -> body != null && !body.isBlank());
  }

  /**
   * Returns the stripped note when {@code keep} accepts it; otherwise empty.
   *
   * @return stripped note when {@code keep} accepts it; otherwise empty
   * @since 1.1.0
   */
  static String usableNote(final String note, final Predicate<String> keep) {
    requireNonNull(keep, "keep");
    String body = note == null ? "" : note.strip();
    if (body.isEmpty() || !keep.test(body)) {
      return "";
    }
    return body;
  }

  /**
   * Mixes useful advisor notes into the main user text via {@link AdvisorPrompts#withGeneratedNotes}.
   * Empty-only lists leave {@code modelUserText} unchanged.
   */
  public static String mix(final String modelUserText, final List<String> answers) {
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(answers, "answers");
    String base = modelUserText.strip();
    if (base.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }
    if (answers.isEmpty()) {
      return base;
    }

    List<String> notes = answers.stream()
      .map(AdvisorPrompt::usableNote)
      .filter(note -> !note.isEmpty())
      .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new))
      .stream()
      .toList();
    if (notes.isEmpty()) {
      return base;
    }
    return AdvisorPrompts.withGeneratedNotes(base, notes);
  }

  static boolean ragTurnWithoutHits(final String modelUserText) {
    if (modelUserText == null || modelUserText.isBlank()) {
      return false;
    }
    return !hasContextSection(modelUserText);
  }

  static boolean hasContextSection(final String modelUserText) {
    return RagPrompts.hasFacts(modelUserText);
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

  static String extractContextBlock(final String modelUserText) {
    return RagPrompts.facts(modelUserText);
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

  private record IndexedNote(int index, String note) {
  }
}
