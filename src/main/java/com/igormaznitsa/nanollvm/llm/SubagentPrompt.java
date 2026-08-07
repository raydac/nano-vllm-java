package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import com.igormaznitsa.nanollvm.prompts.SubagentPrompts;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * Builds isolated subagent chat prompts and mixes advisor answers into the main user text.
 * Wording lives in {@link SubagentPrompts} / {@link RagPrompts}.
 */
public final class SubagentPrompt {

  private static final Pattern TOKEN = Pattern.compile("[\\p{L}\\p{N}]+");

  private SubagentPrompt() {
  }

  /**
   * Role prompt plus shared grounded-extraction rules used for every subagent turn.
   */
  public static String groundedRole(final String rolePrompt) {
    return SubagentPrompts.groundedRole(rolePrompt);
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

    List<ChatMessage> turn =
      new ArrayList<>(ChatMessages.newConversation(groundedRole(rolePrompt)));
    turn.add(ChatMessage.user(user));
    return tokenizer.applyChatTemplate(ChatMessages.toTemplateMaps(turn), true, false);
  }

  /**
   * Keeps slot alignment: blanks out abstentions and claims that are not supported by a Context
   * block (after removing question words). When there is no Context section, answers pass through.
   */
  public static List<String> retainContextGrounded(
    final String modelUserText,
    final List<String> answers
  ) {
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(answers, "answers");
    String base = modelUserText.strip();
    if (!hasContextSection(base)) {
      return answers.stream()
        .map(answer -> answer == null ? "" : answer.strip())
        .toList();
    }

    Set<String> contextTerms = contentfulTerms(extractContextBlock(base));
    Set<String> questionTerms = contentfulTerms(extractQuestion(base));
    if (contextTerms.isEmpty()) {
      return answers.stream().map(answer -> "").toList();
    }

    return answers.stream()
      .map(answer -> {
        String note = answer == null ? "" : answer.strip();
        if (note.isEmpty() || isAbstention(note)) {
          return "";
        }
        return isGroundedInContext(note, contextTerms, questionTerms) ? note : "";
      })
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
        return note.isEmpty() ? "" : SubagentPrompts.claimLine(claimLabel(i), note);
      })
      .filter(line -> !line.isEmpty())
      .toList();
    if (claimLines.isEmpty()) {
      return base;
    }

    String claims = String.join("\n", claimLines);
    return compact
      ? SubagentPrompts.mixCompact(claims, base)
      : SubagentPrompts.mixFull(base, claims);
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

  static boolean isAbstention(final String note) {
    return note != null && SubagentPrompts.ABSTAIN_REPLY.matcher(note.strip()).matches();
  }

  static boolean hasContextSection(final String modelUserText) {
    String lower = modelUserText.toLowerCase(Locale.ROOT);
    return lower.contains(RagPrompts.CONTEXT_HEADING.toLowerCase(Locale.ROOT))
      || lower.contains(RagPrompts.NO_CONTEXT_DOCUMENTS.toLowerCase(Locale.ROOT));
  }

  static String extractContextBlock(final String modelUserText) {
    String lower = modelUserText.toLowerCase(Locale.ROOT);
    String noCtx = RagPrompts.NO_CONTEXT_DOCUMENTS.toLowerCase(Locale.ROOT);
    if (lower.contains(noCtx)) {
      return "";
    }
    String ctxHeading = RagPrompts.CONTEXT_HEADING.toLowerCase(Locale.ROOT);
    int ctx = lower.indexOf(ctxHeading);
    if (ctx < 0) {
      return "";
    }
    int start = ctx + ctxHeading.length();
    int end = modelUserText.length();
    for (String marker : SubagentPrompts.CONTEXT_BLOCK_END_MARKERS) {
      int at = modelUserText.indexOf(marker, start);
      if (at >= 0 && at < end) {
        end = at;
      }
    }
    return modelUserText.substring(start, end).strip();
  }

  static String extractQuestion(final String modelUserText) {
    String lower = modelUserText.toLowerCase(Locale.ROOT);
    String ctxHeading = RagPrompts.CONTEXT_HEADING.toLowerCase(Locale.ROOT);
    int ctx = lower.indexOf("\n\n" + ctxHeading);
    if (ctx < 0) {
      ctx = lower.indexOf("\n" + ctxHeading);
    }
    if (ctx < 0) {
      String qHeading = RagPrompts.QUESTION_HEADING.toLowerCase(Locale.ROOT);
      int q = lower.indexOf(qHeading);
      if (q >= 0) {
        int start = q + qHeading.length();
        int end = modelUserText.length();
        int ctx2 = lower.indexOf(ctxHeading);
        if (ctx2 > start) {
          end = ctx2;
        }
        return modelUserText.substring(start, end).strip();
      }
      return modelUserText.strip();
    }
    return modelUserText.substring(0, ctx).strip();
  }

  static boolean isGroundedInContext(
    final String claim,
    final Set<String> contextTerms,
    final Set<String> questionTerms
  ) {
    Set<String> claimTerms = contentfulTerms(claim);
    claimTerms.removeAll(questionTerms);
    if (claimTerms.isEmpty()) {
      return false;
    }
    long overlap = claimTerms.stream().filter(contextTerms::contains).count();
    int need = Math.max(1, (claimTerms.size() + 2) / 3);
    return overlap >= need;
  }

  static Set<String> contentfulTerms(final String text) {
    if (text == null || text.isBlank()) {
      return Set.of();
    }
    Set<String> terms = new LinkedHashSet<>();
    Matcher matcher = TOKEN.matcher(text.toLowerCase(Locale.ROOT));
    while (matcher.find()) {
      String token = matcher.group();
      if (token.length() >= 4) {
        terms.add(token);
      }
    }
    return terms;
  }
}
