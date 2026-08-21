package com.igormaznitsa.nanollvm.prompts;

/**
 * Model-facing RAG document layout: a short grounding line, facts, blank line, user question last.
 */
public final class RagPrompts {

  public static final String NO_CONTEXT_DOCUMENTS = "No context documents";
  public static final String ABSTAIN_REPLY = "I do not know";
  /**
   * Lead sentence on a grounded RAG user turn: answer from the passages, do not invent a source.
   *
   * @since 1.2.0
   */
  public static final String GROUNDING =
    "Answer using only the passages below. Do not invent books, plays, or sources.";

  public static final String REWRITE_STANDALONE = """
    Rewrite the question as a short keyword search for a document index.
    Reply with only the search keywords, or NONE if nothing can be searched.

    Question: %s
    """.strip();

  public static final String REWRITE_FOLLOW_UP = """
    Rewrite the follow-up as a short keyword search for a document index.
    Use Prior to resolve pronouns and missing names.
    Reply with only the search keywords.
    Reply NONE only when the follow-up is unrelated to Prior.

    Prior: %s
    Follow-up: %s
    """.strip();

  private RagPrompts() {
  }

  /**
   * Grounding line, facts block, then blank line, then the user question.
   */
  public static String withContext(final String question, final String context) {
    return GROUNDING + "\n\n" + context.strip() + "\n\n" + question.strip();
  }

  /**
   * Question only (no retrieved facts).
   */
  public static String withoutContext(final String question) {
    return question.strip();
  }

  public static String facts(final String document) {
    if (document == null || document.isBlank()) {
      return "";
    }
    String text = document.strip();
    int sep = text.lastIndexOf("\n\n");
    if (sep < 0) {
      return "";
    }
    return stripGrounding(text.substring(0, sep).strip());
  }

  public static String question(final String document) {
    if (document == null || document.isBlank()) {
      return "";
    }
    String text = document.strip();
    int sep = text.lastIndexOf("\n\n");
    if (sep < 0) {
      return text;
    }
    return text.substring(sep + 2).strip();
  }

  public static boolean hasFacts(final String document) {
    return !facts(document).isBlank();
  }

  private static String stripGrounding(final String block) {
    if (block.equals(GROUNDING)) {
      return "";
    }
    if (block.startsWith(GROUNDING)) {
      return block.substring(GROUNDING.length()).strip();
    }
    return block;
  }

  public static String rewriteStandalone(final String question) {
    return REWRITE_STANDALONE.formatted(question).strip();
  }

  public static String rewriteFollowUp(final String prior, final String followUp) {
    return REWRITE_FOLLOW_UP.formatted(prior, followUp).strip();
  }
}
