package com.igormaznitsa.nanollvm.prompts;

/**
 * Model-facing RAG document layout: facts first, blank line, user question last.
 */
public final class RagPrompts {

  public static final String NO_CONTEXT_DOCUMENTS = "No context documents";
  public static final String ABSTAIN_REPLY = "I do not know";

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
   * Facts block, then blank line, then the user question.
   */
  public static String withContext(final String question, final String context) {
    return context.strip() + "\n\n" + question.strip();
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
    return text.substring(0, sep).strip();
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

  public static String rewriteStandalone(final String question) {
    return REWRITE_STANDALONE.formatted(question).strip();
  }

  public static String rewriteFollowUp(final String prior, final String followUp) {
    return REWRITE_FOLLOW_UP.formatted(prior, followUp).strip();
  }
}
