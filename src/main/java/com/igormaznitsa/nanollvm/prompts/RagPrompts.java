package com.igormaznitsa.nanollvm.prompts;

/**
 * Model-facing RAG prompt text and section markers shared with advisor grounding.
 */
public final class RagPrompts {

  public static final String CONTEXT_HEADING = "Context:";
  public static final String QUESTION_HEADING = "Question:";
  public static final String NO_CONTEXT_DOCUMENTS = "No context documents";
  public static final String ABSTAIN_REPLY = "I do not know";

  public static final String COMPACT_ANSWER_INSTRUCTION =
    "Answer in one short sentence using names and facts from the Context above.";

  public static final String FULL_ANSWER_INSTRUCTION = """
    Answer using only the context. Do not invent names, dates, or other details.
    If the context does not contain the answer, say you do not know. Be concise.
    """.strip();

  public static final String COMPACT_NO_HIT_INSTRUCTION = """
    No context documents were found for this question.
    Reply with exactly: %s.
    Do not invent places, names, or stories.
    """.formatted(ABSTAIN_REPLY).strip();

  public static final String FULL_NO_HIT_INSTRUCTION = """
    No context documents were retrieved. Say you do not know. Do not invent facts.
    """.strip();

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

  public static String compactHit(final String question, final String context) {
    return question + "\n\n\n" + CONTEXT_HEADING + "\n" + context + "\n\n"
      + COMPACT_ANSWER_INSTRUCTION + "\n";
  }

  public static String fullHit(final String question, final String context) {
    return """
      %s
      %s

      %s %s

      %s
      """.formatted(
      CONTEXT_HEADING,
      context,
      QUESTION_HEADING,
      question,
      FULL_ANSWER_INSTRUCTION).strip();
  }

  public static String compactNoHit(final String question) {
    return question + "\n\n\n" + COMPACT_NO_HIT_INSTRUCTION + "\n";
  }

  public static String fullNoHit(final String question) {
    return """
      %s %s

      %s
      """.formatted(QUESTION_HEADING, question, FULL_NO_HIT_INSTRUCTION).strip();
  }

  public static String rewriteStandalone(final String question) {
    return REWRITE_STANDALONE.formatted(question).strip();
  }

  public static String rewriteFollowUp(final String prior, final String followUp) {
    return REWRITE_FOLLOW_UP.formatted(prior, followUp).strip();
  }
}
