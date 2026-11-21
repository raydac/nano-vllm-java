package com.igormaznitsa.nanollvm.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * What {@code session.filter} does on the next phrase. Each filter returns a {@link ChatAdmission}.
 * A message stays in the transcript only when every filter keeps it, and it is sent to the model
 * only when every filter sends it. Messages already stored are checked again before the new phrase
 * is added.
 *
 * <pre>{@code
 * ChatSession session = llm.chat().filter(
 *     message -> message.content().contains("secret")
 *         ? ChatAdmission.neither()
 *         : ChatAdmission.both(),
 *     message -> message.role() == ChatRole.SYSTEM
 *         ? ChatAdmission.historyOnly()
 *         : ChatAdmission.both(),
 *     message -> message.content().startsWith("note:")
 *         ? ChatAdmission.historyOnly()
 *         : ChatAdmission.both());
 * session.send("What is 2+2?");
 * }</pre>
 */
class ChatMessageFilterTest {

  @Test
  void nextPhraseKeepsTheTranscriptAndChoosesWhatTheModelSees() {
    ChatTranscript session = new ChatTranscript();
    session.addUnfiltered(List.of(
      ChatMessage.system("Be brief."),
      ChatMessage.user("What is 2+2?"),
      ChatMessage.assistant("4"),
      ChatMessage.user("The secret plan"),
      ChatMessage.assistant("I will keep the secret.")));
    session.addFilter(message -> message.content().contains("secret")
      ? ChatAdmission.neither()
      : ChatAdmission.both());
    session.addFilter(message -> message.role() == ChatRole.SYSTEM
      ? ChatAdmission.historyOnly()
      : ChatAdmission.both());
    session.addFilter(message -> message.content().startsWith("note:")
      ? ChatAdmission.historyOnly()
      : ChatAdmission.both());

    assertTrue(session.reapply());
    ChatTranscript.Entry current = session.add(ChatMessage.user("note: show this only in history"));
    session.add(ChatMessage.assistant("Noted."));

    assertEquals(
      List.of(
        ChatMessage.system("Be brief."),
        ChatMessage.user("What is 2+2?"),
        ChatMessage.assistant("4"),
        ChatMessage.user("note: show this only in history"),
        ChatMessage.assistant("Noted.")),
      session.history());
    assertEquals(
      List.of(
        ChatMessage.user("What is 2+2?"),
        ChatMessage.assistant("4"),
        ChatMessage.assistant("Noted.")),
      session.modelMessages(current, "note: show this only in history"));
    assertNull(session.add(ChatMessage.user("another secret")));
  }

  @Test
  void eachFilterMustAllowTheDestination() {
    ChatTranscript transcript = new ChatTranscript();
    transcript.addFilter(message -> message.content().contains("secret")
      ? ChatAdmission.neither()
      : ChatAdmission.both());
    transcript.addFilter(message -> message.role() == ChatRole.SYSTEM
      ? ChatAdmission.historyOnly()
      : ChatAdmission.both());

    transcript.add(ChatMessage.system("Be brief."));
    transcript.add(ChatMessage.user("What is 2+2?"));
    assertNull(transcript.add(ChatMessage.user("The secret plan")));
    assertNull(transcript.add(ChatMessage.assistant("secret answer")));

    assertEquals(
      List.of(ChatMessage.system("Be brief."), ChatMessage.user("What is 2+2?")),
      transcript.history());
    assertEquals(
      List.of(ChatMessage.user("What is 2+2?")),
      transcript.modelMessages(null, "What is 2+2?"));
  }

  @Test
  void modelOnlyStaysOutOfHistoryAndInThePrompt() {
    ChatTranscript transcript = new ChatTranscript();
    transcript.addFilter(message -> message.content().startsWith("Remember")
      ? ChatAdmission.modelOnly()
      : ChatAdmission.both());
    ChatTranscript.Entry current = transcript.add(ChatMessage.user("Remember this privately."));

    assertTrue(transcript.history().isEmpty());
    assertEquals(
      List.of(ChatMessage.user("Remember this privately.")),
      transcript.modelMessages(current, "Remember this privately."));
  }

  @Test
  void preparedUserTextReplacesOnlyTheCurrentModelMessage() {
    ChatTranscript transcript = new ChatTranscript();
    transcript.add(ChatMessage.user("capital of France"));
    ChatTranscript.Entry current = transcript.add(ChatMessage.user("capital of France"));

    assertEquals(
      List.of(
        ChatMessage.user("capital of France"),
        ChatMessage.user("Paris is the capital of France.\n\ncapital of France")),
      transcript.modelMessages(current, "Paris is the capital of France.\n\ncapital of France"));
    assertEquals(
      List.of(ChatMessage.user("capital of France"), ChatMessage.user("capital of France")),
      transcript.history());
  }

  @Test
  void storedMessagesAreCheckedAgainOnTheNextPass() {
    ChatTranscript transcript = new ChatTranscript();
    transcript.addUnfiltered(List.of(
      ChatMessage.system("Be brief."),
      ChatMessage.user("The secret plan")));
    transcript.addFilter(message -> message.content().contains("secret")
      ? ChatAdmission.neither()
      : ChatAdmission.both());

    assertEquals(2, transcript.history().size());
    assertTrue(transcript.reapply());
    assertEquals(List.of(ChatMessage.system("Be brief.")), transcript.history());
    assertEquals(
      List.of(ChatMessage.system("Be brief.")),
      transcript.modelMessages(null, "next"));
  }

  @Test
  void historyOnlyIsKeptAndOmittedFromTheModel() {
    ChatTranscript transcript = new ChatTranscript();
    transcript.addFilter(message -> ChatAdmission.historyOnly());
    transcript.add(ChatMessage.user("notes for me"));

    assertEquals(List.of(ChatMessage.user("notes for me")), transcript.history());
    assertTrue(transcript.modelMessages(null, "notes for me").isEmpty());
  }

  @Test
  void nullAdmissionIsRejected() {
    ChatTranscript transcript = new ChatTranscript();
    transcript.addFilter(message -> null);
    assertThrows(NullPointerException.class, () -> transcript.add(ChatMessage.user("hello")));
  }

  @Test
  void admissionsIntersect() {
    assertEquals(ChatAdmission.neither(),
      ChatAdmission.historyOnly().and(ChatAdmission.modelOnly()));
    assertEquals(ChatAdmission.both(), ChatAdmission.both().and(ChatAdmission.both()));
    assertThrows(NullPointerException.class, () -> ChatAdmission.both().and(null));
  }
}
