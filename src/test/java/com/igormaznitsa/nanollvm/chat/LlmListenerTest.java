package com.igormaznitsa.nanollvm.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class LlmListenerTest {

  @Test
  void textEventFactories() {
    LlmTextEvent delta = LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "hi");
    assertEquals(LlmTextKind.TEXT_ASSISTANT, delta.kind());
    assertEquals("hi", delta.text());
    assertEquals(-1, delta.slot());
    assertTrue(!delta.snapshot());

    LlmTextEvent note = LlmTextEvent.advisorNote(2, "hint");
    assertEquals(LlmTextKind.TEXT_ADVISOR_NOTE, note.kind());
    assertEquals(2, note.slot());
    assertEquals("hint", note.text());
  }

  @Test
  void printStreamListenerFormatsAdvisorAndAnswer() {
    ByteArrayOutputStream think = new ByteArrayOutputStream();
    ByteArrayOutputStream answer = new ByteArrayOutputStream();
    LlmListener listener = LlmListeners.toPrintStreams(
      new PrintStream(think, true, StandardCharsets.UTF_8),
      new PrintStream(answer, true, StandardCharsets.UTF_8),
      false);

    listener.onText(null, LlmTextEvent.advisorNote(1, "check billing"));
    listener.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "Payment "));
    listener.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "failed."));
    ((LlmListeners.PrintStreamLlmListener) listener).closeTurn();

    String thinkText = think.toString(StandardCharsets.UTF_8);
    String answerText = answer.toString(StandardCharsets.UTF_8);
    assertTrue(thinkText.contains("[advisor 1] check billing"), thinkText);
    assertTrue(answerText.contains("Payment failed."), answerText);
  }

  @Test
  void composeInvokesBothListeners() {
    List<LlmTextKind> seen = new ArrayList<>();
    LlmListener combined = LlmListeners.compose(
      (source, event) -> seen.add(event.kind()),
      (source, event) -> seen.add(event.kind()));

    combined.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_DIAGNOSTICS, "x"));
    assertEquals(List.of(LlmTextKind.TEXT_DIAGNOSTICS, LlmTextKind.TEXT_DIAGNOSTICS), seen);
  }
}
