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
    assertEquals("", delta.advisorName());
    assertTrue(!delta.snapshot());

    LlmTextEvent note = LlmTextEvent.advisorNote("Risks", "hint");
    assertEquals(LlmTextKind.TEXT_ADVISOR_NOTE, note.kind());
    assertEquals("Risks", note.advisorName());
    assertEquals("hint", note.text());
  }

  @Test
  void rawDecodeEventKeepsMarkup() {
    LlmTextEvent raw = LlmTextEvent.of(
      LlmTextKind.TEXT_RAW, "<think>plan</think>\nHi<|im_end|>");
    assertEquals(LlmTextKind.TEXT_RAW, raw.kind());
    assertEquals("<think>plan</think>\nHi<|im_end|>", raw.text());
    assertTrue(!raw.snapshot());
  }

  @Test
  void printStreamListenerFormatsAdvisorAndAnswer() {
    ByteArrayOutputStream think = new ByteArrayOutputStream();
    ByteArrayOutputStream answer = new ByteArrayOutputStream();
    LlmListener listener = LlmListeners.toPrintStreams(
      new PrintStream(think, true, StandardCharsets.UTF_8),
      new PrintStream(answer, true, StandardCharsets.UTF_8),
      false);

    listener.onText(null, LlmTextEvent.advisorNote("Billing", "check billing"));
    listener.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "Payment "));
    listener.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "failed."));
    listener.onText(null,
      LlmTextEvent.of(LlmTextKind.TEXT_RAW, "<think>x</think>Payment failed.<|im_end|>"));
    ((LlmListeners.PrintStreamLlmListener) listener).closeTurn();

    String thinkText = think.toString(StandardCharsets.UTF_8);
    String answerText = answer.toString(StandardCharsets.UTF_8);
    assertTrue(thinkText.contains("[Billing] check billing"), thinkText);
    assertTrue(answerText.contains("Payment failed."), answerText);
    assertTrue(!thinkText.contains("<|im_end|>"), thinkText);
    assertTrue(!answerText.contains("<think>"), answerText);
  }

  @Test
  void printStreamListenerFormatsPreparedUserDebug() {
    ByteArrayOutputStream think = new ByteArrayOutputStream();
    LlmListener listener = LlmListeners.toPrintStreams(
      new PrintStream(think, true, StandardCharsets.UTF_8),
      new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
      false);

    listener.onText(null, LlmTextEvent.debug("Advisor hints\n\nQuestion: hello"));

    String thinkText = think.toString(StandardCharsets.UTF_8);
    assertTrue(thinkText.contains("debug> --- prepared model user ---"), thinkText);
    assertTrue(thinkText.contains("debug> Advisor hints"), thinkText);
    assertTrue(thinkText.contains("debug> Question: hello"), thinkText);
    assertTrue(thinkText.contains("debug> ---"), thinkText);
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

  @Test
  void unwrapPrintStreamSurvivesDiagnosticsCompose() {
    ByteArrayOutputStream answer = new ByteArrayOutputStream();
    LlmListener printed = LlmListeners.toPrintStreams(
      new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
      new PrintStream(answer, true, StandardCharsets.UTF_8),
      false);
    LlmListener withDiagnostics = LlmListeners.compose(printed, (source, event) -> {
    });

    var sink = LlmListeners.unwrapPrintStream(withDiagnostics);
    assertTrue(sink != null);

    sink.resetTurn();
    withDiagnostics.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "Hi"));
    sink.closeTurn();
    sink.resetTurn();
    withDiagnostics.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "Again"));
    sink.closeTurn();

    String text = answer.toString(StandardCharsets.UTF_8);
    assertTrue(text.contains("assistant> Hi"), text);
    assertTrue(text.contains("assistant> Again"), text);
  }

  @Test
  void discardAnswerAllowsFreshAssistantPrefix() {
    ByteArrayOutputStream answer = new ByteArrayOutputStream();
    var sink = (LlmListeners.PrintStreamLlmListener) LlmListeners.toPrintStreams(
      new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
      new PrintStream(answer, true, StandardCharsets.UTF_8),
      false);

    sink.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "Okay, I understand."));
    sink.discardAnswer();
    sink.onText(null, LlmTextEvent.of(LlmTextKind.TEXT_ASSISTANT, "The universe is vast."));
    sink.closeTurn();

    String text = answer.toString(StandardCharsets.UTF_8);
    assertTrue(text.contains("assistant> The universe is vast."), text);
  }
}
