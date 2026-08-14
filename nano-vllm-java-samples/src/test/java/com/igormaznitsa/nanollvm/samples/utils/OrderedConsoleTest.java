package com.igormaznitsa.nanollvm.samples.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class OrderedConsoleTest {

  @Test
  void answerStaysOnOutWhileThinkingGoesToInfo() {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream err = new ByteArrayOutputStream();
    OrderedConsole console = new OrderedConsole(
      new PrintStream(out, true, StandardCharsets.UTF_8),
      new PrintStream(err, true, StandardCharsets.UTF_8));

    PrintStream think = console.infoStream();
    PrintStream answer = console.stream();
    think.print("thinking> note\n");
    think.flush();
    think.print("debug> prepared\n");
    think.flush();
    answer.print("assistant> hello");
    answer.flush();
    console.println();
    console.printf(java.util.Locale.ROOT, "(turn %d)%n", 1);

    String outText = out.toString(StandardCharsets.UTF_8);
    String errText = err.toString(StandardCharsets.UTF_8);
    assertTrue(errText.contains("thinking> note"), errText);
    assertTrue(errText.contains("debug> prepared"), errText);
    assertTrue(outText.contains("assistant> hello"), outText);
    assertTrue(outText.contains("(turn 1)"), outText);
    assertTrue(!outText.contains("thinking>"), outText);
    assertTrue(!outText.contains("debug>"), outText);
  }

  @Test
  void sameMillisecondChunksKeepSequenceOrder() {
    ByteArrayOutputStream sink = new ByteArrayOutputStream();
    OrderedConsole console =
      new OrderedConsole(new PrintStream(sink, true, StandardCharsets.UTF_8));
    console.print("A");
    console.print("B");
    console.print("C");
    assertEquals("ABC", sink.toString(StandardCharsets.UTF_8));
  }

  @Test
  void infoLinesGoToInfoSinkNotChatOut() {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream err = new ByteArrayOutputStream();
    OrderedConsole console = new OrderedConsole(
      new PrintStream(out, true, StandardCharsets.UTF_8),
      new PrintStream(err, true, StandardCharsets.UTF_8));

    console.printlnInfo("Preparing RAG corpus");
    console.println("rag?> ");

    assertTrue(err.toString(StandardCharsets.UTF_8).contains("Preparing RAG corpus"),
      err.toString(StandardCharsets.UTF_8));
    assertTrue(out.toString(StandardCharsets.UTF_8).contains("rag?>"),
      out.toString(StandardCharsets.UTF_8));
    assertTrue(!out.toString(StandardCharsets.UTF_8).contains("Preparing RAG"),
      out.toString(StandardCharsets.UTF_8));
  }
}
