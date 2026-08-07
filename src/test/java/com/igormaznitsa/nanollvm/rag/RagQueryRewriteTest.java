package com.igormaznitsa.nanollvm.rag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

class RagQueryRewriteTest {

  @Test
  void parseAcceptsKeywordLine() {
    assertEquals(Optional.of("Philipp Wilhelm Grimm father"),
      RagQueryRewrite.parse("Philipp Wilhelm Grimm father"));
  }

  @Test
  void parseStripsSearchPrefixAndQuotes() {
    assertEquals(Optional.of("Jacob Wilhelm Grimm names"),
      RagQueryRewrite.parse("SEARCH: \"Jacob Wilhelm Grimm names\""));
  }

  @Test
  void parseTreatsNoneAsEmpty() {
    assertTrue(RagQueryRewrite.parse("NONE").isEmpty());
    assertTrue(RagQueryRewrite.parse("none").isEmpty());
    assertTrue(RagQueryRewrite.parse("  None  ").isEmpty());
  }

  @Test
  void parseUsesFirstNonBlankLine() {
    assertEquals(Optional.of("Grimm brothers father died"),
      RagQueryRewrite.parse("\n\nGrimm brothers father died\nextra narrative"));
  }

  @Test
  void parseIgnoresThinkBlocks() {
    assertEquals(Optional.of("Hanau residence city"),
      RagQueryRewrite.parse("<think>ponder</think>\nHanau residence city"));
  }

  @Test
  void userMessageIncludesPriorWhenPresent() {
    String msg = RagQueryRewrite.userMessage("names of grimm brothers", "name of their father");
    assertTrue(msg.contains("Prior: names of grimm brothers"));
    assertTrue(msg.contains("Follow-up: name of their father"));
  }

  @Test
  void userMessageOmitsPriorWhenBlank() {
    String msg = RagQueryRewrite.userMessage("  ", "city where they lived");
    assertFalse(msg.contains("Prior:"));
    assertTrue(msg.contains("Question: city where they lived"));
  }
}
