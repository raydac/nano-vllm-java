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
      RagSession.QueryRewrite.parse("Philipp Wilhelm Grimm father"));
  }

  @Test
  void parseStripsSearchPrefixAndQuotes() {
    assertEquals(Optional.of("Jacob Wilhelm Grimm names"),
      RagSession.QueryRewrite.parse("SEARCH: \"Jacob Wilhelm Grimm names\""));
  }

  @Test
  void parseTreatsNoneAsEmpty() {
    assertTrue(RagSession.QueryRewrite.parse("NONE").isEmpty());
    assertTrue(RagSession.QueryRewrite.parse("none").isEmpty());
    assertTrue(RagSession.QueryRewrite.parse("  None  ").isEmpty());
  }

  @Test
  void parseUsesFirstNonBlankLine() {
    assertEquals(Optional.of("Grimm brothers father died"),
      RagSession.QueryRewrite.parse("\n\nGrimm brothers father died\nextra narrative"));
  }

  @Test
  void parseIgnoresThinkBlocks() {
    assertEquals(Optional.of("Hanau residence city"),
      RagSession.QueryRewrite.parse("<think>ponder</think>\nHanau residence city"));
    assertEquals(Optional.of("Hanau residence city"),
      RagSession.QueryRewrite.parse(
        "[reasoning]ponder[/reasoning]\nHanau residence city",
        com.igormaznitsa.nanollvm.chat.ThinkTags.of("[reasoning]", "[/reasoning]")));
  }

  @Test
  void userMessageIncludesPriorWhenPresent() {
    String msg =
      RagSession.QueryRewrite.userMessage("names of grimm brothers", "name of their father");
    assertTrue(msg.contains("Prior: names of grimm brothers"));
    assertTrue(msg.contains("Follow-up: name of their father"));
  }

  @Test
  void userMessageOmitsPriorWhenBlank() {
    String msg = RagSession.QueryRewrite.userMessage("  ", "city where they lived");
    assertFalse(msg.contains("Prior:"));
    assertTrue(msg.contains("Question: city where they lived"));
  }
}
