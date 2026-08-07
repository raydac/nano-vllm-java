package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class SubagentPromptTest {

  @Test
  void mixAppendsNumberedAdvisorNotes() {
    String mixed = SubagentPrompt.mix("Question: capital of France?", List.of(
      "Paris is the capital.",
      "  Confirm from geography.  "
    ));
    assertTrue(mixed.startsWith("Question: capital of France?"));
    assertTrue(mixed.contains("Advisor notes"));
    assertTrue(mixed.contains("[1] Paris is the capital."));
    assertTrue(mixed.contains("[2] Confirm from geography."));
  }

  @Test
  void mixCompactEndsWithFinalAnswerCue() {
    String mixed = SubagentPrompt.mix("What about Estonia?", List.of("Check maps."), true);
    assertTrue(mixed.contains("Notes:"));
    assertTrue(mixed.contains("[1] Check maps."));
    assertTrue(mixed.contains("Final answer (one short sentence):"));
  }

  @Test
  void mixSkipsBlankAnswers() {
    assertEquals(
      "Stay concise.",
      SubagentPrompt.mix("Stay concise.", List.of("", "  ")));
  }

  @Test
  void mixRejectsBlankBase() {
    assertThrows(IllegalArgumentException.class,
      () -> SubagentPrompt.mix("  ", List.of("note")));
  }
}
