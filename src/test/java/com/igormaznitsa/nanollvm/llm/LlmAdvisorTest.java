package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.chat.ChatHistory;
import com.igormaznitsa.nanollvm.chat.ChatMessage;
import java.util.List;
import org.junit.jupiter.api.Test;

class LlmAdvisorTest {

  @Test
  void builderTrimsNameAndPrompt() {
    LlmAdvisor advisor = LlmAdvisor.builder()
      .name("  Facts  ")
      .prompt("  Check claims.  ")
      .build();
    assertEquals("Facts", advisor.name());
    assertEquals("Check claims.", advisor.prompt());
  }

  @Test
  void builderRejectsBlankNameOrPrompt() {
    assertThrows(IllegalArgumentException.class,
      () -> LlmAdvisor.builder().name("  ").prompt("ok").build());
    assertThrows(IllegalArgumentException.class,
      () -> LlmAdvisor.builder().name("ok").prompt("  ").build());
    assertThrows(NullPointerException.class,
      () -> LlmAdvisor.builder().prompt("ok").build());
  }

  @Test
  void defaultMixerInsertsNotes() {
    String mixed = LlmAdvisorMixer.defaults().mixPrompt(
      null,
      List.of(new AdvisorResponse("Facts", "Paris is the capital.")),
      ChatHistory.of(List.of(ChatMessage.user("prior"))),
      "What is the capital of France?");
    assertTrue(mixed.contains("Paris is the capital."), mixed);
    assertTrue(mixed.contains("What is the capital of France?"), mixed);
  }
}
