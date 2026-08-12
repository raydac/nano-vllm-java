package com.igormaznitsa.nanollvm.samples.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SampleChatPromptsTest {

  @Test
  void qwenThinkingSystemMentionsThinkTags() {
    assertTrue(SampleChatPrompts.QWEN_THINKING_SYSTEM.contains("<think>"));
    assertTrue(SampleChatPrompts.QWEN_THINKING_SYSTEM.contains("You are the Assistant"));
    assertFalse(SampleChatPrompts.PLAIN_ASSISTANT_SYSTEM.contains("<think>"));
  }

  @Test
  void forDemoFallsBackToEmptyWithoutTokenizer() {
    assertEquals("", SampleChatPrompts.forDemo("llama", null));
    assertEquals("", SampleChatPrompts.forDemo("lfm2", null));
  }

  @Test
  void setupBoilerplateDetectsShortAcks() {
    assertTrue(SampleChatPrompts.isSetupBoilerplate("Okay, I'm ready."));
    assertTrue(SampleChatPrompts.isSetupBoilerplate("Okay, I understand. Let's begin."));
    assertFalse(SampleChatPrompts.isSetupBoilerplate("Hello! How can I help you today?"));
    assertFalse(SampleChatPrompts.isSetupBoilerplate("The president of Estonia is Alar Karis."));
  }

  @Test
  void samplingForDemoUsesTurnBasedTopKOnlyWhenFlagged() {
    assertEquals(0, SampleChatPrompts.samplingForDemo(null, 32).topK());
  }
}
