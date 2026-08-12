package com.igormaznitsa.nanollvm.samples.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SampleAdvisorPromptsTest {

  @Test
  void rolesIncludeReplyInstructionAndAddonIsOptional() {
    assertTrue(SampleAdvisorPrompts.ROLE_PRACTICAL.contains("Practical"));
    assertTrue(SampleAdvisorPrompts.ROLE_PRACTICAL.contains(SampleAdvisorPrompts.FOR_ADVISOR));
    assertTrue(SampleAdvisorPrompts.ROLE_ABSTRACT.contains("Abstract"));
    assertTrue(SampleAdvisorPrompts.ROLE_CONSEQUENCE.contains("Consequence"));
    assertEquals("", SampleAdvisorPrompts.withAdvisorAddon(""));
    assertTrue(SampleAdvisorPrompts.withAdvisorAddon("Be brief.")
      .contains(SampleAdvisorPrompts.ADVISOR_AWARE_ADDON));
  }
}
