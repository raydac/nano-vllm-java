package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import com.igormaznitsa.nanollvm.prompts.SubagentPrompts;
import java.util.List;
import org.junit.jupiter.api.Test;

class SubagentPromptTest {

  @Test
  void mixUsesClaimLabelsNotRankedIndexes() {
    String mixed = SubagentPrompt.mix("Question: capital of France?", List.of(
        "Paris is the capital.",
        "  Confirm from geography.  "
    ));
    assertTrue(mixed.contains(SubagentPrompts.MIX_CLAIMS_HEADER));
    assertTrue(mixed.contains(SubagentPrompts.claimLine("A", "Paris is the capital.")));
    assertTrue(mixed.contains(SubagentPrompts.claimLine("B", "Confirm from geography.")));
    assertFalse(mixed.contains("[1]"));
  }

  @Test
  void mixKeepsOriginalClaimSlotsWhenOneEmpty() {
    String mixed = SubagentPrompt.mix(
        "Question?",
        List.of("Practical note.", "", "Future note."));
    assertTrue(mixed.contains(SubagentPrompts.claimLine("A", "Practical note.")));
    assertTrue(mixed.contains(SubagentPrompts.claimLine("C", "Future note.")));
    assertFalse(mixed.contains("claim-B"));
  }

  @Test
  void mixCompactPrependsClaimsSoContextStaysLast() {
    String base = """
        what was the destination?

        %s
        - Take them to your grandmother.

        %s
        """.formatted(RagPrompts.CONTEXT_HEADING, RagPrompts.COMPACT_ANSWER_INSTRUCTION).strip();
    String mixed = SubagentPrompt.mix(base, List.of("It was a garden."), true);
    assertTrue(mixed.startsWith(SubagentPrompts.MIX_CLAIMS_HEADER));
    assertTrue(mixed.contains(SubagentPrompts.claimLine("A", "It was a garden.")));
    assertTrue(mixed.endsWith(RagPrompts.COMPACT_ANSWER_INSTRUCTION));
    int claimsAt = mixed.indexOf("claim-A");
    int contextAt = mixed.indexOf(RagPrompts.CONTEXT_HEADING);
    assertTrue(claimsAt >= 0 && contextAt > claimsAt);
  }

  @Test
  void groundedRoleAppendsExtractionRules() {
    String role = SubagentPrompt.groundedRole("Practical extractor.");
    assertTrue(role.startsWith("Practical extractor."));
    assertTrue(role.contains(RagPrompts.ABSTAIN_REPLY));
    assertTrue(role.contains("Context"));
  }

  @Test
  void claimLabelUsesLettersThenNumbers() {
    assertEquals("A", SubagentPrompt.claimLabel(0));
    assertEquals("C", SubagentPrompt.claimLabel(2));
    assertEquals("27", SubagentPrompt.claimLabel(26));
  }

  @Test
  void retainContextGroundedDropsInventedPlaces() {
    String ragUser = """
        what was the destination point for the red hood?

        %s
        - Take them to your grandmother. She is sick and weak.
        - The grandmother lived out in the woods.

        %s
        """.formatted(RagPrompts.CONTEXT_HEADING, RagPrompts.COMPACT_ANSWER_INSTRUCTION).strip();
    List<String> kept = SubagentPrompt.retainContextGrounded(ragUser, List.of(
        "The destination was a secluded hill overlooking the valley.",
        RagPrompts.ABSTAIN_REPLY + ".",
        "She was going to her grandmother in the woods."));
    assertEquals("", kept.get(0));
    assertEquals("", kept.get(1));
    assertTrue(kept.get(2).toLowerCase().contains("grandmother"));
  }

  @Test
  void retainContextGroundedPassesThroughWithoutContextSection() {
    List<String> kept = SubagentPrompt.retainContextGrounded(
        "What should we do next?",
        List.of("Ship a small fix first."));
    assertEquals("Ship a small fix first.", kept.getFirst());
  }

  @Test
  void mixRejectsBlankBase() {
    assertThrows(IllegalArgumentException.class,
        () -> SubagentPrompt.mix("  ", List.of("note")));
  }
}
