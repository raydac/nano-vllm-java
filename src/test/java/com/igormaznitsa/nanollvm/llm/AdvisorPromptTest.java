package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.prompts.AdvisorPrompts;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import java.util.List;
import org.junit.jupiter.api.Test;

class AdvisorPromptTest {

  @Test
  void mixUsesClaimLabelsNotRankedIndexes() {
    String mixed = AdvisorPrompt.mix("Question: capital of France?", List.of(
        "Paris is the capital.",
        "  Confirm from geography.  "
    ));
    assertTrue(mixed.contains(AdvisorPrompts.MIX_CLAIMS_HEADER));
    assertTrue(mixed.contains(AdvisorPrompts.claimLine("A", "Paris is the capital.")));
    assertTrue(mixed.contains(AdvisorPrompts.claimLine("B", "Confirm from geography.")));
    assertFalse(mixed.contains("[1]"));
  }

  @Test
  void mixKeepsOriginalClaimSlotsWhenOneEmpty() {
    String mixed = AdvisorPrompt.mix(
        "Question?",
        List.of("Practical note.", "", "Future note."));
    assertTrue(mixed.contains(AdvisorPrompts.claimLine("A", "Practical note.")));
    assertTrue(mixed.contains(AdvisorPrompts.claimLine("C", "Future note.")));
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
    String mixed = AdvisorPrompt.mix(base, List.of("It was a garden."), true);
    assertTrue(mixed.startsWith(AdvisorPrompts.MIX_CLAIMS_HEADER));
    assertTrue(mixed.contains(AdvisorPrompts.MIX_FULL_FOOTER));
    assertTrue(mixed.contains(AdvisorPrompts.claimLine("A", "It was a garden.")));
    assertTrue(mixed.endsWith(RagPrompts.COMPACT_ANSWER_INSTRUCTION));
    int claimsAt = mixed.indexOf("claim-A");
    int contextAt = mixed.indexOf(RagPrompts.CONTEXT_HEADING);
    assertTrue(claimsAt >= 0 && contextAt > claimsAt);
  }

  @Test
  void groundedRoleAppendsExtractionRules() {
    String role = AdvisorPrompt.groundedRole("Practical extractor.");
    assertTrue(role.startsWith("Practical extractor."));
    assertTrue(role.contains("pre-answer advisor"));
    assertTrue(role.contains("Do not reply with only"));
    assertTrue(role.contains("Context"));
    assertTrue(role.contains("open-ended"));
    assertTrue(role.contains("partial summary"));
  }

  @Test
  void claimLabelUsesLettersThenNumbers() {
    assertEquals("A", AdvisorPrompt.claimLabel(0));
    assertEquals("C", AdvisorPrompt.claimLabel(2));
    assertEquals("27", AdvisorPrompt.claimLabel(26));
  }

  @Test
  void selectNotesDropsAbstentionAndUngroundedWhenContextPresent() {
    String ragUser = """
      what was their main interest?

        %s
      - Jacob and Wilhelm Grimm were German authors of fairy tales and folklore.

        %s
        """.formatted(RagPrompts.CONTEXT_HEADING, RagPrompts.COMPACT_ANSWER_INSTRUCTION).strip();
    List<String> selected = AdvisorPrompt.selectNotesForMix(ragUser, List.of(
      "Jacob Grimm and Wilhelm Grimm were famous Danish storytellers.",
        RagPrompts.ABSTAIN_REPLY + ".",
      "Jacob Grimm and Wilhelm Grimm were famous German authors of fairy tales and folklore."));
    assertEquals(List.of(
        "Jacob Grimm and Wilhelm Grimm were famous German authors of fairy tales and folklore."),
      selected);
  }

  @Test
  void selectNotesKeepsComplementaryGroundedHints() {
    String ragUser = """
      where did they work?

      %s
      - Jacob Grimm worked as a librarian in Kassel. Wilhelm Grimm also worked in Kassel.
      - Later they taught in Göttingen and then lived in Berlin.

      %s
      """.formatted(RagPrompts.CONTEXT_HEADING, RagPrompts.COMPACT_ANSWER_INSTRUCTION).strip();
    List<String> selected = AdvisorPrompt.selectNotesForMix(ragUser, List.of(
      "They worked as librarians in Kassel.",
      "They later taught in Göttingen and lived in Berlin."));
    assertEquals(2, selected.size());
  }

  @Test
  void selectNotesSkipsMixOnRagNoHit() {
    String noHit = RagPrompts.compactNoHit("what is alpha centauri?");
    List<String> selected = AdvisorPrompt.selectNotesForMix(noHit, List.of(
      "The question asks for a star system outside the fairy-tale index."));
    assertTrue(selected.isEmpty());
  }

  @Test
  void groundedRoleForRagNoHitsAddsDocumentIndexAngle() {
    String role = AdvisorPrompts.groundedRole("Practical extractor.", true);
    assertTrue(role.contains("Indexed documents"));
  }

  @Test
  void ragTurnWithoutHitsSkipsContextGrounding() {
    String noHit = RagPrompts.compactNoHit("что такое альфа центавра?");
    assertTrue(AdvisorPrompt.ragTurnWithoutHits(noHit));
    assertFalse(AdvisorPrompt.hasContextSection(noHit));
  }

  @Test
  void abstentionMatchesPrefixWhenModelAddsExplanation() {
    assertTrue(AdvisorPrompt.isAbstention(
      RagPrompts.ABSTAIN_REPLY + " when Context is absent, empty, or off-topic."));
  }

  @Test
  void mixRejectsBlankBase() {
    assertThrows(IllegalArgumentException.class,
      () -> AdvisorPrompt.mix("  ", List.of("note")));
  }
}
