package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatRole;
import com.igormaznitsa.nanollvm.prompts.AdvisorPrompts;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class AdvisorPromptTest {

  @Test
  void withGeneratedNotesInsertsIntoFactsBeforeQuestion() {
    String mixed = AdvisorPrompt.mix(
      RagPrompts.withContext("capital of France?", "- Paris is in France."),
      List.of("Paris is the capital.", "Confirm from geography."));
    assertTrue(mixed.endsWith("capital of France?"), mixed);
    assertTrue(mixed.contains(AdvisorPrompts.mixNoteLine("Paris is the capital.")), mixed);
    assertTrue(mixed.indexOf("Paris is in France.") < mixed.indexOf("capital of France?"), mixed);
    assertTrue(mixed.indexOf("Paris is the capital.") < mixed.indexOf("capital of France?"), mixed);
  }

  @Test
  void mixSkipsEmptyAndDedupesIdenticalNotes() {
    String mixed = AdvisorPrompt.mix(
      "Question?",
      List.of("Practical note.", "", "Practical note.", "Future note."));
    assertTrue(mixed.contains(AdvisorPrompts.mixNoteLine("Practical note.")));
    assertTrue(mixed.contains(AdvisorPrompts.mixNoteLine("Future note.")));
    assertTrue(mixed.endsWith("Question?"), mixed);
    assertEquals(1,
      mixed.split(Pattern.quote(AdvisorPrompts.mixNoteLine("Practical note.")), -1).length - 1);
  }

  @Test
  void mixMapsUsefulNotesIntoFactsBlock() {
    String mixed = AdvisorPrompt.mix(
      RagPrompts.withContext("who are the grimm brothers?",
        "- The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.\n"
          + "- The Brothers Grimm were born in Hanau, Germany."),
      List.of(
        "",
        "The Brothers Grimm were born in Hanau, Germany."));
    assertTrue(mixed.endsWith("who are the grimm brothers?"), mixed);
    assertTrue(mixed.contains(AdvisorPrompts.mixNoteLine(
      "The Brothers Grimm were born in Hanau, Germany.")), mixed);
    assertFalse(mixed.contains("Context:"), mixed);
  }

  @Test
  void mixOmitsBlockWhenEveryAdvisorIsEmpty() {
    String base = RagPrompts.withContext(
      "who are the grimm brothers?",
      "- The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.");
    String mixed = AdvisorPrompt.mix(base, List.of("", "  "));
    assertEquals(base.strip(), mixed);
  }

  @Test
  void mixKeepsQuestionLast() {
    String base = RagPrompts.withContext(
      "what was the destination?",
      "- Take them to your grandmother.");
    String mixed = AdvisorPrompt.mix(base, List.of("Take them to your grandmother."));
    assertTrue(mixed.endsWith("what was the destination?"), mixed);
    assertTrue(mixed.contains(AdvisorPrompts.mixNoteLine("Take them to your grandmother.")));
    int factsAt = mixed.indexOf("- Take them to your grandmother.");
    int questionAt = mixed.lastIndexOf("what was the destination?");
    assertTrue(factsAt >= 0 && questionAt > factsAt);
  }

  @Test
  void dialogTurnUsesCallerRolePromptAsSystem() {
    List<ChatMessage> turn = AdvisorPrompt.dialogTurn(
      "Practical viewpoint.",
      List.of(),
      "hello");
    assertEquals(ChatRole.SYSTEM, turn.getFirst().role());
    assertEquals("Practical viewpoint.", turn.getFirst().content());
  }

  @Test
  void selectNotesKeepsOnlyContextGroundedNotes() {
    String ragUser = RagPrompts.withContext(
      "what was their main interest?",
      "- Jacob and Wilhelm Grimm were German authors of fairy tales and folklore.");
    List<String> selected = AdvisorPrompt.selectNotesForMix(ragUser, List.of(
      "Jacob Grimm and Wilhelm Grimm were famous German authors of fairy tales and folklore.",
      "",
      "I think the Grimm brothers are fascinating."));
    assertEquals(List.of(
        "Jacob Grimm and Wilhelm Grimm were famous German authors of fairy tales and folklore."),
      selected);
  }

  @Test
  void usableNoteDropsBlank() {
    assertEquals("", AdvisorPrompt.usableNote(""));
    assertEquals("", AdvisorPrompt.usableNote("  "));
    assertEquals("", AdvisorPrompt.usableNote(null));
    assertEquals("Useful hint.", AdvisorPrompt.usableNote(" Useful hint. "));
    assertEquals("", AdvisorPrompt.usableNote(
      "Okay, I understand.",
      note -> !note.toLowerCase(java.util.Locale.ROOT).contains("understand")));
  }

  @Test
  void mixKeepsRealNotes() {
    String mixed = AdvisorPrompt.mix(
      "who was their father?",
      List.of("Philipp Wilhelm Grimm was their father."));
    assertTrue(mixed.contains("Philipp Wilhelm Grimm was their father."), mixed);
    assertTrue(mixed.endsWith("who was their father?"), mixed);
  }

  @Test
  void selectNotesKeepsGroundedNameFacts() {
    String ragUser = RagPrompts.withContext(
      "what do you think about the grimm brothers?",
      "- The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.");
    List<String> selected = AdvisorPrompt.selectNotesForMix(ragUser, List.of(
      "The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.",
      "I think the Grimm brothers are a fascinating and complex group of individuals."));
    assertEquals(List.of("The Brothers Grimm were Jacob Grimm and Wilhelm Grimm."), selected);
  }

  @Test
  void dialogTurnIncludesPriorUsersThenAdvisorFacingUser() {
    List<ChatMessage> prior = List.of(
      ChatMessage.user("what do you think about the grimm brothers?"),
      ChatMessage.assistant("They were Jacob and Wilhelm Grimm."),
      ChatMessage.user("and who was their father?"));
    String prepared = RagPrompts.withContext(
      "and who was their father?",
      "- Their father was Philipp Wilhelm Grimm.");
    List<ChatMessage> turn = AdvisorPrompt.dialogTurn("Practical viewpoint.", prior, prepared);

    assertEquals(ChatRole.SYSTEM, turn.getFirst().role());
    assertEquals("Practical viewpoint.", turn.getFirst().content());
    assertEquals(ChatRole.USER, turn.get(1).role());
    assertEquals("what do you think about the grimm brothers?", turn.get(1).content());
    assertEquals(ChatRole.USER, turn.get(2).role());
    assertEquals("and who was their father?", turn.get(2).content());
    assertEquals(ChatRole.USER, turn.getLast().role());
    assertTrue(turn.getLast().content().contains("Philipp Wilhelm Grimm"));
    assertTrue(turn.getLast().content().endsWith("and who was their father?"));
    assertEquals(4, turn.size());
    assertTrue(turn.stream().noneMatch(m -> m.role() == ChatRole.ASSISTANT));
  }

  @Test
  void advisorFacingUserKeepsFactsThenQuestion() {
    String prepared = RagPrompts.withContext(
      "who are the grimm brothers?",
      "- The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.");
    String facing = AdvisorPrompt.advisorFacingUserText(prepared);
    assertTrue(facing.endsWith("who are the grimm brothers?"));
    assertTrue(facing.contains("Jacob Grimm and Wilhelm Grimm"));
    assertFalse(facing.contains("Context:"));
  }

  @Test
  void extractContextBlockIsFactsOnly() {
    String prepared = RagPrompts.withContext(
      "who are the grimm brothers?",
      "- The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.");
    String context = AdvisorPrompt.extractContextBlock(prepared);
    assertEquals("- The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.", context);
  }

  @Test
  void selectNotesDropsUngroundedPriorLatchWhenContextIsNamesOnly() {
    String ragUser = RagPrompts.withContext(
      "not very clear, I want to know their names",
      "- Their names were Jacob Grimm and Wilhelm Grimm.");
    List<String> selected = AdvisorPrompt.selectNotesForMix(ragUser, List.of(
      "The Brothers Grimm were among these seven brave men.",
      "The Brothers Grimm were among these seven brave men.",
      "Their names were Jacob Grimm and Wilhelm Grimm."));
    assertEquals(List.of("Their names were Jacob Grimm and Wilhelm Grimm."), selected);
  }

  @Test
  void mixDedupesIdenticalAdvisorNotes() {
    String mixed = AdvisorPrompt.mix(
      RagPrompts.withContext("who are the grimm brothers?",
        "- The Brothers Grimm were Jacob Grimm and Wilhelm Grimm."),
      List.of(
        "The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.",
        "The Brothers Grimm were Jacob Grimm and Wilhelm Grimm."));
    String line = AdvisorPrompts.mixNoteLine(
      "The Brothers Grimm were Jacob Grimm and Wilhelm Grimm.");
    assertTrue(mixed.endsWith("who are the grimm brothers?"), mixed);
    assertEquals(1, mixed.split(Pattern.quote(line), -1).length - 1, mixed);
  }

  @Test
  void ragTurnWithoutHitsSkipsContextGrounding() {
    String noHit = RagPrompts.withoutContext("что такое альфа центавра?");
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
