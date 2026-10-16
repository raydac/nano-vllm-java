package com.igormaznitsa.nanollvm.jev;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import org.junit.jupiter.api.Test;

final class JevTest {

  private static List<List<String>> shown(final List<String> options, final List<int[]> rotations) {
    return rotations.stream()
      .map(permutation -> Arrays.stream(permutation).mapToObj(options::get).toList())
      .toList();
  }

  @Test
  void choiceRejectsABlankDuplicateOrOversizedSet() {
    assertThrows(IllegalArgumentException.class,
      () -> JevQuestion.choice("  ", List.of("a", "b")));
    assertThrows(IllegalArgumentException.class,
      () -> JevQuestion.choice("Which?", List.of("a")));
    assertThrows(IllegalArgumentException.class,
      () -> JevQuestion.choice("Which?", List.of("a", "a")));
    assertThrows(IllegalArgumentException.class,
      () -> JevQuestion.choice("Which?", List.of("a", " ")));
  }

  @Test
  void yesNoAndScoreExposeStableOptions() {
    JevQuestion yesNo = JevQuestion.yesNo("Is it down?").named("health");
    assertEquals(JevQuestion.Kind.YES_NO, yesNo.kind());
    assertEquals(List.of("Yes", "No"), yesNo.options());
    assertEquals("health", yesNo.name());
    assertFalse(yesNo.isScore());

    JevQuestion bins = JevQuestion.score("How full?", 2);
    assertEquals(List.of("0 to 0.5", "0.5 to 1"), bins.options());
    assertEquals(List.of(0.25d, 0.75d), bins.binCenters());
    assertEquals(List.of("0 to 10", "10 to 20"), bins.between(0d, 20d).options());

    JevQuestion levels = JevQuestion.score("How urgent?", List.of("low", "high"));
    assertEquals(List.of(0d, 1d), levels.binCenters());
    assertTrue(levels.isScore());
    assertThrows(IllegalArgumentException.class, () -> JevQuestion.score("Rate", 1));
    assertThrows(IllegalArgumentException.class,
      () -> JevQuestion.score("Rate", 2).between(1d, 1d));
  }

  @Test
  void positionBiasCancelsAndContentSurvives() {
    List<int[]> rotations = List.of(new int[] {0, 1}, new int[] {1, 0});
    double[][] positionOnly = {new double[] {0.7d, 0.3d}, new double[] {0.7d, 0.3d}};
    double[] cancelled = JevMarginal.marginalize(positionOnly, rotations);
    assertEquals(0.5d, cancelled[0], 1e-9);
    assertEquals(0.5d, cancelled[1], 1e-9);

    double[][] content = {new double[] {0.8d, 0.2d}, new double[] {0.2d, 0.8d}};
    double[] kept = JevMarginal.marginalize(content, rotations);
    assertEquals(0.8d, kept[0], 1e-9);
    assertEquals(0.2d, kept[1], 1e-9);
    assertEquals(0d, JevMarginal.orderFlipRate(content, rotations), 0d);
    assertEquals(1d, JevMarginal.orderFlipRate(positionOnly, rotations), 0d);
  }

  @Test
  void priorBoostsTheRareLabel() {
    double[][] positions = {new double[] {0.5d, 0.5d}};
    double[][] prior = {new double[] {0.8d, 0.2d}};
    double[][] corrected = JevMarginal.applyPrior(positions, prior, 1d);
    assertEquals(0.2d, corrected[0][0], 1e-9);
    assertEquals(0.8d, corrected[0][1], 1e-9);
  }

  @Test
  void rotationsFollowTheLevelAndCanonicalOptionSet() {
    JevQuestion choice =
      JevQuestion.choice("Which team?", List.of("technical", "billing", "sales"));
    assertEquals(1, JevMarginal.rotations(choice, JevLevel.RAW).size());
    assertEquals(3, JevMarginal.rotations(choice, JevLevel.L0).size());

    List<String> reversed = List.of("sales", "billing", "technical");
    assertEquals(
      shown(choice.options(), JevMarginal.choiceRotations(choice.options())),
      shown(reversed, JevMarginal.choiceRotations(reversed)));

    JevQuestion yesNo = JevQuestion.yesNo("Down?");
    assertEquals(2, JevMarginal.rotations(yesNo, JevLevel.L0).size());
    JevQuestion score = JevQuestion.score("How full?");
    assertEquals(1, JevMarginal.rotations(score, JevLevel.L0).size());
  }

  @Test
  void promptListsTheRotatedOptions() {
    JevQuestion choice = JevQuestion.choice("Which team?", List.of("billing", "technical"));
    String user = JevPrompts.userText(
      "  ", choice, new int[] {1, 0}, List.of("A", "B"));
    assertTrue(user.contains("State:\n(empty)"));
    assertTrue(user.contains("Question: Which team?"));
    assertTrue(user.contains("A. technical"));
    assertTrue(user.contains("B. billing"));
    assertTrue(user.endsWith("Answer with the letter only."));

    String yesNo = JevPrompts.userText(
      "The page returns 500.",
      JevQuestion.yesNo("Is this a technical failure?"),
      new int[] {1, 0},
      List.of("Yes", "No"));
    assertTrue(yesNo.contains("Answer No or Yes."));
  }

  @Test
  void labelsMustBeOneRealTokenAndScoreFallsBackToLetters() {
    Tokenizer tokenizer = Tokenizer.fromJsonDocuments("""
        {
          "model": {
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "A": 1, "B": 2, "Yes": 3, "No": 4, "</s>": 5}
          }
        }
        """,
      "{\"eos_token\":\"</s>\"}",
      null,
      "{\"vocab_size\":6}");

    JevLabels.Resolved choice = JevLabels.resolve(
      tokenizer, JevQuestion.choice("Which?", List.of("billing", "technical")));
    assertEquals(List.of("A", "B"), choice.labels());
    assertEquals(1, choice.ids()[0]);
    assertEquals(2, choice.ids()[1]);

    assertThrows(IllegalArgumentException.class, () -> JevLabels.resolve(
      tokenizer, JevQuestion.choice("Which?", List.of("a", "b", "c"))));

    JevLabels.Resolved score = JevLabels.resolve(tokenizer, JevQuestion.score("How full?", 2));
    assertEquals(List.of("A", "B"), score.labels());

    int[] flipped = JevLabels.idsForRotation(
      JevQuestion.yesNo("Down?"), new int[] {3, 4}, new int[] {1, 0});
    assertEquals(4, flipped[0]);
    assertEquals(3, flipped[1]);
  }

  @Test
  void decisionReportsTheWinnerAndRefusesAWeakerLevel() {
    JevDecision decision = new JevDecision(
      "route",
      JevLevel.RAW,
      "billing",
      List.of(new JevOption("billing", 0.7d), new JevOption("technical", 0.3d)),
      0.4d,
      1,
      0d,
      false,
      List.of());
    assertEquals(0.7d, decision.probability("billing"), 0d);
    assertTrue(decision.score().isEmpty());
    assertEquals(decision, decision.require(JevLevel.RAW));
    assertThrows(IllegalStateException.class, () -> decision.require(JevLevel.L0));
    assertThrows(IllegalArgumentException.class, () -> decision.probability("sales"));

    JevDecision scored = new JevDecision(
      "",
      JevLevel.L0,
      "0 to 0.5",
      List.of(new JevOption("0 to 0.5", 0.25d), new JevOption("0.5 to 1", 0.75d)),
      0.5d,
      1,
      0d,
      false,
      List.of(0.25d, 0.75d));
    OptionalDouble value = scored.score();
    assertTrue(value.isPresent());
    assertEquals(0.625d, value.getAsDouble(), 1e-9);
  }
}
