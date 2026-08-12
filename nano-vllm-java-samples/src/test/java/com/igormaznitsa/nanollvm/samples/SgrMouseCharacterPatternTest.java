package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.googlecode.lanterna.input.CharacterPattern;
import com.googlecode.lanterna.input.KeyType;
import com.googlecode.lanterna.input.MouseAction;
import com.googlecode.lanterna.input.MouseActionType;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class SgrMouseCharacterPatternTest {

  private final SgrMouseCharacterPattern pattern = new SgrMouseCharacterPattern();

  @Test
  void decodesLeftClickDown() {
    MouseAction action = this.match("\u001b[<0;12;5M");
    assertEquals(MouseActionType.CLICK_DOWN, action.getActionType());
    assertEquals(1, action.getButton());
    assertEquals(11, action.getPosition().getColumn());
    assertEquals(4, action.getPosition().getRow());
  }

  @Test
  void decodesLeftClickRelease() {
    MouseAction action = this.match("\u001b[<0;12;5m");
    assertEquals(MouseActionType.CLICK_RELEASE, action.getActionType());
  }

  @Test
  void decodesMove() {
    MouseAction action = this.match("\u001b[<35;3;2M");
    assertEquals(MouseActionType.MOVE, action.getActionType());
    assertEquals(0, action.getButton());
  }

  private MouseAction match(final String sequence) {
    List<Character> chars = new ArrayList<>();
    CharacterPattern.Matching matching = null;
    for (int i = 0; i < sequence.length(); i++) {
      chars.add(sequence.charAt(i));
      matching = this.pattern.match(chars);
      assertNotNull(matching, "partial match failed at " + i);
      if (i < sequence.length() - 1) {
        assertEquals(CharacterPattern.Matching.NOT_YET, matching);
      }
    }
    assertNotNull(matching);
    assertNotNull(matching.fullMatch);
    assertTrue(matching.fullMatch instanceof MouseAction);
    assertEquals(KeyType.MouseEvent, matching.fullMatch.getKeyType());
    return (MouseAction) matching.fullMatch;
  }
}