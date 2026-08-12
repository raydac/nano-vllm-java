package com.igormaznitsa.nanollvm.samples;

import com.googlecode.lanterna.TerminalPosition;
import com.googlecode.lanterna.input.CharacterPattern;
import com.googlecode.lanterna.input.KeyDecodingProfile;
import com.googlecode.lanterna.input.MouseAction;
import com.googlecode.lanterna.input.MouseActionType;
import java.util.List;

/**
 * Decodes xterm SGR mouse reports ({@code CSI < Pb ; Px ; Py M/m}). Lanterna 3.1.2 only ships the
 * legacy X10 decoder ({@code CSI M Cb Cx Cy}), which many modern terminals no longer send for clicks.
 */
final class SgrMouseCharacterPattern implements CharacterPattern {

  @Override
  public Matching match(final List<Character> seq) {
    if (seq.isEmpty()) {
      return Matching.NOT_YET;
    }
    if (seq.get(0) != KeyDecodingProfile.ESC_CODE) {
      return null;
    }
    if (seq.size() == 1) {
      return Matching.NOT_YET;
    }
    if (seq.get(1) != '[') {
      return null;
    }
    if (seq.size() == 2) {
      return Matching.NOT_YET;
    }
    if (seq.get(2) != '<') {
      return null;
    }

    for (int i = 3; i < seq.size(); i++) {
      char c = seq.get(i);
      if (c == 'M' || c == 'm') {
        return this.decode(seq.subList(3, i), c == 'M');
      }
      if (c != ';' && (c < '0' || c > '9')) {
        return null;
      }
    }
    return Matching.NOT_YET;
  }

  private Matching decode(final List<Character> body, final boolean pressOrMotion) {
    StringBuilder raw = new StringBuilder(body.size());
    for (Character character : body) {
      raw.append(character.charValue());
    }
    String[] parts = raw.toString().split(";", -1);
    if (parts.length != 3) {
      return null;
    }

    final int pb;
    final int px;
    final int py;
    try {
      pb = Integer.parseInt(parts[0]);
      px = Integer.parseInt(parts[1]);
      py = Integer.parseInt(parts[2]);
    } catch (NumberFormatException ex) {
      return null;
    }
    if (px < 1 || py < 1) {
      return null;
    }

    MouseActionType type;
    int button;
    if (!pressOrMotion) {
      type = MouseActionType.CLICK_RELEASE;
      button = (pb & 3) + 1;
      if (button == 4) {
        button = 0;
      }
    } else if ((pb & 64) != 0) {
      type = (pb & 1) == 0 ? MouseActionType.SCROLL_UP : MouseActionType.SCROLL_DOWN;
      button = type == MouseActionType.SCROLL_UP ? 4 : 5;
    } else if ((pb & 32) != 0) {
      int encoded = (pb & 3) + 1;
      if (encoded == 4) {
        type = MouseActionType.MOVE;
        button = 0;
      } else {
        type = MouseActionType.DRAG;
        button = encoded;
      }
    } else {
      type = MouseActionType.CLICK_DOWN;
      button = (pb & 3) + 1;
    }

    return new Matching(new MouseAction(type, button, new TerminalPosition(px - 1, py - 1)));
  }
}
