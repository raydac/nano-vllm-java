package com.igormaznitsa.nanollvm.chat;

import java.io.PrintStream;
import java.util.List;

public final class StreamPrinter {

  private static final String ANSI_THINK = "\u001B[2;36m";
  private static final String ANSI_RESET = "\u001B[0m";

  private final PrintStream thinkOut;
  private final PrintStream answerOut;
  private final boolean color;
  private String shownThink = "";
  private String shownAnswer = "";
  private boolean thinkStarted;
  private boolean thinkClosed;
  private boolean answerStarted;

  public StreamPrinter(final PrintStream thinkOut, final PrintStream answerOut,
                       final boolean color) {
    this.thinkOut = thinkOut;
    this.answerOut = answerOut;
    this.color = color;
  }

  public void update(final AssistantParts parts) {
    this.emitThink(parts.thinking());
    if (!parts.thinkOpen() && this.thinkStarted && !this.thinkClosed) {
      this.closeThinkLine();
    }
    if (!parts.thinkOpen()) {
      this.emitAnswer(parts.answer());
    }
  }

  public void closeTurn() {
    if (this.thinkStarted && !this.thinkClosed) {
      this.closeThinkLine();
    }
    if (!this.answerStarted) {
      this.answerOut.print("assistant> ");
      this.answerStarted = true;
    }
    this.answerOut.println();
    this.answerOut.flush();
  }

  /**
   * Writes completed advisor notes to the thinking stream before the main generate starts.
   * Each note is a full line so the main model's incremental {@code thinking>} state stays clean.
   */
  public void emitAdvisorNotes(final List<String> notes) {
    if (notes == null || notes.isEmpty()) {
      return;
    }
    for (int i = 0; i < notes.size(); i++) {
      String note = notes.get(i) == null ? "" : notes.get(i).strip();
      if (note.isEmpty()) {
        continue;
      }
      this.thinkOut.print("thinking> ");
      if (this.color) {
        this.thinkOut.print(ANSI_THINK);
      }
      this.thinkOut.print("[subagent %d] %s".formatted(i + 1, note));
      if (this.color) {
        this.thinkOut.print(ANSI_RESET);
      }
      this.thinkOut.println();
    }
    this.thinkOut.flush();
  }

  private void emitThink(final String think) {
    if (think.isEmpty()) {
      return;
    }
    if (!this.thinkStarted) {
      this.thinkOut.print("thinking> ");
      if (this.color) {
        this.thinkOut.print(ANSI_THINK);
      }
      this.thinkStarted = true;
    }
    if (think.equals(this.shownThink)) {
      return;
    }
    if (think.length() > this.shownThink.length() && think.startsWith(this.shownThink)) {
      this.thinkOut.print(think.substring(this.shownThink.length()));
      this.thinkOut.flush();
      this.shownThink = think;
      return;
    }
    this.thinkOut.print("\rthinking> ");
    if (this.color) {
      this.thinkOut.print(ANSI_THINK);
    }
    this.thinkOut.print(think);
    if (this.shownThink.length() > think.length()) {
      this.thinkOut.print(" ".repeat(this.shownThink.length() - think.length()));
      this.thinkOut.print("\rthinking> ");
      if (this.color) {
        this.thinkOut.print(ANSI_THINK);
      }
      this.thinkOut.print(think);
    }
    this.thinkOut.flush();
    this.shownThink = think;
  }

  private void closeThinkLine() {
    if (this.color) {
      this.thinkOut.print(ANSI_RESET);
    }
    this.thinkOut.println();
    this.thinkOut.flush();
    this.thinkClosed = true;
  }

  private void emitAnswer(final String answer) {
    if (answer.isEmpty()) {
      return;
    }
    if (!this.answerStarted) {
      this.answerOut.print("assistant> ");
      this.answerStarted = true;
    }
    if (answer.equals(this.shownAnswer)) {
      return;
    }
    if (answer.length() > this.shownAnswer.length() && answer.startsWith(this.shownAnswer)) {
      this.answerOut.print(answer.substring(this.shownAnswer.length()));
      this.answerOut.flush();
      this.shownAnswer = answer;
      return;
    }
    this.answerOut.print("\rassistant> ");
    this.answerOut.print(answer);
    if (this.shownAnswer.length() > answer.length()) {
      this.answerOut.print(" ".repeat(this.shownAnswer.length() - answer.length()));
      this.answerOut.print("\rassistant> ");
      this.answerOut.print(answer);
    }
    this.answerOut.flush();
    this.shownAnswer = answer;
  }
}
