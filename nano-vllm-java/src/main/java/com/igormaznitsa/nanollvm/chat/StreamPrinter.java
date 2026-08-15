package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import java.io.PrintStream;

/**
 * Incremental CLI printer for thinking / answer streams. Session-scoped and not thread-safe;
 * prefer {@link LlmListeners#toPrintStreams} for typical use.
 */
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
    this.thinkOut = requireNonNull(thinkOut, "thinkOut");
    this.answerOut = requireNonNull(answerOut, "answerOut");
    this.color = color;
  }

  public void reset() {
    this.shownThink = "";
    this.shownAnswer = "";
    this.thinkStarted = false;
    this.thinkClosed = false;
    this.answerStarted = false;
  }

  public void update(final ChatReply parts) {
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
   * Clears a partially streamed answer line so a retry can print a fresh {@code assistant>} without
   * leaving a second answer on screen.
   */
  public void discardAnswer() {
    if (this.answerStarted) {
      int width = "assistant> ".length() + Math.max(this.shownAnswer.length(), 8);
      this.answerOut.print('\r');
      this.answerOut.print(" ".repeat(width));
      this.answerOut.print('\r');
      this.answerOut.flush();
    }
    this.shownAnswer = "";
    this.answerStarted = false;
  }

  public void emitAdvisorNote(final String advisorName, final String note) {
    String name = requireNonNull(advisorName, "advisorName").strip();
    if (name.isEmpty()) {
      throw new IllegalArgumentException("advisorName must not be blank");
    }
    String body = note == null ? "" : note;
    this.thinkOut.print("thinking> ");
    if (this.color) {
      this.thinkOut.print(ANSI_THINK);
    }
    this.thinkOut.printf("[%s] %s", name, body);
    if (this.color) {
      this.thinkOut.print(ANSI_RESET);
    }
    this.thinkOut.println();
    this.thinkOut.flush();
  }

  public void emitDebug(final String text) {
    String body = text == null ? "" : text;
    this.thinkOut.println("debug> --- prepared model user ---");
    if (body.isEmpty()) {
      this.thinkOut.println("debug> (empty)");
    } else {
      body.lines().forEach(line -> this.thinkOut.println("debug> " + line));
    }
    this.thinkOut.println("debug> ---");
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
