package io.nanovllm.chat;

import java.io.PrintStream;

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

  public StreamPrinter(PrintStream thinkOut, PrintStream answerOut, boolean color) {
    this.thinkOut = thinkOut;
    this.answerOut = answerOut;
    this.color = color;
  }

  public void update(AssistantParts parts) {
    this.emitThink(parts.thinking());
    if (!parts.thinkOpen() && this.thinkStarted && !this.thinkClosed) {
      this.closeThinkLine();
    }
    if (!parts.thinkOpen()) {
      this.emitAnswer(FactMemory.stripMemoryDirectives(parts.answer()));
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

  private void emitThink(String think) {
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
    if (think.length() > this.shownThink.length() && think.startsWith(this.shownThink)) {
      this.thinkOut.print(think.substring(this.shownThink.length()));
      this.thinkOut.flush();
      this.shownThink = think;
    }
  }

  private void closeThinkLine() {
    if (this.color) {
      this.thinkOut.print(ANSI_RESET);
    }
    this.thinkOut.println();
    this.thinkOut.flush();
    this.thinkClosed = true;
  }

  private void emitAnswer(String answer) {
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
    // Corrected rewrite (e.g. "I am …" → "Your name is …"): replace the line.
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
