package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.LLM;
import java.io.PrintStream;
import java.util.Locale;

/**
 * Factories and status helpers for {@link LlmListener}. Chat text and engine status share this path.
 */
public final class LlmListeners {

  private LlmListeners() {
  }

  /**
   * No-op listener (load/generate status and chat text are discarded).
   */
  public static LlmListener silent() {
    return Silent.INSTANCE;
  }

  /**
   * {@code true} when {@code listener} is {@code null} or {@link #silent()}.
   */
  public static boolean isSilent(final LlmListener listener) {
    return listener == null || listener == Silent.INSTANCE;
  }

  /**
   * Forwards each event to {@code first} then {@code second}. {@code null} / silent sides collapse
   * so a single listener is not wrapped.
   */
  public static LlmListener compose(final LlmListener first, final LlmListener second) {
    LlmListener left = first == null ? Silent.INSTANCE : first;
    LlmListener right = second == null ? Silent.INSTANCE : second;
    if (left == Silent.INSTANCE) {
      return right;
    }
    if (right == Silent.INSTANCE) {
      return left;
    }
    return new Composite(left, right);
  }

  /**
   * Status only: {@link LlmTextKind#STATUS_INFO} → {@code err},
   * {@link LlmTextKind#STATUS_PROGRESS} → {@code out}.
   */
  public static LlmListener toSystem() {
    return ofStatusStreams(System.out, System.err);
  }

  /**
   * Status only: {@link LlmTextKind#STATUS_PROGRESS} → {@code out},
   * {@link LlmTextKind#STATUS_INFO} → {@code err}. Chat text is ignored.
   *
   * @param out progress sink; must not be {@code null}
   * @param err info sink; must not be {@code null}
   */
  public static LlmListener ofStatusStreams(final PrintStream out, final PrintStream err) {
    PrintStream progress = requireNonNull(out, "out");
    PrintStream info = requireNonNull(err, "err");
    return (source, event) -> {
      switch (event.kind()) {
        case STATUS_PROGRESS -> {
          progress.print(event.text());
          progress.flush();
        }
        case STATUS_INFO -> {
          info.print(event.text());
          info.flush();
        }
        default -> {
        }
      }
    };
  }

  /**
   * CLI chat adapter: thinking / advisor notes → {@code thinkOut}, answer → {@code answerOut}.
   */
  public static LlmListener toPrintStreams(
    final PrintStream thinkOut,
    final PrintStream answerOut,
    final boolean color
  ) {
    return new PrintStreamLlmListener(
      new StreamPrinter(
        requireNonNull(thinkOut, "thinkOut"),
        requireNonNull(answerOut, "answerOut"),
        color));
  }

  /**
   * Emits {@link LlmTextKind#STATUS_INFO} (appends a newline when missing).
   */
  public static void info(final LlmListener listener, final LLM source, final String message) {
    String body = message == null ? "" : message;
    emit(listener, source, LlmTextKind.STATUS_INFO, body.endsWith("\n") ? body : body + "\n");
  }

  /**
   * {@link String#format(Locale, String, Object...)} then {@link LlmTextKind#STATUS_INFO}
   * ({@link Locale#ROOT}).
   */
  @SuppressWarnings("AnnotateFormatMethod")
  public static void infof(
    final LlmListener listener,
    final LLM source,
    final String format,
    final Object... args
  ) {
    emit(listener, source, LlmTextKind.STATUS_INFO, String.format(Locale.ROOT, format, args));
  }

  /**
   * {@link String#format(Locale, String, Object...)} then {@link LlmTextKind#STATUS_PROGRESS}
   * ({@link Locale#ROOT}).
   */
  @SuppressWarnings("AnnotateFormatMethod")
  public static void progressf(
    final LlmListener listener,
    final LLM source,
    final String format,
    final Object... args
  ) {
    emit(listener, source, LlmTextKind.STATUS_PROGRESS, String.format(Locale.ROOT, format, args));
  }

  private static void emit(
    final LlmListener listener,
    final LLM source,
    final LlmTextKind kind,
    final String text
  ) {
    LlmListener sink = listener == null ? Silent.INSTANCE : listener;
    sink.onText(source, LlmTextEvent.of(kind, text));
  }

  static PrintStreamLlmListener unwrapPrintStream(final LlmListener listener) {
    if (listener instanceof PrintStreamLlmListener print) {
      return print;
    }
    if (listener instanceof Composite(LlmListener left, LlmListener right)) {
      PrintStreamLlmListener fromLeft = unwrapPrintStream(left);
      if (fromLeft != null) {
        return fromLeft;
      }
      return unwrapPrintStream(right);
    }
    return null;
  }

  enum Silent implements LlmListener {
    INSTANCE;

    /**
     * {@inheritDoc}
     */
    @Override
    public void onText(final LLM source, final LlmTextEvent event) {
    }
  }

  private record Composite(LlmListener left, LlmListener right) implements LlmListener {
    /** {@inheritDoc} */
    @Override
    public void onText(final LLM source, final LlmTextEvent event) {
      this.left.onText(source, event);
      this.right.onText(source, event);
    }
  }

  static final class PrintStreamLlmListener implements LlmListener {

    private final StreamPrinter printer;
    private String think = "";
    private String answer = "";

    PrintStreamLlmListener(final StreamPrinter printer) {
      this.printer = printer;
    }

    /** {@inheritDoc} */
    @Override
    public void onText(final LLM source, final LlmTextEvent event) {
      switch (event.kind()) {
        case TEXT_ADVISOR_NOTE -> this.printer.emitAdvisorNote(event.advisorName(), event.text());
        case TEXT_DEBUG -> this.printer.emitDebug(event.text());
        case TEXT_THINKING -> {
          this.think = this.merge(this.think, event);
          this.printer.update(new ChatReply(this.think, this.answer, true));
        }
        case TEXT_ASSISTANT -> {
          this.answer = this.merge(this.answer, event);
          this.printer.update(new ChatReply(this.think, this.answer, false));
        }
        case TEXT_DIAGNOSTICS, TEXT_RAW, STATUS_INFO, STATUS_PROGRESS -> {
        }
      }
    }

    void closeTurn() {
      this.printer.closeTurn();
    }

    void discardAnswer() {
      this.printer.discardAnswer();
      this.answer = "";
    }

    void resetTurn() {
      this.think = "";
      this.answer = "";
      this.printer.reset();
    }

    private String merge(final String previous, final LlmTextEvent event) {
      if (event.snapshot()) {
        return event.text();
      }
      return previous + event.text();
    }
  }
}
