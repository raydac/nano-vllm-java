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
   *
   * @return the interned silent sink
   */
  public static LlmListener silent() {
    return Silent.INSTANCE;
  }

  /**
   * {@code true} when {@code listener} is {@code null} or {@link #silent()}.
   *
   * @param listener sink to test; {@code null} counts as silent
   * @return whether events would be discarded
   */
  public static boolean isSilent(final LlmListener listener) {
    return listener == null || listener == Silent.INSTANCE;
  }

  /**
   * Forwards each event to {@code first} then {@code second}. {@code null} / silent sides collapse
   * so a single listener is not wrapped.
   *
   * @param first  left sink; {@code null} → silent
   * @param second right sink; {@code null} → silent
   * @return {@code second}, {@code first}, or a composite
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
   *
   * @return listener writing to {@link System#out} / {@link System#err}
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
   * @return status-only listener
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
   *
   * @param thinkOut  thinking / advisor / debug sink
   * @param answerOut visible assistant answer sink
   * @param color     when {@code true}, dim cyan ANSI styling on the thinking stream
   * @return session-oriented listener wrapping a {@link StreamPrinter}
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
   *
   * @param listener sink; {@code null} → silent
   * @param source   engine, or {@code null} during load
   * @param message  info line
   */
  public static void info(final LlmListener listener, final LLM source, final String message) {
    String body = message == null ? "" : message;
    emit(listener, source, LlmTextKind.STATUS_INFO, body.endsWith("\n") ? body : body + "\n");
  }

  /**
   * {@link String#format(Locale, String, Object...)} then {@link LlmTextKind#STATUS_INFO}
   * ({@link Locale#ROOT}).
   *
   * @param listener sink; {@code null} → silent
   * @param source   engine, or {@code null} during load
   * @param format   {@link Locale#ROOT} format string
   * @param args     format arguments
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
   *
   * @param listener sink; {@code null} → silent
   * @param source   engine, or {@code null} during load
   * @param format   {@link Locale#ROOT} format string
   * @param args     format arguments
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

  /**
   * Forwards {@code text} as {@code kind} to {@code listener} ({@code null} → silent).
   */
  private static void emit(
    final LlmListener listener,
    final LLM source,
    final LlmTextKind kind,
    final String text
  ) {
    LlmListener sink = listener == null ? Silent.INSTANCE : listener;
    sink.onText(source, LlmTextEvent.of(kind, text));
  }

  /**
   * Finds a {@link PrintStreamLlmListener} inside {@code listener} or a {@link Composite}, so
   * {@link ChatSession} can reset / discard / close a CLI turn.
   */
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

  /**
   * Interned no-op {@link LlmListener}.
   */
  enum Silent implements LlmListener {
    INSTANCE;

    /**
     * {@inheritDoc}
     */
    @Override
    public void onText(final LLM source, final LlmTextEvent event) {
    }
  }

  /**
   * Forwards each event to {@code left} then {@code right}.
   */
  private record Composite(LlmListener left, LlmListener right) implements LlmListener {
    /** {@inheritDoc} */
    @Override
    public void onText(final LLM source, final LlmTextEvent event) {
      this.left.onText(source, event);
      this.right.onText(source, event);
    }
  }

  /**
   * Session CLI adapter: accumulates thinking/answer and drives a {@link StreamPrinter}.
   * {@link LlmTextKind#TEXT_RAW} and status kinds are ignored (the session already prints parsed
   * channels).
   */
  static final class PrintStreamLlmListener implements LlmListener {

    private final StreamPrinter printer;
    private String think = "";
    private String answer = "";

    /**
     * Binds the CLI printer that this adapter drives.
     */
    PrintStreamLlmListener(final StreamPrinter printer) {
      this.printer = requireNonNull(printer, "printer");
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

    /**
     * Ends the printed think/answer lines for this turn.
     */
    void closeTurn() {
      this.printer.closeTurn();
    }

    /**
     * Clears a partially streamed answer so a retry can reprint {@code assistant>}.
     */
    void discardAnswer() {
      this.printer.discardAnswer();
      this.answer = "";
    }

    /**
     * Clears accumulated think/answer and the printer for a new generate or retry.
     */
    void resetTurn() {
      this.think = "";
      this.answer = "";
      this.printer.reset();
    }

    /**
     * Appends a delta or replaces with a snapshot.
     */
    private String merge(final String previous, final LlmTextEvent event) {
      if (event.snapshot()) {
        return event.text();
      }
      return previous + event.text();
    }
  }
}
