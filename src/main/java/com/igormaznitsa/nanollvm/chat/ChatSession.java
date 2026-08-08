package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.AdvisorEnrichment;
import com.igormaznitsa.nanollvm.llm.AdvisorRunner;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingDefaults;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Multi-turn chat over an {@link LLM}: history, chat template, truncation, reply parsing, and
 * {@link LlmListener} events.
 *
 * <p>Not thread-safe; use one session per conversation thread. Text and status events compose
 * {@link LLM#listener()} with any session {@link #listen(LlmListener)} / {@link #streamTo} sink.
 */
public final class ChatSession {

  private final LLM llm;
  private final List<ChatMessage> history;
  private SamplingParams samplingParams;
  private Duration timeout = Duration.ZERO;
  private LlmListener sessionListener = LlmListeners.silent();
  private LlmListener listener = LlmListeners.silent();
  private LlmListeners.PrintStreamLlmListener printSink;
  private Boolean enableThinking;

  /**
   * Opens a session with {@link LLM#defaultSampling()} and a fresh conversation seeded from the
   * model (system turn when the architecture uses one).
   *
   * @param llm engine that owns generation; must not be {@code null}
   */
  public ChatSession(final LLM llm) {
    this(llm, llm.defaultSampling());
  }

  /**
   * Opens a session with explicit sampling knobs and a fresh conversation seeded from the model.
   *
   * @param llm            engine that owns generation; must not be {@code null}
   * @param samplingParams temperature / top-k / top-p / max tokens for each {@link #send}; must not
   *                       be {@code null}
   */
  public ChatSession(final LLM llm, final SamplingParams samplingParams) {
    this.llm = requireNonNull(llm, "llm");
    this.samplingParams = requireNonNull(samplingParams, "samplingParams");
    this.history = new ArrayList<>(llm.newConversation());
    this.bindListener();
  }

  /**
   * Factory that pins a max new-token budget via {@link SamplingDefaults#forTokenizer}.
   *
   * @param llm       engine that owns generation
   * @param maxTokens upper bound on new tokens per turn (architecture-aware defaults for the rest)
   * @return a new session
   */
  public static ChatSession open(final LLM llm, final int maxTokens) {
    return new ChatSession(llm, SamplingDefaults.forTokenizer(llm.tokenizer(), maxTokens));
  }

  /**
   * Replaces sampling parameters for subsequent turns.
   *
   * @param samplingParams new knobs; must not be {@code null}
   * @return {@code this} for fluent configuration
   */
  public ChatSession sampling(final SamplingParams samplingParams) {
    this.samplingParams = requireNonNull(samplingParams, "samplingParams");
    return this;
  }

  /**
   * Sampling parameters currently used by {@link #send} / {@link #sendPrepared}.
   *
   * @return the live sampling params (not a defensive copy of nested state)
   */
  public SamplingParams samplingParams() {
    return this.samplingParams;
  }

  /**
   * Caps wall-clock time for the underlying {@link LLM#generate} of each turn.
   * {@code null} or non-positive means unbounded.
   *
   * @param timeout generate deadline, or {@code null} / zero / negative for no limit
   * @return {@code this} for fluent configuration
   */
  public ChatSession timeout(final Duration timeout) {
    this.timeout = timeout == null ? Duration.ZERO : timeout;
    return this;
  }

  /**
   * Sets the session-level {@link LlmListener}. Composed with {@link LLM#listener()} so status
   * events from the engine still reach the LLM sink. {@code null} clears the session extra to
   * {@link LlmListeners#silent()}.
   *
   * @param listener chat / diagnostics sink for this session, or {@code null} for silent
   * @return {@code this} for fluent configuration
   */
  public ChatSession listen(final LlmListener listener) {
    this.sessionListener = listener == null ? LlmListeners.silent() : listener;
    this.bindListener();
    return this;
  }

  /**
   * CLI sugar: installs {@link LlmListeners#toPrintStreams} as the session listener (thinking →
   * {@code thinkOut}, answer → {@code answerOut}).
   *
   * @param thinkOut  destination for thinking / advisor notes
   * @param answerOut destination for the visible assistant answer
   * @param color     when {@code true}, dim cyan ANSI styling on the thinking stream
   * @return {@code this} for fluent configuration
   */
  public ChatSession streamTo(final PrintStream thinkOut, final PrintStream answerOut,
                              final boolean color) {
    return this.listen(LlmListeners.toPrintStreams(thinkOut, answerOut, color));
  }

  /**
   * Enables or disables thinking-scratchpad invitation for this session.
   *
   * <p>When {@code false}, the chat template seeds an empty {@code <think></think>} so Qwen-style
   * models skip long chain-of-thought (important for RAG token budgets). When never called, the
   * default from {@link Tokenizer#invitesThinking()} applies.
   *
   * @param enableThinking {@code true} to invite chain-of-thought, {@code false} to suppress it
   * @return {@code this} for fluent configuration
   */
  public ChatSession enableThinking(final boolean enableThinking) {
    this.enableThinking = enableThinking;
    return this;
  }

  /**
   * Composes a diagnostics sink for {@link LlmTextKind#TEXT_DIAGNOSTICS} only.
   * Does not wipe a prior {@link #listen} / {@link #streamTo}; {@code null} is a no-op.
   *
   * @param diagnostics consumer of diagnostic lines (salvage, empty-reply fallback, …)
   * @return {@code this} for fluent configuration
   */
  public ChatSession diagnostics(final Consumer<String> diagnostics) {
    if (diagnostics == null) {
      return this;
    }
    this.sessionListener = LlmListeners.compose(this.sessionListener, (source, event) -> {
      if (event.kind() == LlmTextKind.TEXT_DIAGNOSTICS) {
        diagnostics.accept(event.text());
      }
    });
    this.bindListener();
    return this;
  }

  private void bindListener() {
    this.listener = LlmListeners.compose(this.llm.listener(), this.sessionListener);
    this.printSink = LlmListeners.unwrapPrintStream(this.sessionListener);
  }

  /**
   * Snapshot of the conversation so far (system seed plus user / assistant turns).
   *
   * @return an unmodifiable copy; mutations require {@link #send} / {@link #clear}
   */
  public List<ChatMessage> history() {
    return List.copyOf(this.history);
  }

  /**
   * Drops user/assistant turns and reseeds from {@link LLM#newConversation()} (keeps sampling,
   * timeout, and listeners).
   */
  public void clear() {
    this.history.clear();
    this.history.addAll(this.llm.newConversation());
  }

  /**
   * Appends {@code userText} to history, runs advisors when configured, generates a reply, and
   * appends the assistant answer to history.
   *
   * @param userText non-blank user turn
   * @return parsed thinking / answer for this turn
   * @throws IllegalArgumentException if {@code userText} is blank after strip
   * @throws NullPointerException     if {@code userText} is {@code null}
   */
  public ChatReply send(final String userText) {
    requireNonNull(userText, "userText");
    return this.sendPrepared(userText, userText);
  }

  /**
   * Stores {@code historyUserText} in history while generating from {@code modelUserText}
   * (RAG: short visible history, context-augmented model turn).
   *
   * @param historyUserText text recorded in {@link #history()}
   * @param modelUserText   text used in the chat template for this generate
   * @return parsed thinking / answer for this turn
   * @throws IllegalArgumentException if either text is blank after strip
   */
  public ChatReply sendPrepared(final String historyUserText, final String modelUserText) {
    return this.sendPrepared(historyUserText, modelUserText, false);
  }

  /**
   * Stores one user text in history but may generate from a different prepared user string.
   *
   * @param historyUserText   text recorded in {@link #history()}; must not be blank
   * @param modelUserText     text used for advisors and the chat template; must not be blank
   * @param isolateGeneration when {@code true}, the model sees only the system seed (if any) plus
   *                          the prepared user turn — prior assistant answers stay in history for
   *                          the app but are not fed into this generate (avoids tiny-model latch)
   * @return parsed thinking / answer for this turn
   * @throws IllegalArgumentException if either text is blank after strip
   */
  public ChatReply sendPrepared(
    final String historyUserText,
    final String modelUserText,
    final boolean isolateGeneration
  ) {
    requireNonNull(historyUserText, "historyUserText");
    requireNonNull(modelUserText, "modelUserText");
    String historyUser = historyUserText.strip();
    String modelUser = modelUserText.strip();
    if (historyUser.isEmpty()) {
      throw new IllegalArgumentException("historyUserText must not be blank");
    }
    if (modelUser.isEmpty()) {
      throw new IllegalArgumentException("modelUserText must not be blank");
    }

    this.history.add(ChatMessage.user(historyUser));
    Tokenizer tokenizer = this.llm.tokenizer();
    ChatMessages.truncateHistory(
      this.history, tokenizer, this.llm.config().maxModelLen(), this.samplingParams.maxTokens());

    boolean gemmaChat = tokenizer.isGemmaChat();
    TurnStream turn = this.beginTurn();
    AdvisorEnrichment enrichment =
      AdvisorRunner.enrich(this.llm, modelUser, this.samplingParams);
    this.emitAdvisorNotes(enrichment.advisorNotes());

    ChatReply reply = this.generateTurn(turn, enrichment.modelUserText(), isolateGeneration);

    if (gemmaChat && ChatPrompts.isSetupBoilerplate(reply.answer())) {
      ChatMessages.scrubSetupBoilerplateTurns(this.history);
      this.emitDiagnostics("(setup boilerplate — retrying without filler history)");
      turn = this.beginTurn();
      reply = this.generateTurn(turn, enrichment.modelUserText(), isolateGeneration);
      if (ChatPrompts.isSetupBoilerplate(reply.answer())) {
        reply = this.boilerplateFallback(enrichment);
      }
    }

    return this.finishTurn(reply, turn);
  }

  private ChatReply boilerplateFallback(final AdvisorEnrichment enrichment) {
    if (enrichment.hasGroundedNotes()) {
      String salvage = enrichment.groundedNotes().stream()
        .map(note -> note == null ? "" : note.strip())
        .filter(note -> !note.isEmpty())
        .collect(joining(" "));
      this.emitDiagnostics("(setup boilerplate — used grounded advisor notes as answer)");
      return new ChatReply("", salvage.strip(), false);
    }
    this.emitDiagnostics("(setup boilerplate — used plain greeting fallback)");
    return new ChatReply("", "Hello! What would you like to know?", false);
  }

  private TurnStream beginTurn() {
    if (this.printSink != null) {
      this.printSink.resetTurn();
    }
    return new TurnStream();
  }

  private void emitAdvisorNotes(final List<String> notes) {
    if (notes == null || notes.isEmpty()) {
      return;
    }
    for (int i = 0; i < notes.size(); i++) {
      String note = notes.get(i) == null ? "" : notes.get(i).strip();
      String body = note.isEmpty() ? "(no usable note)" : note;
      this.listener.onText(this.llm, LlmTextEvent.advisorNote(i + 1, body));
    }
  }

  private void emitDiagnostics(final String message) {
    this.listener.onText(this.llm, LlmTextEvent.of(LlmTextKind.TEXT_DIAGNOSTICS, message));
  }

  private ChatReply generateTurn(
    final TurnStream turn,
    final String lastUserOverride,
    final boolean isolateGeneration
  ) {
    Tokenizer tokenizer = this.llm.tokenizer();
    boolean gemmaChat = tokenizer.isGemmaChat();
    boolean enableThinking = this.thinkingEnabled(tokenizer);
    List<ChatMessage> forTemplate = isolateGeneration
      ? this.isolatedTurn(lastUserOverride)
      : this.historyForTemplate(lastUserOverride);
    String prompt = tokenizer.applyChatTemplate(
      ChatMessages.toTemplateMaps(forTemplate), true, enableThinking);

    List<Integer> streamedIds = new ArrayList<>();
    List<LLM.GenerationOutput> outputs = this.llm.generate(
      List.of(prompt),
      this.samplingParams,
      false,
      this.timeout,
      tokenId -> {
        streamedIds.add(tokenId);
        turn.push(this.llm, this.listener,
          AssistantParts.parse(tokenizer.decode(streamedIds, gemmaChat)));
      }
    );

    return ChatReply.from(
      AssistantParts.parse(tokenizer.decode(outputs.getFirst().tokenIds(), gemmaChat)));
  }

  private boolean thinkingEnabled(final Tokenizer tokenizer) {
    return this.enableThinking != null ? this.enableThinking : tokenizer.invitesThinking();
  }

  private List<ChatMessage> isolatedTurn(final String modelUserText) {
    List<ChatMessage> turn = new ArrayList<>(this.llm.newConversation());
    turn.add(ChatMessage.user(modelUserText));
    return turn;
  }

  private List<ChatMessage> historyForTemplate(final String lastUserOverride) {
    if (this.history.isEmpty()) {
      return List.of();
    }
    ChatMessage last = this.history.getLast();
    if (last.role() != ChatRole.USER || last.content().equals(lastUserOverride)) {
      return this.history;
    }
    List<ChatMessage> copy = new ArrayList<>(this.history);
    copy.set(copy.size() - 1, ChatMessage.user(lastUserOverride));
    return copy;
  }

  private ChatReply finishTurn(final ChatReply reply, final TurnStream turn) {
    String answer = reply.answer().strip();
    String thinking = reply.thinking();
    boolean thinkOpen = reply.thinkOpen();

    if (this.shouldSalvageAnswer(answer, thinking)) {
      answer = AssistantParts.salvageFromThinking(thinking);
      this.emitDiagnostics(thinkOpen
        ? "(reply recovered from unclosed thinking)"
        : "(reply recovered from thinking; model omitted or truncated visible answer)");
    }
    if (answer.isBlank()) {
      answer = "Sorry — I couldn't form a reply. Please try again.";
      this.emitDiagnostics("(empty reply — used fallback)");
    }

    ChatReply finished = new ChatReply(thinking, answer, false);
    turn.push(this.llm, this.listener,
      new AssistantParts(finished.thinking(), finished.answer(), false));
    if (this.printSink != null) {
      this.printSink.closeTurn();
    }
    this.history.add(ChatMessage.assistant(finished.answer()));
    return finished;
  }

  private boolean shouldSalvageAnswer(final String answer, final String thinking) {
    if (thinking.isBlank()) {
      return false;
    }
    if (answer.isBlank()) {
      return true;
    }
    return answer.length() <= 12 && thinking.length() >= 120;
  }

  private static final class TurnStream {
    private String shownThink = "";
    private String shownAnswer = "";

    void push(final LLM llm, final LlmListener listener, final AssistantParts parts) {
      this.emit(llm, listener, LlmTextKind.TEXT_THINKING, parts.thinking(), this.shownThink);
      this.shownThink = parts.thinking();
      if (!parts.thinkOpen()) {
        this.emit(llm, listener, LlmTextKind.TEXT_ASSISTANT, parts.answer(), this.shownAnswer);
        this.shownAnswer = parts.answer();
      }
    }

    private void emit(
      final LLM llm,
      final LlmListener listener,
      final LlmTextKind kind,
      final String current,
      final String shown
    ) {
      if (current.equals(shown)) {
        return;
      }
      if (current.length() > shown.length() && current.startsWith(shown)) {
        listener.onText(llm, LlmTextEvent.of(kind, current.substring(shown.length())));
        return;
      }
      listener.onText(llm, LlmTextEvent.snapshot(kind, current));
    }
  }
}
