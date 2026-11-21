package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.AdvisorEnrichment;
import com.igormaznitsa.nanollvm.llm.AdvisorResponse;
import com.igormaznitsa.nanollvm.llm.GenerationStats;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.llm.ToolCall;
import com.igormaznitsa.nanollvm.llm.ToolDialect;
import com.igormaznitsa.nanollvm.llm.ToolDialects;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Multi-turn chat over an {@link LLM}: history, chat template, truncation, reply parsing, and
 * {@link LlmListener} events.
 *
 * <p>Open with {@link LLM#chat()} (or {@link LLM#chat(int)} to cap reply length). Pass that
 * session back to {@link LLM#chat(ChatSession)} to keep using its transcript;
 * {@link LLM#chat()} with no session still starts a new one. Not thread-safe;
 * one session per conversation thread. Each rendered turn keeps its text and the token ids
 * of that text; a later {@link #send(String)} tokenizes only a turn whose text changed.
 * {@link #send(String)} is the usual turn. Show {@link ChatReply#answer()} to the user.
 *
 * <h2>If you want…</h2>
 * <ul>
 *   <li><b>The same wording every turn</b> — {@link #deterministic()} or
 *       {@link com.igormaznitsa.nanollvm.llm.LLM.Builder#deterministic()}</li>
 *   <li><b>Shorter replies</b> — {@link #maxTokens(int)} or {@link #sampling(SamplingParams)}</li>
 *   <li><b>A wall-clock limit per turn</b> — {@link #timeout(Duration)}</li>
 *   <li><b>Print thinking vs answer as they arrive</b> — {@link #streamTo} or {@link #listen}</li>
 *   <li><b>Skip the model's private scratchpad</b> — {@link #enableThinking(boolean) enableThinking(false)}
 *       (saves tokens; RAG sessions already default this off)</li>
 *   <li><b>Let the model call a function</b> — register {@link com.igormaznitsa.nanollvm.llm.ModelTool}
 *       values with {@link LLM.Builder#tool}. This session runs at most four calls per user
 *       message, then shows the reply that follows the last result.</li>
 *   <li><b>Fewer remembered turns</b> — {@link #maxHistoryMessages(int)} (process default:
 *       {@link ResourceLimits#maxHistoryMessages()})</li>
 *   <li><b>Leave some turns out of the transcript or out of the model prompt</b> —
 *       {@link #filter(ChatMessageFilter...)}; each filter returns a {@link ChatAdmission},
 *       and every filter must allow a destination</li>
 *   <li><b>Retry blank / garbage answers</b> — {@link #recoverUnusableAnswers(boolean)} (off by default)</li>
 *   <li><b>See the exact prompt sent to the model</b> — {@link #emitDebugPrompts(boolean)} (off by default)</li>
 * </ul>
 *
 * <p>Text and status events compose {@link LLM#listener()} with any session {@link #listen} /
 * {@link #streamTo} sink. Streaming emits {@link LlmTextKind#TEXT_RAW} (full decode, markers kept)
 * plus parsed {@link LlmTextKind#TEXT_THINKING} / {@link LlmTextKind#TEXT_ASSISTANT}.
 * {@link LlmTextKind#TEXT_DEBUG} stays off until {@link #emitDebugPrompts(boolean) emitDebugPrompts(true)}.
 */
public final class ChatSession {

  private final LLM llm;
  private final ChatTranscript transcript = new ChatTranscript();
  private SamplingParams samplingParams;
  private Duration timeout = Duration.ZERO;
  private LlmListener sessionListener = LlmListeners.silent();
  private LlmListener listener = LlmListeners.silent();
  private LlmListeners.PrintStreamLlmListener printSink;
  private Boolean enableThinking;
  private ThinkTags thinkTags;
  private long lastAdvisorNanos;
  private GenerationStats lastGenerateStats = GenerationStats.NONE;
  private int maxHistoryMessages = ResourceLimits.current().maxHistoryMessages();
  private boolean emitDebugPrompts;
  private boolean recoverUnusableAnswers;
  private Predicate<String> unusableAnswer = body -> body == null || body.isBlank();
  private String unusableAnswerFallback = "Sorry — I couldn't form a reply. Please try again.";
  private String unusableHistoryText = "";
  private boolean stitchKnown;
  private boolean blockStitch;
  private List<TokenizedText> tokenizedBlocks = List.of();

  /**
   * Opens a session with {@link LLM#defaultSampling()} and a fresh conversation seeded from the
   * model (system turn when the architecture uses one).
   *
   * @param llm engine that owns generation; must not be {@code null}
   * @throws NullPointerException if {@code llm} is {@code null}
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
   * @throws NullPointerException if {@code llm} or {@code samplingParams} is {@code null}
   */
  public ChatSession(final LLM llm, final SamplingParams samplingParams) {
    this.llm = requireNonNull(llm, "llm");
    this.samplingParams = requireNonNull(samplingParams, "samplingParams");
    this.transcript.addUnfiltered(llm.newConversation());
    this.bindListener();
  }

  /**
   * Factory that pins a max new-token budget via {@link LLM#defaultSampling(int)}.
   *
   * @param llm       engine that owns generation
   * @param maxTokens upper bound on new tokens per turn (other engine knobs kept)
   * @return a new session
   */
  public static ChatSession open(final LLM llm, final int maxTokens) {
    requireNonNull(llm, "llm");
    return new ChatSession(llm, llm.defaultSampling(maxTokens));
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
   * Greedy argmax for subsequent turns ({@code topK = 1}, nucleus off). Other knobs stay.
   *
   * @return {@code this} for fluent configuration
   * @see SamplingParams#asDeterministic()
   * @since 1.2.0
   */
  public ChatSession deterministic() {
    this.samplingParams = this.samplingParams.asDeterministic();
    return this;
  }

  /**
   * Sets max new tokens for subsequent turns; other sampling knobs stay.
   *
   * @param maxTokens upper bound on new tokens per turn
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public ChatSession maxTokens(final int maxTokens) {
    this.samplingParams = this.samplingParams.withMaxTokens(maxTokens);
    return this;
  }

  /**
   * Appends few-shot turns after the engine system seed, then trims to the history cap.
   * Registered {@link #filter}s admit each message. A message both destinations reject is not
   * stored.
   *
   * @param messages seed turns; must not be {@code null} (elements must be non-null and non-blank)
   * @return {@code this} for fluent configuration
   * @throws IllegalArgumentException if a message body is blank
   * @since 1.1.0
   */
  public ChatSession seed(final ChatMessage... messages) {
    requireNonNull(messages, "messages");
    return this.seed(List.of(messages));
  }

  /**
   * Appends few-shot turns after the engine system seed, then trims to the history cap.
   * Registered {@link #filter}s admit each message. A message both destinations reject is not
   * stored.
   *
   * @param messages seed turns; must not be {@code null} (elements must be non-null and non-blank)
   * @return {@code this} for fluent configuration
   * @throws IllegalArgumentException if a message body is blank
   * @since 1.1.0
   */
  public ChatSession seed(final List<ChatMessage> messages) {
    requireNonNull(messages, "messages");
    for (ChatMessage message : messages) {
      requireNonNull(message, "message");
      if (message.content().isBlank()) {
        throw new IllegalArgumentException("seed message content must not be blank");
      }
      this.transcript.add(message);
    }
    this.trimHistoryToCap();
    return this;
  }

  /**
   * Registers filters consulted when a message is added and again at the start of the next
   * {@link #send}. A message stays in {@link #history()} only when every filter keeps it, and
   * it is sent to the model only when every filter sends it. With no filter, both stay open.
   * Calling this again adds filters; it does not replace the ones already registered.
   *
   * <pre>{@code
   * ChatSession session = llm.chat().filter(
   *     message -> message.content().contains("secret")
   *         ? ChatAdmission.neither()
   *         : ChatAdmission.both(),
   *     message -> message.role() == ChatRole.SYSTEM
   *         ? ChatAdmission.historyOnly()
   *         : ChatAdmission.both());
   * session.send("What is 2+2?");
   * }</pre>
   *
   * @param filters one or more admissions; must not be {@code null} or empty, and no element
   *                may be {@code null}
   * @return {@code this} for fluent configuration
   * @throws NullPointerException     if {@code filters} or an element is {@code null}
   * @throws IllegalArgumentException if {@code filters} is empty
   * @since 1.5.0
   */
  public ChatSession filter(final ChatMessageFilter... filters) {
    requireNonNull(filters, "filters");
    if (filters.length == 0) {
      throw new IllegalArgumentException("filters must not be empty");
    }
    for (ChatMessageFilter filter : filters) {
      this.transcript.addFilter(requireNonNull(filter, "filter"));
    }
    return this;
  }

  /**
   * Engine that owns generation for this session.
   *
   * @return the {@link LLM} passed at construction
   */
  public LLM llm() {
    return this.llm;
  }

  /**
   * Sampling parameters currently used by {@link #send} / {@link #sendPrepared}.
   *
   * @return the live sampling knobs (immutable)
   */
  public SamplingParams samplingParams() {
    return this.samplingParams;
  }

  /**
   * Wall time of the advisor pass on the last {@link #send} / {@link #sendPrepared}
   * ({@code 0} when no advisors ran or before the first turn).
   *
   * @return advisor nanos for the last turn
   */
  public long lastAdvisorNanos() {
    return this.lastAdvisorNanos;
  }

  /**
   * Engine stats for the main assistant generate(s) on the last {@link #send} /
   * {@link #sendPrepared} (includes optional unusable-answer retries' elapsed time; excludes advisors).
   *
   * @return last main-generate stats, or {@link GenerationStats#NONE} before the first turn
   */
  public GenerationStats lastGenerateStats() {
    return this.lastGenerateStats;
  }

  /**
   * {@link GenerationStats#elapsedNanos()} of {@link #lastGenerateStats()}.
   *
   * @return last main-generate wall time in nanos
   */
  public long lastGenerateNanos() {
    return this.lastGenerateStats.elapsedNanos();
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
   * When {@code true}, invites the model to write a private scratchpad (often {@code <think>…})
   * before the visible answer. When {@code false}, ChatML templates may seed an empty open/close
   * pair from this session's {@link #thinkTags()} when {@link Tokenizer#invitesThinking(String, String)}
   * is true for those markers, so the model skips that scratchpad (important for RAG token budgets).
   * When never called, the default from vocab membership of those tags applies.
   *
   * @param enableThinking {@code true} to invite the scratchpad, {@code false} to suppress it
   * @return {@code this} for fluent configuration
   */
  public ChatSession enableThinking(final boolean enableThinking) {
    this.enableThinking = enableThinking;
    return this;
  }

  /**
   * Scratchpad open/close markers for parse and ChatML skip-seed. Defaults to
   * {@link LLM#thinkTags()} from the shared {@link com.igormaznitsa.nanollvm.models.LlmModel}. Set
   * this only to override the model pair for one conversation.
   *
   * @param thinkTags must not be {@code null}
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public ChatSession thinkTags(final ThinkTags thinkTags) {
    this.thinkTags = requireNonNull(thinkTags, "thinkTags");
    return this;
  }

  /**
   * Markers currently used by {@link #send} / streaming parse ({@link LLM#thinkTags()} unless
   * {@link #thinkTags(ThinkTags)} was called).
   *
   * @return session override, or the model's pair
   * @since 1.1.0
   */
  public ThinkTags thinkTags() {
    return this.thinkTags != null ? this.thinkTags : this.llm.thinkTags();
  }

  /**
   * When {@code true}, retries once (scrubbing matching assistant turns) and may salvage from
   * advisor notes if the main answer matches {@link #unusableAnswer(Predicate)}. Off by default —
   * enable from demos/apps that need it for small turn-based models.
   *
   * @param enable {@code true} to retry / salvage unusable answers
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public ChatSession recoverUnusableAnswers(final boolean enable) {
    this.recoverUnusableAnswers = enable;
    return this;
  }

  /**
   * Predicate for answers treated as unusable when {@link #recoverUnusableAnswers(boolean)} is on.
   * Default: blank only.
   *
   * @param predicate test on stripped answer text; must not be {@code null}
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public ChatSession unusableAnswer(final Predicate<String> predicate) {
    this.unusableAnswer = requireNonNull(predicate, "predicate");
    return this;
  }

  /**
   * Fallback visible reply when recovery still yields nothing usable.
   *
   * @param fallback non-blank user-facing string
   * @return {@code this} for fluent configuration
   * @throws IllegalArgumentException if {@code fallback} is blank after strip
   * @since 1.1.0
   */
  public ChatSession unusableAnswerFallback(final String fallback) {
    this.unusableAnswerFallback = requireNonNull(fallback, "fallback").strip();
    if (this.unusableAnswerFallback.isEmpty()) {
      throw new IllegalArgumentException("fallback must not be blank");
    }
    return this;
  }

  /**
   * Text written over an unusable assistant turn before the retry. Empty by default, so the
   * library does not insert a greeting. Set this when the checkpoint needs some assistant text
   * in history.
   *
   * @param text replacement; must not be {@code null}
   * @return {@code this}
   */
  public ChatSession unusableHistoryText(final String text) {
    this.unusableHistoryText = requireNonNull(text, "text");
    return this;
  }

  /**
   * Caps retained dialog turns (system + user + assistant). Oldest non-system messages are dropped
   * when the cap is exceeded. Default from {@link ResourceLimits#maxHistoryMessages()}.
   *
   * @param maxHistoryMessages must be {@code >= 1}
   * @return {@code this} for fluent configuration
   * @throws IllegalArgumentException if {@code maxHistoryMessages < 1}
   */
  public ChatSession maxHistoryMessages(final int maxHistoryMessages) {
    if (maxHistoryMessages < 1) {
      throw new IllegalArgumentException("maxHistoryMessages must be >= 1");
    }
    this.maxHistoryMessages = maxHistoryMessages;
    this.trimHistoryToCap();
    return this;
  }

  /**
   * When {@code true}, emits {@link LlmTextKind#TEXT_DEBUG} with the prepared model-user text after
   * advisors. Off by default so full prompts are not sent to listeners; turn on only when debugging.
   *
   * @param emitDebugPrompts {@code true} to emit prepared-prompt debug events
   * @return {@code this} for fluent configuration
   */
  public ChatSession emitDebugPrompts(final boolean emitDebugPrompts) {
    this.emitDebugPrompts = emitDebugPrompts;
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

  /**
   * Composes {@link LLM#listener()} with the session sink and unwraps a CLI print adapter if present.
   */
  private void bindListener() {
    this.listener = LlmListeners.compose(this.llm.listener(), this.sessionListener);
    this.printSink = LlmListeners.unwrapPrintStream(this.sessionListener);
  }

  /**
   * Messages the filters still keep in the transcript, in conversation order. A message admitted
   * as {@link ChatAdmission#modelOnly()} is absent here and can still be sent to the model.
   * Messages already stored are reconsidered at the start of the next {@link #send}.
   *
   * @return an unmodifiable copy; mutations require {@link #send} / {@link #clear}
   */
  public List<ChatMessage> history() {
    return this.transcript.history();
  }

  /**
   * Drops the transcript and reseeds the engine system prompt. Registered {@link #filter}s admit
   * that seed. Sampling, timeout, listeners, and filters stay.
   */
  public void clear() {
    this.transcript.clear();
    for (ChatMessage message : this.llm.newConversation()) {
      this.transcript.add(message);
    }
    this.tokenizedBlocks = List.of();
  }

  /**
   * Appends {@code userText} to history, runs advisors when configured, generates a reply, and
   * appends the assistant answer to history. Registered {@link #filter}s admit the user phrase,
   * the assistant answer, and messages already stored. The visible {@link ChatReply#answer()} is
   * still returned when the assistant message itself is left out of the transcript.
   *
   * @param userText non-blank user turn
   * @return finished {@link ChatReply}: {@code answer} for the user, optional {@code thinking},
   * {@code thinkOpen == false}, and measured {@code stats}
   * @throws IllegalArgumentException                                          if {@code userText} is blank after strip
   * @throws NullPointerException                                              if {@code userText} is {@code null}
   * @throws IllegalStateException                                             if filters leave the model prompt empty
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException if {@link LLM#cancel()} fires
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException   if the session timeout elapses
   */
  public ChatReply send(final String userText) {
    requireNonNull(userText, "userText");
    return this.sendPrepared(userText, userText);
  }

  /**
   * One turn with explicit sampling; session sampling is restored afterward.
   *
   * @param userText       non-blank user turn
   * @param samplingParams knobs for this turn only
   * @return finished {@link ChatReply}
   * @since 1.1.0
   */
  public ChatReply send(final String userText, final SamplingParams samplingParams) {
    requireNonNull(samplingParams, "samplingParams");
    SamplingParams previous = this.samplingParams;
    this.samplingParams = samplingParams;
    try {
      return this.send(userText);
    } finally {
      this.samplingParams = previous;
    }
  }

  /**
   * Stores {@code historyUserText} in history while generating from {@code modelUserText}
   * (RAG: short visible history, context-augmented model turn). {@link #filter}s admit the stored
   * text. When they also send it to the model, the template uses {@code modelUserText}.
   *
   * @param historyUserText text recorded in {@link #history()}
   * @param modelUserText   text used in the chat template for this generate
   * @return finished {@link ChatReply}: {@code answer} for the user, optional {@code thinking},
   * {@code thinkOpen == false}, and measured {@code stats}
   * @throws IllegalArgumentException                                          if either text is blank after strip
   * @throws IllegalStateException                                             if filters leave the model prompt empty
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException if {@link LLM#cancel()} fires
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException   if the session timeout elapses
   */
  public ChatReply sendPrepared(final String historyUserText, final String modelUserText) {
    return this.sendPrepared(historyUserText, modelUserText, false);
  }

  /**
   * Stores one user text in history but may generate from a different prepared user string.
   * {@link #filter}s admit that stored text. When they also send it to the model, the template
   * uses {@code modelUserText} for that one turn.
   *
   * @param historyUserText   text recorded in {@link #history()}; must not be blank
   * @param modelUserText     text used for advisors and the chat template; must not be blank
   * @param isolateGeneration when {@code true}, the model sees only the system seed (if any) plus
   *                          the prepared user turn — prior assistant answers stay in history for
   *                          the app but are not fed into this generate (avoids tiny-model latch)
   * @return finished {@link ChatReply}: {@code answer} for the user, optional {@code thinking},
   * {@code thinkOpen == false}, and measured {@code stats}
   * @throws IllegalArgumentException                                          if either text is blank after strip
   * @throws IllegalStateException                                             if filters leave the model prompt empty
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException if {@link LLM#cancel()} fires
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException   if the session timeout elapses
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

    if (this.transcript.reapply()) {
      this.tokenizedBlocks = List.of();
    }
    ChatTranscript.Entry current = this.transcript.add(ChatMessage.user(historyUser));
    this.trimHistoryToCap();
    Tokenizer tokenizer = this.llm.tokenizer();
    this.truncateToTokenBudget(
        this.thinkingEnabled(tokenizer), this.thinkTags(), current, modelUser);

    TurnStream turn = this.beginTurn();

    long advisorStarted = System.nanoTime();
    AdvisorEnrichment enrichment =
        this.llm.runAdvisors(modelUser, this.priorDialogForAdvisors(current), this.samplingParams);
    this.emitAdvisorNotes(enrichment.responses());
    this.emitPreparedUserDebug(enrichment);
    this.lastAdvisorNanos = System.nanoTime() - advisorStarted;

    this.lastGenerateStats = GenerationStats.NONE;
    List<ChatMessage> prompt = this.promptForTurn(
        current, enrichment.modelUserText(), isolateGeneration);
    ChatReply reply = this.llm.tools().isEmpty()
        ? this.generateTurn(turn, prompt)
        : this.generateWithTools(turn, prompt);

    if (this.llm.tools().isEmpty()
      && this.recoverUnusableAnswers
      && this.isUnusableMainAnswer(reply.answer())) {
      this.transcript.scrub(this.unusableAnswer, this.unusableHistoryText);
      this.emitDiagnostics("(unusable main answer — retrying without filler history)");
      this.discardPrintedAnswer();
      turn = this.beginTurn();
      prompt = this.promptForTurn(current, enrichment.modelUserText(), isolateGeneration);
      reply = this.generateTurn(turn, prompt);
      if (this.isUnusableMainAnswer(reply.answer())) {
        this.discardPrintedAnswer();
        reply = this.advisorSalvageFallback(enrichment);
      }
    }

    return this.finishTurn(reply, turn);
  }

  /**
   * Model prompt for this turn. Isolated turns keep the engine system seed (when the model flag
   * allows it) and the current user text. Otherwise every stored message whose model flag is open
   * is included, with {@code modelText} in place of the current user message.
   */
  private List<ChatMessage> promptForTurn(
      final ChatTranscript.Entry current,
      final String modelText,
      final boolean isolateGeneration
  ) {
    List<ChatMessage> prompt = isolateGeneration
        ? this.isolatedTurn(modelText, current)
        : this.transcript.modelMessages(current, modelText);
    if (prompt.isEmpty()) {
      throw new IllegalStateException("chat filters left no message for the model");
    }
    return prompt;
  }

  /**
   * User turns already in history, excluding the turn just appended when it was stored.
   */
  private List<ChatMessage> priorDialogForAdvisors(final ChatTranscript.Entry current) {
    List<ChatMessage> users = this.transcript.history().stream()
      .filter(message -> message.role() == ChatRole.USER)
      .toList();
    if (current != null && current.inHistory() && this.transcript.contains(current)) {
      return users.isEmpty() ? List.of() : users.subList(0, users.size() - 1);
    }
    return users;
  }

  /**
   * {@code true} when {@link #unusableAnswer} matches, or the answer is an advisor name (tiny-model latch).
   */
  private boolean isUnusableMainAnswer(final String answer) {
    String body = answer == null ? "" : answer.strip();
    if (this.unusableAnswer.test(body)) {
      return true;
    }
    return this.llm.advisors().stream()
      .map(advisor -> advisor.name().strip())
      .anyMatch(name -> name.equalsIgnoreCase(body));
  }

  /**
   * Last-resort reply: joined advisor salvage notes, else {@link #unusableAnswerFallback}.
   */
  private ChatReply advisorSalvageFallback(final AdvisorEnrichment enrichment) {
    String salvage = String.join(" ", enrichment.answerSalvageNotes());
    if (!salvage.isBlank()) {
      this.emitDiagnostics("(unusable main answer — used advisor notes as answer)");
      return new ChatReply("", salvage.strip(), false, this.lastGenerateStats);
    }
    this.emitDiagnostics("(unusable main answer — used plain reply fallback)");
    return new ChatReply("", this.unusableAnswerFallback, false, this.lastGenerateStats);
  }

  /**
   * Resets the CLI printer for a new generate (or retry) and returns a fresh stream accumulator.
   */
  private TurnStream beginTurn() {
    if (this.printSink != null) {
      this.printSink.resetTurn();
    }
    return new TurnStream();
  }

  /**
   * Clears a partially printed {@code assistant>} line so a retry does not leave two answers.
   */
  private void discardPrintedAnswer() {
    if (this.printSink != null) {
      this.printSink.discardAnswer();
    }
  }

  /**
   * Emits {@link LlmTextKind#TEXT_ADVISOR_NOTE} for each advisor response.
   */
  private void emitAdvisorNotes(final List<AdvisorResponse> responses) {
    if (responses == null || responses.isEmpty()) {
      return;
    }
    for (AdvisorResponse response : responses) {
      this.listener.onText(
        this.llm, LlmTextEvent.advisorNote(response.advisorName(), response.text()));
    }
  }

  /**
   * Emits {@link LlmTextKind#TEXT_DEBUG} with the prepared model-user string when enabled.
   */
  private void emitPreparedUserDebug(final AdvisorEnrichment enrichment) {
    if (!this.emitDebugPrompts) {
      return;
    }
    this.listener.onText(this.llm, LlmTextEvent.debug(enrichment.modelUserText()));
  }

  /**
   * Emits {@link LlmTextKind#TEXT_DIAGNOSTICS} (salvage / retry / empty-reply notices).
   */
  private void emitDiagnostics(final String message) {
    this.listener.onText(this.llm, LlmTextEvent.of(LlmTextKind.TEXT_DIAGNOSTICS, message));
  }

  /**
   * One {@link LLM#generate}: chat template, token stream parse, measured stats.
   */
  private ChatReply generateTurn(final TurnStream turn, final List<ChatMessage> forTemplate) {
    Tokenizer tokenizer = this.llm.tokenizer();
    boolean skipSpecials = tokenizer.skipSpecialTokensOnChatDecode();
    boolean enableThinking = this.thinkingEnabled(tokenizer);
    ThinkTags tags = this.thinkTags();
    List<Integer> promptIds = this.promptTokenIds(forTemplate, enableThinking, tags);

    List<Integer> streamedIds = new ArrayList<>();
    List<LLM.GenerationOutput> outputs = this.llm.generateTokenIds(
      List.of(promptIds),
      this.samplingParams,
      false,
      this.timeout,
      tokenId -> {
        streamedIds.add(tokenId);
        String raw = tokenizer.decode(streamedIds, false);
        String forParse = skipSpecials ? tokenizer.decode(streamedIds, true) : raw;
        turn.push(
          this.llm,
          this.listener,
          raw,
          ChatReply.parse(forParse, tags, this.llm.chatSpecials()));
      }
    );

    LLM.GenerationOutput output = outputs.getFirst();
    this.lastGenerateStats = this.mergeGenerateStats(this.lastGenerateStats, output.stats());
    return ChatReply.parse(
        tokenizer.decode(output.tokenIds(), skipSpecials),
        tags,
        this.llm.chatSpecials())
      .withStats(this.lastGenerateStats);
  }

  /**
   * One user turn that may call tools. Each execution is a fresh generate. The session history
   * still stores only the final visible answer; tool markup stays in the working copy.
   */
  private ChatReply generateWithTools(final TurnStream turn, final List<ChatMessage> forTemplate) {
    Tokenizer tokenizer = this.llm.tokenizer();
    ThinkTags tags = this.thinkTags();
    boolean enableThinking = this.thinkingEnabled(tokenizer);
    ToolDialect dialect = ToolDialects.of(this.llm.model().architectureName());
    List<ChatMessage> working = new ArrayList<>(forTemplate);

    String raw;
    int executed = 0;
    while (true) {
      String prompt = dialect.render(
        working, this.llm.tools(), tokenizer, enableThinking, tags);
      raw = this.generateRaw(turn, prompt, dialect, tags);
      List<ToolCall> calls = dialect.parseCalls(raw);
      if (calls.isEmpty() || executed == this.llm.maxToolExecutions()) {
        break;
      }
      working.addAll(
        dialect.appendExchange(raw, calls, ToolDialects.execute(this.llm.tools(), calls)));
      executed++;
    }
    return ChatReply.parse(dialect.visibleText(raw), tags, this.llm.chatSpecials())
      .withStats(this.lastGenerateStats);
  }

  /**
   * One generate whose raw decode keeps special tokens so a tool dialect can see them.
   */
  private String generateRaw(
    final TurnStream turn,
    final String prompt,
    final ToolDialect dialect,
    final ThinkTags tags
  ) {
    Tokenizer tokenizer = this.llm.tokenizer();
    List<Integer> streamedIds = new ArrayList<>();
    List<LLM.GenerationOutput> outputs = this.llm.generate(
      List.of(prompt),
      this.samplingParams,
      false,
      this.timeout,
      tokenId -> {
        streamedIds.add(tokenId);
        String raw = tokenizer.decode(streamedIds, false);
        turn.push(
          this.llm,
          this.listener,
          raw,
          ChatReply.parse(dialect.visibleText(raw), tags, this.llm.chatSpecials()));
      }
    );
    LLM.GenerationOutput output = outputs.getFirst();
    this.lastGenerateStats = this.mergeGenerateStats(this.lastGenerateStats, output.stats());
    return tokenizer.decode(output.tokenIds(), false);
  }

  /**
   * Adds {@code next.elapsedNanos()} onto {@code prior} when a retry ran a second generate.
   */
  private GenerationStats mergeGenerateStats(
    final GenerationStats prior,
    final GenerationStats next
  ) {
    if (prior.equals(GenerationStats.NONE)) {
      return next;
    }
    return new GenerationStats(
      next.promptTokens(),
      next.completionTokens(),
      prior.elapsedNanos() + next.elapsedNanos());
  }

  /**
   * Session {@link #enableThinking(boolean)} if set; otherwise the tokenizer's scratchpad flag,
   * then vocab membership of the active tags.
   */
  private boolean thinkingEnabled(final Tokenizer tokenizer) {
    if (this.enableThinking != null) {
      return this.enableThinking;
    }
    if (tokenizer.invitesThinking()) {
      return true;
    }
    ThinkTags tags = this.thinkTags();
    return tokenizer.invitesThinking(tags.open(), tags.close());
  }

  /**
   * Engine system seed, when the model flag allows it, plus the current user text when that
   * message is still stored and its model flag is open.
   */
  private List<ChatMessage> isolatedTurn(
      final String modelUserText,
      final ChatTranscript.Entry current
  ) {
    List<ChatMessage> turn = new ArrayList<>();
    for (ChatMessage message : this.llm.newConversation()) {
      if (this.transcript.admit(message).sendToModel()) {
        turn.add(message);
      }
    }
    if (current != null
        && this.transcript.contains(current)
        && this.transcript.admit(current.message()).sendToModel()) {
      turn.add(ChatMessage.user(modelUserText));
    }
    return turn;
  }

  /**
   * Salvages / fallbacks a blank or truncated answer, pushes a final stream snapshot, records
   * the assistant turn in history.
   */
  private ChatReply finishTurn(final ChatReply reply, final TurnStream turn) {
    String answer = reply.answer().strip();
    String thinking = reply.thinking();
    boolean thinkOpen = reply.thinkOpen();

    if (this.shouldSalvageAnswer(answer, thinking)) {
      answer = ChatReply.salvageFromThinking(thinking);
      this.emitDiagnostics(thinkOpen
        ? "(reply recovered from unclosed thinking)"
        : "(reply recovered from thinking; model omitted or truncated visible answer)");
    }
    if (answer.isBlank()) {
      answer = this.unusableAnswerFallback;
      this.emitDiagnostics("(empty reply — used fallback)");
    }

    ChatReply finished = new ChatReply(thinking, answer, false, this.lastGenerateStats);
    turn.push(this.llm, this.listener, turn.shownRaw, finished);
    this.closePrintTurn();
    this.transcript.add(ChatMessage.assistant(finished.answer()));
    this.trimHistoryToCap();
    return finished;
  }

  /**
   * Drops oldest turns until the rendered prompt fits the token budget. Unchanged blocks reuse
   * token ids already stored for that exact text.
   */
  private void truncateToTokenBudget(
      final boolean enableThinking,
      final ThinkTags tags,
      final ChatTranscript.Entry current,
      final String modelText
  ) {
    int budget = ChatMessages.promptTokenBudget(
      this.llm.config().maxModelLen(), this.samplingParams.maxTokens());
    int minKeep = this.transcript.firstRole() == ChatRole.SYSTEM ? 2 : 1;
    while (this.transcript.size() > minKeep
        && this.promptTokenIds(
        this.transcript.modelMessages(current, modelText), enableThinking, tags).size() > budget) {
      int sizeBeforeDrop = this.transcript.size();
      this.transcript.dropOldest();
      if (this.transcript.size() == sizeBeforeDrop) {
        return;
      }
    }
  }

  /**
   * Token ids of the chat template. A block whose text is already stored is not encoded again.
   * Blocks are stitched only when encoding them separately matches one encode of the joined prompt.
   */
  private List<Integer> promptTokenIds(final List<ChatMessage> messages,
                                       final boolean enableThinking,
                                       final ThinkTags tags) {
    List<String> blocks = this.renderBlocks(messages, enableThinking, tags);
    if (!this.blockTokensReusable(this.llm.tokenizer(), enableThinking, tags)) {
      return this.llm.tokenizer().encode(String.join("", blocks));
    }
    List<TokenizedText> used = new ArrayList<>(blocks.size());
    List<Integer> ids = new ArrayList<>();
    for (String block : blocks) {
      int[] tokens = this.cachedTokens(block).orElseGet(() -> this.encodeBlock(block));
      used.add(new TokenizedText(block, tokens));
      for (int id : tokens) {
        ids.add(id);
      }
    }
    this.rememberBlocks(used, enableThinking, tags);
    return ids;
  }

  private boolean blockTokensReusable(final Tokenizer tokenizer,
                                      final boolean enableThinking,
                                      final ThinkTags tags) {
    if (this.stitchKnown) {
      return this.blockStitch;
    }
    List<String> probe = tokenizer.chatBlocks(
      List.of(
        Map.of("role", "user", "content", "a b"),
        Map.of("role", "assistant", "content", "c d")),
      true,
      enableThinking,
      tags.open(),
      tags.close());
    List<Integer> joined = tokenizer.encode(String.join("", probe));
    List<Integer> stitched = new ArrayList<>();
    for (String block : probe) {
      stitched.addAll(tokenizer.encode(block));
    }
    this.blockStitch = joined.equals(stitched);
    this.stitchKnown = true;
    return this.blockStitch;
  }

  private List<String> renderBlocks(final List<ChatMessage> messages,
                                    final boolean enableThinking,
                                    final ThinkTags tags) {
    return this.llm.tokenizer().chatBlocks(
      ChatMessages.toTemplateMaps(messages),
      true,
      enableThinking,
      tags.open(),
      tags.close());
  }

  private Optional<int[]> cachedTokens(final String block) {
    return this.tokenizedBlocks.stream()
      .filter(cached -> cached.text().equals(block))
      .map(TokenizedText::tokenIds)
      .findFirst();
  }

  private int[] encodeBlock(final String block) {
    List<Integer> ids = this.llm.tokenizer().encode(block);
    int[] tokens = new int[ids.size()];
    for (int i = 0; i < ids.size(); i++) {
      tokens[i] = ids.get(i);
    }
    return tokens;
  }

  /**
   * Keeps token ids for the blocks just used. A prompt that is not the stored history also keeps
   * ids for the history blocks, so one isolated turn does not force the next turn to re-encode.
   */
  private void rememberBlocks(final List<TokenizedText> used,
                              final boolean enableThinking,
                              final ThinkTags tags) {
    List<String> historyBlocks = this.renderBlocks(this.transcript.history(), enableThinking, tags);
    List<TokenizedText> kept = new ArrayList<>(used);
    for (TokenizedText cached : this.tokenizedBlocks) {
      if (this.containsText(used, cached.text()) || !historyBlocks.contains(cached.text())) {
        continue;
      }
      kept.add(cached);
    }
    this.tokenizedBlocks = List.copyOf(kept);
  }

  private boolean containsText(final List<TokenizedText> items, final String text) {
    return items.stream().anyMatch(item -> item.text().equals(text));
  }

  /**
   * Drops oldest non-system turns until {@link #maxHistoryMessages} is honored.
   */
  private void trimHistoryToCap() {
    this.transcript.trimHistory(this.maxHistoryMessages);
  }

  /**
   * Ends the CLI think/answer lines for this turn when a print sink is installed.
   */
  private void closePrintTurn() {
    if (this.printSink != null) {
      this.printSink.closeTurn();
    }
  }

  /**
   * {@code true} when thinking is non-blank and the answer is empty or a short stub vs a long
   * scratchpad (model closed the tag but put the real reply in notes).
   */
  private boolean shouldSalvageAnswer(final String answer, final String thinking) {
    if (thinking.isBlank()) {
      return false;
    }
    if (answer.isBlank()) {
      return true;
    }
    return answer.length() <= 12 && thinking.length() >= 120;
  }

  /**
   * One rendered prompt piece: the exact template text and the token ids of that text.
   */
  private record TokenizedText(String text, int[] tokenIds) {
    private TokenizedText {
      tokenIds = tokenIds.clone();
    }
  }

  /**
   * Per-turn stream state: last shown raw / think / answer so listener events are deltas (or
   * snapshots when the parse revises an earlier prefix).
   */
  private static final class TurnStream {
    private String shownRaw = "";
    private String shownThink = "";
    private String shownAnswer = "";

    /**
     * Emits raw / thinking / answer deltas (or snapshots) for this prefix.
     */
    void push(
      final LLM llm,
      final LlmListener listener,
      final String raw,
      final ChatReply parts
    ) {
      String decoded = raw == null ? "" : raw;
      this.emit(llm, listener, LlmTextKind.TEXT_RAW, decoded, this.shownRaw);
      this.shownRaw = decoded;
      this.emit(llm, listener, LlmTextKind.TEXT_THINKING, parts.thinking(), this.shownThink);
      this.shownThink = parts.thinking();
      if (!parts.thinkOpen()) {
        this.emit(llm, listener, LlmTextKind.TEXT_ASSISTANT, parts.answer(), this.shownAnswer);
        this.shownAnswer = parts.answer();
      }
    }

    /**
     * Suffix delta when {@code current} extends {@code shown}; otherwise a full snapshot.
     */
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
