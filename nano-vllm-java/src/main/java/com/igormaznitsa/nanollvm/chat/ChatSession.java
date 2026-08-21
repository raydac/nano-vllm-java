package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.AdvisorEnrichment;
import com.igormaznitsa.nanollvm.llm.AdvisorResponse;
import com.igormaznitsa.nanollvm.llm.GenerationStats;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Multi-turn chat over an {@link LLM}: history, chat template, truncation, reply parsing, and
 * {@link LlmListener} events.
 *
 * <p>Open with {@link LLM#chat()} (or {@link LLM#chat(int)} to cap reply length). Not thread-safe;
 * one session per conversation thread. {@link #send(String)} is the usual turn. Show
 * {@link ChatReply#answer()} to the user.
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
 *   <li><b>Fewer remembered turns</b> — {@link #maxHistoryMessages(int)} (process default:
 *       {@link ResourceLimits#maxHistoryMessages()})</li>
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
  private final List<ChatMessage> history;
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
    this.history = new ArrayList<>(llm.newConversation());
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
      this.history.add(message);
    }
    this.trimHistoryToCap();
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
   * @return finished {@link ChatReply}: {@code answer} for the user, optional {@code thinking},
   *         {@code thinkOpen == false}, and measured {@code stats}
   * @throws IllegalArgumentException                                          if {@code userText} is blank after strip
   * @throws NullPointerException                                              if {@code userText} is {@code null}
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
   * (RAG: short visible history, context-augmented model turn).
   *
   * @param historyUserText text recorded in {@link #history()}
   * @param modelUserText   text used in the chat template for this generate
   * @return finished {@link ChatReply}: {@code answer} for the user, optional {@code thinking},
   *         {@code thinkOpen == false}, and measured {@code stats}
   * @throws IllegalArgumentException                                          if either text is blank after strip
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException if {@link LLM#cancel()} fires
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException   if the session timeout elapses
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
   * @return finished {@link ChatReply}: {@code answer} for the user, optional {@code thinking},
   *         {@code thinkOpen == false}, and measured {@code stats}
   * @throws IllegalArgumentException                                          if either text is blank after strip
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

    this.history.add(ChatMessage.user(historyUser));
    this.trimHistoryToCap();
    Tokenizer tokenizer = this.llm.tokenizer();
    ChatMessages.truncateHistory(
      this.history,
      tokenizer,
      this.llm.config().maxModelLen(),
      this.samplingParams.maxTokens(),
      this.thinkingEnabled(tokenizer),
      this.thinkTags());

    TurnStream turn = this.beginTurn();

    long advisorStarted = System.nanoTime();
    AdvisorEnrichment enrichment =
      this.llm.runAdvisors(modelUser, this.priorDialogForAdvisors(), this.samplingParams);
    this.emitAdvisorNotes(enrichment.responses());
    this.emitPreparedUserDebug(enrichment);
    this.lastAdvisorNanos = System.nanoTime() - advisorStarted;

    this.lastGenerateStats = GenerationStats.NONE;
    ChatReply reply = this.generateTurn(turn, enrichment.modelUserText(), isolateGeneration);

    if (this.recoverUnusableAnswers && this.isUnusableMainAnswer(reply.answer())) {
      ChatMessages.scrubMatchingAssistantTurns(this.history, this.unusableAnswer);
      this.emitDiagnostics("(unusable main answer — retrying without filler history)");
      this.discardPrintedAnswer();
      turn = this.beginTurn();
      reply = this.generateTurn(turn, enrichment.modelUserText(), isolateGeneration);
      if (this.isUnusableMainAnswer(reply.answer())) {
        this.discardPrintedAnswer();
        reply = this.advisorSalvageFallback(enrichment);
      }
    }

    return this.finishTurn(reply, turn);
  }

  /**
   * User turns already in history, excluding the turn just appended (advisor context).
   */
  private List<ChatMessage> priorDialogForAdvisors() {
    List<ChatMessage> users = this.history.stream()
      .filter(message -> message.role() == ChatRole.USER)
      .toList();
    if (users.size() <= 1) {
      return List.of();
    }
    return users.subList(0, users.size() - 1);
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
   * {@code isolateGeneration} feeds only system seed + prepared user text.
   */
  private ChatReply generateTurn(
    final TurnStream turn,
    final String lastUserOverride,
    final boolean isolateGeneration
  ) {
    Tokenizer tokenizer = this.llm.tokenizer();
    boolean skipSpecials = tokenizer.skipSpecialTokensOnChatDecode();
    boolean enableThinking = this.thinkingEnabled(tokenizer);
    ThinkTags tags = this.thinkTags();
    List<ChatMessage> forTemplate = isolateGeneration
      ? this.isolatedTurn(lastUserOverride)
      : this.historyForTemplate(lastUserOverride);
    String prompt = tokenizer.applyChatTemplate(
      ChatMessages.toTemplateMaps(forTemplate),
      true,
      enableThinking,
      tags.open(),
      tags.close());

    List<Integer> streamedIds = new ArrayList<>();
    List<LLM.GenerationOutput> outputs = this.llm.generate(
      List.of(prompt),
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
   * Session {@link #enableThinking(boolean)} if set; otherwise vocab membership of the active tags.
   */
  private boolean thinkingEnabled(final Tokenizer tokenizer) {
    if (this.enableThinking != null) {
      return this.enableThinking;
    }
    ThinkTags tags = this.thinkTags();
    return tokenizer.invitesThinking(tags.open(), tags.close());
  }

  /**
   * System seed (if any) plus one user turn — prior dialog is omitted from the template.
   */
  private List<ChatMessage> isolatedTurn(final String modelUserText) {
    List<ChatMessage> turn = new ArrayList<>(this.llm.newConversation());
    turn.add(ChatMessage.user(modelUserText));
    return turn;
  }

  /**
   * History for the chat template, swapping the last user turn to {@code lastUserOverride} when it
   * differs from what was stored (RAG prepared text).
   */
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
    this.history.add(ChatMessage.assistant(finished.answer()));
    this.trimHistoryToCap();
    return finished;
  }

  /**
   * Drops oldest non-system turns until {@link #maxHistoryMessages} is honored.
   */
  private void trimHistoryToCap() {
    while (this.history.size() > this.maxHistoryMessages) {
      int dropAt = 0;
      if (!this.history.isEmpty() && this.history.getFirst().role() == ChatRole.SYSTEM) {
        if (this.history.size() == 1) {
          break;
        }
        dropAt = 1;
      }
      this.history.remove(dropAt);
    }
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
