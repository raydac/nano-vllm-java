package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Retrieval-augmented chat over an {@link LLM}: {@link RagIndex} → prompt → generate.
 *
 * <p>Open with {@link LLM#rag(RagIndex)} after {@link RagFactory#make}. {@link #ask(String)}
 * returns the visible answer; {@link #send(String)} returns a full {@link ChatReply}.
 *
 * <h2>If you want…</h2>
 * <ul>
 *   <li><b>More / fewer passages in the prompt</b> — {@link #topK(int)} and
 *       {@link #maxContextChars(int)} (character cap on concatenated hits; not chunk size —
 *       that is {@link RagLoadOptions} at index load)</li>
 *   <li><b>Less random grounded answers</b> — {@link #sampling(SamplingParams)} with a low
 *       temperature; hits already clamp temperature to {@value #GROUNDED_TEMPERATURE_CAP} when hotter</li>
 *   <li><b>Ignore earlier assistant replies when answering from hits</b> —
 *       {@link #isolateGeneration(boolean)} (off by default)</li>
 *   <li><b>Same timeout / stream / thinking knobs as chat</b> — {@link #timeout}, {@link #streamTo},
 *       {@link #enableThinking} (thinking is already off here by default)</li>
 * </ul>
 *
 * <p>History stores the original user text; the model sees a context-augmented last turn.
 * Short follow-ups with a prior turn are rewritten by an isolated LLM call into standalone
 * search keywords. Rewrite is skipped when the follow-up alone already retrieves hits; when the
 * model returns {@code NONE} or empty, retrieval falls back to Prior+follow-up keywords instead
 * of aborting. Off-topic follow-ups still skip retrieval via {@link RagIndex#isOutsideCorpus}.
 * Short first turns and longer standalone questions use the raw text for BM25. Without an LLM,
 * retrieval falls back to concatenating the previous longer user turn. {@link PreparedRag}
 * re-ranks hits by term coverage and passage length at retrieve time.
 * {@link #isolateGeneration(boolean)} omits prior assistant answers from grounded (hit) generates
 * when on (default {@code false}; demos may enable for small turn-based models). No-hit turns always isolate so prior corpus answers cannot latch.
 * Thinking is off by default so small max-token budgets are not spent on {@code <think>} blocks.
 * Grounded turns clamp sampling temperature to {@value #GROUNDED_TEMPERATURE_CAP} when the caller
 * set a hotter value. Not thread-safe.
 *
 * <p>{@link #open(ChatSession, RagIndex)} reuses that session's {@link LLM} (rewrite stays enabled).
 *
 * <pre>{@code
 * PreparedRag rag = RagFactory.make(Path.of("docs"));
 * ChatReply reply = llm.rag(rag).topK(2).send("What is the capital of France?");
 * }</pre>
 */
public final class RagSession {

  private static final float GROUNDED_TEMPERATURE_CAP = 0.15f;

  private final LLM llm;
  private final ChatSession chat;
  private final RagIndex index;
  private int topK = 4;
  private int maxContextChars = 3500;
  private boolean isolateGeneration;
  private SamplingParams baseSampling;
  private int maxTokensWhenNoHits = 384;
  private List<RagHit> lastHits = List.of();
  private String anchorQuery = "";
  private String lastSource = "";
  private String lastRetrievalQuery = "";

  private RagSession(final LLM llm, final ChatSession chat, final RagIndex index) {
    this.llm = requireNonNull(llm, "llm");
    this.chat = requireNonNull(chat, "chat");
    this.index = requireNonNull(index, "index");
    this.isolateGeneration = false;
    this.chat.enableThinking(false);
  }

  /**
   * Opens a session using {@link LLM#defaultSampling()} and a fresh inner {@link ChatSession}.
   *
   * @param llm   engine that owns generation and optional query rewrite; must not be {@code null}
   * @param index corpus index; must not be {@code null}; may be shared across sessions
   * @return a new session; not thread-safe
   */
  public static RagSession open(final LLM llm, final RagIndex index) {
    requireNonNull(llm, "llm");
    return new RagSession(llm, new ChatSession(llm), index);
  }

  /**
   * Opens a session with a max new-token budget via {@link ChatSession#open(LLM, int)}.
   *
   * @param llm       engine that owns generation and optional query rewrite
   * @param index     corpus index; must not be {@code null}
   * @param maxTokens upper bound on new tokens per grounded turn (other engine knobs kept)
   * @return a new session; not thread-safe
   */
  public static RagSession open(final LLM llm, final RagIndex index, final int maxTokens) {
    requireNonNull(llm, "llm");
    return new RagSession(llm, ChatSession.open(llm, maxTokens), index);
  }

  /**
   * Reuses {@code chat}'s {@link LLM} and conversation (rewrite stays enabled). Sampling, listeners,
   * history, and recovery knobs on {@code chat} remain in effect.
   *
   * @param chat  existing chat session; must not be {@code null}
   * @param index corpus index; must not be {@code null}
   * @return a new RAG wrapper around {@code chat}
   */
  public static RagSession open(final ChatSession chat, final RagIndex index) {
    requireNonNull(chat, "chat");
    return new RagSession(chat.llm(), chat, index);
  }

  /**
   * Builds the model-facing user turn from retrieved passages plus {@code question}. Empty hits
   * produce the no-context RAG prompt.
   *
   * @param hits            retrieved passages, highest score first; must not be {@code null}
   * @param question        user question; must not be blank after strip
   * @param maxContextChars cap on concatenated passage text; must be {@code >= 64}
   * @return prompt text for the chat template (history still stores the original question)
   * @throws IllegalArgumentException if {@code question} is blank after strip or
   *                                  {@code maxContextChars < 64}
   */
  public static String formatUserMessage(
    final List<RagHit> hits,
    final String question,
    final int maxContextChars
  ) {
    return UserMessage.format(hits, question, maxContextChars);
  }

  /**
   * {@link #formatUserMessage(List, String, int)} with no passage-length cap
   * ({@link Integer#MAX_VALUE}).
   *
   * @param hits     retrieved passages, highest score first; must not be {@code null}
   * @param question user question; must not be blank after strip
   * @return prompt text for the chat template
   */
  public static String formatUserMessage(final List<RagHit> hits, final String question) {
    return UserMessage.format(hits, question, Integer.MAX_VALUE);
  }

  /**
   * Number of passages kept after retrieve (and prior-source preference). Default {@code 4}.
   *
   * @param topK must be {@code > 0}
   * @return {@code this} for fluent configuration
   * @throws IllegalArgumentException if {@code topK <= 0}
   */
  public RagSession topK(final int topK) {
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    this.topK = topK;
    return this;
  }

  /**
   * Cap on concatenated passage text in the model-facing user turn. Default {@code 3500}.
   * Independent of the per-chunk ceiling on {@link RagLoadOptions#maxChunkChars()} (set that at
   * index load).
   *
   * @param maxContextChars must be {@code >= 64}
   * @return {@code this} for fluent configuration
   * @throws IllegalArgumentException if {@code maxContextChars < 64}
   */
  public RagSession maxContextChars(final int maxContextChars) {
    if (maxContextChars < 64) {
      throw new IllegalArgumentException("maxContextChars must be >= 64");
    }
    this.maxContextChars = maxContextChars;
    return this;
  }

  /**
   * RAG defaults to thinking off so the token budget goes to the grounded answer. Re-enable when
   * the tokenizer {@link com.igormaznitsa.nanollvm.tokenizer.Tokenizer#invitesThinking()} and the
   * budget allow.
   *
   * @param enableThinking {@code true} to invite chain-of-thought, {@code false} to suppress it
   * @return {@code this} for fluent configuration
   */
  public RagSession enableThinking(final boolean enableThinking) {
    this.chat.enableThinking(enableThinking);
    return this;
  }

  /**
   * Scratchpad markers for parse and ChatML skip-seed. Delegates to the inner
   * {@link ChatSession#thinkTags(ThinkTags)}. Prefer
   * {@link com.igormaznitsa.nanollvm.models.LlmModel#OPTION_THINK_TAGS} at load so every session
   * sharing the checkpoint uses the same pair.
   *
   * @param thinkTags must not be {@code null}
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession thinkTags(final ThinkTags thinkTags) {
    this.chat.thinkTags(thinkTags);
    return this;
  }

  /**
   * When {@code true}, retries once (scrubbing matching assistant turns) and may salvage from
   * advisor notes if the main answer matches {@link #unusableAnswer(Predicate)}. Off by default.
   *
   * @param enable {@code true} to retry / salvage unusable answers
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession recoverUnusableAnswers(final boolean enable) {
    this.chat.recoverUnusableAnswers(enable);
    return this;
  }

  /**
   * Predicate for answers treated as unusable when {@link #recoverUnusableAnswers(boolean)} is on.
   * Default: blank only.
   *
   * @param predicate must not be {@code null}
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession unusableAnswer(final Predicate<String> predicate) {
    this.chat.unusableAnswer(predicate);
    return this;
  }

  /**
   * Fallback visible reply when recovery still yields nothing usable.
   *
   * @param fallback non-blank text shown to the user
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession unusableAnswerFallback(final String fallback) {
    this.chat.unusableAnswerFallback(fallback);
    return this;
  }

  /**
   * Caps retained dialog turns (system + user + assistant). Oldest non-system messages are dropped
   * when the cap is exceeded. Default from
   * {@link com.igormaznitsa.nanollvm.utils.ResourceLimits#maxHistoryMessages()}.
   *
   * @param maxHistoryMessages must be {@code >= 1}
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession maxHistoryMessages(final int maxHistoryMessages) {
    this.chat.maxHistoryMessages(maxHistoryMessages);
    return this;
  }

  /**
   * When {@code true}, emits {@link com.igormaznitsa.nanollvm.chat.LlmTextKind#TEXT_DEBUG} with the
   * prepared model-user text after advisors. Off by default.
   *
   * @param emitDebugPrompts {@code true} to send prepared prompts to listeners
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession emitDebugPrompts(final boolean emitDebugPrompts) {
    this.chat.emitDebugPrompts(emitDebugPrompts);
    return this;
  }

  /**
   * Replaces sampling parameters for subsequent turns. When a turn retrieves hits, temperature
   * above {@value #GROUNDED_TEMPERATURE_CAP} is clamped to that cap so grounded answers stay
   * conservative. No-hit turns keep the caller's temperature and may raise max tokens via
   * {@link #maxTokensWhenNoHits(int)}.
   *
   * @param samplingParams new knobs; must not be {@code null}
   * @return {@code this} for fluent configuration
   * @see SamplingParams
   */
  public RagSession sampling(final SamplingParams samplingParams) {
    this.baseSampling = requireNonNull(samplingParams, "samplingParams");
    this.chat.sampling(this.baseSampling);
    return this;
  }

  /**
   * Appends few-shot turns on the inner chat session.
   *
   * @param messages seed turns after the engine system seed; must not be {@code null}
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession seed(final ChatMessage... messages) {
    this.chat.seed(messages);
    return this;
  }

  /**
   * Appends few-shot turns on the inner chat session.
   *
   * @param messages seed turns after the engine system seed; must not be {@code null}
   * @return {@code this} for fluent configuration
   * @since 1.1.0
   */
  public RagSession seed(final List<ChatMessage> messages) {
    this.chat.seed(messages);
    return this;
  }

  /**
   * When retrieval returns no passages, use at least this many new tokens (capped by
   * {@link SamplingParams#maxTokens()} from {@link #sampling(SamplingParams)} when lower). Default
   * {@code 384}. Grounded turns still use the {@link #sampling(SamplingParams)} budget.
   *
   * @param maxTokens must be {@code >= 1}
   * @return {@code this} for fluent configuration
   * @throws IllegalArgumentException if {@code maxTokens < 1}
   */
  public RagSession maxTokensWhenNoHits(final int maxTokens) {
    if (maxTokens < 1) {
      throw new IllegalArgumentException("maxTokens must be >= 1");
    }
    this.maxTokensWhenNoHits = maxTokens;
    return this;
  }

  /**
   * Caps wall-clock time for the inner {@link ChatSession} generate of each turn. Delegates to
   * {@link ChatSession#timeout(Duration)}.
   *
   * @param timeout generate deadline, or {@code null} / zero / negative for no limit
   * @return {@code this} for fluent configuration
   */
  public RagSession timeout(final Duration timeout) {
    this.chat.timeout(timeout);
    return this;
  }

  /**
   * CLI sugar: installs {@link ChatSession#streamTo(PrintStream, PrintStream, boolean)} on the inner
   * session.
   *
   * @param thinkOut  destination for thinking / advisor notes
   * @param answerOut destination for the visible assistant answer
   * @param color     when {@code true}, dim cyan ANSI styling on the thinking stream
   * @return {@code this} for fluent configuration
   */
  public RagSession streamTo(final PrintStream thinkOut, final PrintStream answerOut,
                             final boolean color) {
    this.chat.streamTo(thinkOut, answerOut, color);
    return this;
  }

  /**
   * Sets the session-level {@link LlmListener} on the inner {@link ChatSession}.
   *
   * @param listener chat / diagnostics sink, or {@code null} for silent
   * @return {@code this} for fluent configuration
   */
  public RagSession listen(final LlmListener listener) {
    this.chat.listen(listener);
    return this;
  }

  /**
   * Composes a diagnostics sink on the inner {@link ChatSession}
   * ({@link ChatSession#diagnostics(Consumer)}). Does not wipe a prior {@link #listen(LlmListener)} /
   * {@link #streamTo(PrintStream, PrintStream, boolean)}; {@code null} is a no-op.
   *
   * @param diagnostics consumer of diagnostic lines, or {@code null} as a no-op
   * @return {@code this} for fluent configuration
   */
  public RagSession diagnostics(final Consumer<String> diagnostics) {
    this.chat.diagnostics(diagnostics);
    return this;
  }

  /**
   * Passages used for the last {@link #send(String)} / {@link #ask(String)} (empty when the corpus
   * was skipped or retrieval returned nothing).
   *
   * @return an unmodifiable snapshot; empty before the first turn
   */
  public List<RagHit> lastHits() {
    return this.lastHits;
  }

  /**
   * Inner chat session (history, sampling, listeners). Prefer the {@code RagSession} knobs when they
   * exist so RAG callers need not drop through this method.
   *
   * @return the wrapped {@link ChatSession}; not a copy
   */
  public ChatSession chat() {
    return this.chat;
  }

  /**
   * Drops conversation history and retrieval state (hits, anchor query, last source). Sampling,
   * timeout, and listeners on the inner session stay.
   */
  public void clear() {
    this.chat.clear();
    this.lastHits = List.of();
    this.anchorQuery = "";
    this.lastSource = "";
    this.lastRetrievalQuery = "";
  }

  /**
   * Per-turn sampling: raise max tokens when retrieval missed, otherwise cap temperature at
   * {@value #GROUNDED_TEMPERATURE_CAP} so grounded answers stay conservative.
   */
  private void applyTurnSampling() {
    SamplingParams base = this.baseSampling != null
      ? this.baseSampling
      : this.chat.samplingParams();
    int maxTokens = base.maxTokens();
    float temperature = base.temperature();
    if (this.lastHits.isEmpty()) {
      if (this.maxTokensWhenNoHits > maxTokens) {
        maxTokens = this.maxTokensWhenNoHits;
      }
    } else if (temperature > GROUNDED_TEMPERATURE_CAP) {
      temperature = GROUNDED_TEMPERATURE_CAP;
    }
    this.chat.sampling(base.withTemperature(temperature).withMaxTokens(maxTokens));
  }

  /**
   * Convenience: {@link #send(String)} then {@link ChatReply#answer()}.
   *
   * @param question non-blank user turn
   * @return visible assistant answer (thinking stripped)
   * @throws IllegalArgumentException                                          if {@code question} is blank after strip
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException if cancel fires
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException   if the session timeout elapses
   */
  public String ask(final String question) {
    return this.send(question).answer();
  }

  /**
   * {@code false} when the index reports zero passages — skip rewrite, retrieve, and RAG prompts.
   * Unknown sizes ({@link RagIndex#size()} {@code < 0}) still use the RAG path.
   */
  private boolean hasCorpus() {
    return this.index.size() != 0;
  }

  /**
   * When {@code true}, grounded turns (retrieval hits) see only the RAG-augmented user message,
   * not earlier assistant replies — avoids tiny-model latch on prior answers. Default {@code false}.
   * No-hit turns always isolate regardless of this flag.
   *
   * @param isolateGeneration {@code true} to omit prior assistant answers from grounded generates
   * @return {@code this} for fluent configuration
   */
  public RagSession isolateGeneration(final boolean isolateGeneration) {
    this.isolateGeneration = isolateGeneration;
    return this;
  }

  /**
   * Generates a grounded reply for {@code question} (rewrite / retrieve / prompt / chat). History
   * stores the original question; the model sees a context-augmented last turn when retrieval hits.
   *
   * @param question non-blank user turn
   * @return finished {@link ChatReply}: {@code answer} for the user, optional {@code thinking},
   *         and measured {@code stats}
   * @throws IllegalArgumentException                                          if {@code question} is blank after strip
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationCancelledException if cancel fires
   * @throws com.igormaznitsa.nanollvm.exceptions.GenerationTimeoutException   if the session timeout elapses
   */
  public ChatReply send(final String question) {
    requireNonNull(question, "question");
    String q = question.strip();
    if (q.isEmpty()) {
      throw new IllegalArgumentException("question must not be blank");
    }

    if (!this.hasCorpus()) {
      this.lastHits = List.of();
      this.lastRetrievalQuery = "";
      return this.chat.send(q);
    }

    this.lastHits = List.copyOf(this.retrieve(q));
    this.updateAnchorAfterRetrieve(q);
    if (!this.lastHits.isEmpty()) {
      this.lastSource = this.lastHits.getFirst().chunk().source();
    }

    String prompt = UserMessage.format(this.lastHits, q, this.maxContextChars);
    this.applyTurnSampling();
    boolean isolate = this.lastHits.isEmpty() || this.isolateGeneration;
    return this.chat.sendPrepared(q, prompt, isolate);
  }

  /**
   * Resolves a search string, fetches a candidate pool from the index, then clips to
   * {@link #topK(int)} (preferring the prior source on short follow-ups).
   *
   * @return an unmodifiable list; empty when the query is off-topic or rewrite yields nothing
   */
  private List<RagHit> retrieve(final String question) {
    Optional<String> retrievalQuery = this.resolveRetrievalQuery(question);
    if (retrievalQuery.isEmpty()) {
      this.lastRetrievalQuery = "";
      return List.of();
    }
    this.lastRetrievalQuery = retrievalQuery.get();
    int pool = Math.max(this.topK * 4, this.topK);
    List<RagHit> candidates = this.index.retrieve(this.lastRetrievalQuery, pool);
    if (Retrieval.shortFollowUp(question) && !this.lastSource.isEmpty()) {
      candidates = Retrieval.preferPriorSource(candidates, this.lastSource, this.topK);
    } else {
      candidates = Retrieval.clip(candidates, this.topK);
    }
    return candidates;
  }

  /**
   * Keeps the last successful retrieval string as the follow-up anchor, or replaces the anchor
   * when {@code question} is a standalone (non-short) turn.
   */
  private void updateAnchorAfterRetrieve(final String question) {
    if (Retrieval.shortFollowUp(question)) {
      if (!this.lastHits.isEmpty() && !this.lastRetrievalQuery.isBlank()) {
        this.anchorQuery = this.lastRetrievalQuery;
      }
      return;
    }
    if (Retrieval.updatesAnchorFromQuestion(question)) {
      this.anchorQuery = question;
    }
  }

  /**
   * Chooses the index query: skip off-topic; use a short follow-up as-is when it already hits;
   * otherwise rewrite or expand with the conversation anchor.
   *
   * @return empty when retrieval should be skipped
   */
  private Optional<String> resolveRetrievalQuery(final String question) {
    if (this.index.isOutsideCorpus(question)) {
      return Optional.empty();
    }

    boolean shortWithPrior = Retrieval.shortFollowUp(question)
      && !this.anchorQuery.isBlank();

    if (shortWithPrior) {
      if (Retrieval.hasHits(this.index, question)) {
        return Optional.of(question);
      }
      Optional<String> rewritten = QueryRewrite.rewrite(this.llm, this.anchorQuery, question);
      return Retrieval.queryAfterRewrite(
        question, this.anchorQuery, rewritten.orElse(null), this.index);
    }

    return Optional.of(question);
  }

  /**
   * Formats retrieved passages plus the user question into one model-facing user message.
   */
  static final class UserMessage {

    private UserMessage() {
    }

    /**
     * Concatenates hit texts (capped) and wraps them with the RAG user-turn templates; empty
     * context uses the no-passage prompt.
     */
    static String format(
      final List<RagHit> hits,
      final String question,
      final int maxContextChars
    ) {
      requireNonNull(hits, "hits");
      requireNonNull(question, "question");
      String q = question.strip();
      if (q.isEmpty()) {
        throw new IllegalArgumentException("question must not be blank");
      }
      if (maxContextChars < 64) {
        throw new IllegalArgumentException("maxContextChars must be >= 64");
      }

      String context = truncateContext(hits, maxContextChars);
      if (context.isBlank()) {
        return RagPrompts.withoutContext(q);
      }
      return RagPrompts.withContext(q, context);
    }

    /**
     * Joins passage bullets until {@code maxContextChars}; the first block may be sliced if it
     * alone exceeds the cap.
     */
    private static String truncateContext(final List<RagHit> hits, final int maxContextChars) {
      if (hits.isEmpty()) {
        return "";
      }
      List<String> parts = new ArrayList<>();
      int used = 0;
      for (RagHit hit : hits) {
        String block = "- " + hit.chunk().text().strip();
        int next = used == 0 ? block.length() : used + 2 + block.length();
        if (next > maxContextChars && used > 0) {
          break;
        }
        if (block.length() > maxContextChars && used == 0) {
          parts.add(block.substring(0, maxContextChars));
          break;
        }
        parts.add(block);
        used = next;
      }
      return String.join("\n", parts);
    }
  }

  /**
   * Isolated LLM rewrite of short RAG follow-ups into standalone search keywords.
   */
  static final class QueryRewrite {

    private static final SamplingParams REWRITE_SAMPLING =
      SamplingParams.builder().temperature(0.1f).maxTokens(48).build();
    private static final java.util.regex.Pattern SEARCH_PREFIX =
      java.util.regex.Pattern.compile("(?i)^\\s*(?:search\\s*:\\s*)?");
    private static final java.util.regex.Pattern WRAP_QUOTES =
      java.util.regex.Pattern.compile("^[\"'`]+|[\"'`]+$");

    private QueryRewrite() {
    }

    /**
     * Prompt for the isolated rewrite generate: standalone keywords when there is no prior, else
     * a follow-up rewrite against {@code priorContext}.
     */
    static String userMessage(final String priorContext, final String followUp) {
      requireNonNull(followUp, "followUp");
      String follow = followUp.strip();
      if (follow.isEmpty()) {
        throw new IllegalArgumentException("followUp must not be blank");
      }
      String prior = priorContext == null ? "" : priorContext.strip();
      if (prior.isEmpty()) {
        return RagPrompts.rewriteStandalone(follow);
      }
      return RagPrompts.rewriteFollowUp(prior, follow);
    }

    /**
     * {@link #parse(String, ThinkTags, ChatSpecials)} with library default markers.
     */
    static Optional<String> parse(final String rawModelText) {
      return parse(rawModelText, ThinkTags.DEFAULT, ChatSpecials.DEFAULT);
    }

    /** {@link #parse(String, ThinkTags, ChatSpecials)} with {@link ChatSpecials#DEFAULT}. */
    static Optional<String> parse(final String rawModelText, final ThinkTags tags) {
      return parse(rawModelText, tags, ChatSpecials.DEFAULT);
    }

    /**
     * First non-blank answer line, minus a leading {@code Search:} and wrapping quotes.
     * {@code NONE} / blank / over-long leftovers become empty.
     */
    static Optional<String> parse(
      final String rawModelText,
      final ThinkTags tags,
      final ChatSpecials specials
    ) {
      requireNonNull(tags, "tags");
      requireNonNull(specials, "specials");
      if (rawModelText == null || rawModelText.isBlank()) {
        return Optional.empty();
      }
      String answer = ChatReply.parse(rawModelText, tags, specials).answer().strip();
      if (answer.isEmpty()) {
        return Optional.empty();
      }
      String firstLine = answer.lines()
        .map(String::strip)
        .filter(line -> !line.isEmpty())
        .findFirst()
        .orElse("");
      if (firstLine.isEmpty()) {
        return Optional.empty();
      }
      String cleaned = WRAP_QUOTES.matcher(SEARCH_PREFIX.matcher(firstLine).replaceFirst(""))
        .replaceAll("")
        .strip();
      if (cleaned.isEmpty()) {
        return Optional.empty();
      }
      if ("none".equals(cleaned.toLowerCase(java.util.Locale.ROOT))) {
        return Optional.empty();
      }
      if (cleaned.length() > 240) {
        cleaned = cleaned.substring(0, 240).strip();
      }
      return Optional.of(cleaned);
    }

    /**
     * One isolated generate that turns a short follow-up into standalone search keywords (low
     * temperature, small token budget).
     */
    static Optional<String> rewrite(
      final LLM llm,
      final String priorContext,
      final String followUp
    ) {
      requireNonNull(llm, "llm");
      String user = userMessage(priorContext, followUp);
      List<com.igormaznitsa.nanollvm.chat.ChatMessage> turn =
        List.of(com.igormaznitsa.nanollvm.chat.ChatMessage.user(user));
      ThinkTags tags = llm.thinkTags();
      String prompt = llm.tokenizer().applyChatTemplate(
        com.igormaznitsa.nanollvm.chat.ChatMessages.toTemplateMaps(turn),
        true,
        false,
        tags.open(),
        tags.close());
      String raw = llm.generate(List.of(prompt), REWRITE_SAMPLING).getFirst().text();
      return parse(raw, tags, llm.chatSpecials());
    }
  }

  /**
   * Short-follow-up retrieval helpers (package-visible for tests).
   */
  static final class Retrieval {

    static final int SHORT_FOLLOW_UP_MAX_TOKENS = 6;

    private static final double PRIOR_SOURCE_COMPETITIVE = 0.55;

    private Retrieval() {
    }

    /**
     * {@code true} when {@code question} tokenizes to fewer than {@link #SHORT_FOLLOW_UP_MAX_TOKENS}
     * pieces (pronoun follow-ups, not standalone questions).
     */
    static boolean shortFollowUp(final String question) {
      return PreparedRag.tokenize(question).size() < SHORT_FOLLOW_UP_MAX_TOKENS;
    }

    /** {@code true} when {@link RagIndex#retrieve(String, int)} returns at least one hit. */
    static boolean hasHits(final RagIndex index, final String query) {
      requireNonNull(index, "index");
      requireNonNull(query, "query");
      return !index.retrieve(query, 1).isEmpty();
    }

    /**
     * Chooses the BM25 string after an optional rewrite of a short follow-up.
     * Prefer a usable rewrite; otherwise expand with Prior. Caller must already reject
     * off-topic follow-ups via {@link RagIndex#isOutsideCorpus(String)}.
     *
     * @param question  the short follow-up as typed
     * @param anchor    prior standalone retrieval keywords
     * @param rewritten rewritten query, or {@code null} when rewrite yielded nothing usable
     * @param index     used to reject an off-topic rewrite
     * @return a present query string (never empty for this path)
     */
    static Optional<String> queryAfterRewrite(
      final String question,
      final String anchor,
      final String rewritten,
      final RagIndex index
    ) {
      requireNonNull(question, "question");
      requireNonNull(index, "index");
      if (rewritten != null && !index.isOutsideCorpus(rewritten)) {
        return Optional.of(rewritten);
      }
      return Optional.of(anchorExpandedQuery(question, anchor));
    }

    /** {@code true} when {@code question} is long enough to replace the conversation anchor. */
    static boolean updatesAnchorFromQuestion(final String question) {
      return !shortFollowUp(question);
    }

    /**
     * Prior keywords plus the short follow-up, used when rewrite is empty or off-topic.
     */
    static String anchorExpandedQuery(final String question, final String anchor) {
      if (anchor == null || anchor.isBlank() || !shortFollowUp(question)) {
        return question;
      }
      return anchor + '\n' + question;
    }

    /**
     * On a short follow-up, keep hits from {@code priorSource} when their top score is within
     * {@value #PRIOR_SOURCE_COMPETITIVE} of the global best; otherwise clip the mixed pool.
     */
    static List<RagHit> preferPriorSource(
      final List<RagHit> candidates,
      final String priorSource,
      final int topK
    ) {
      if (candidates.isEmpty()) {
        return List.of();
      }
      if (priorSource == null || priorSource.isBlank()) {
        return clip(candidates, topK);
      }
      List<RagHit> same = candidates.stream()
        .filter(hit -> priorSource.equals(hit.chunk().source()))
        .toList();
      if (same.isEmpty()) {
        return clip(candidates, topK);
      }
      double best = candidates.getFirst().score();
      if (same.getFirst().score() >= best * PRIOR_SOURCE_COMPETITIVE) {
        return clip(same, topK);
      }
      return clip(candidates, topK);
    }

    static List<RagHit> clip(final List<RagHit> hits, final int topK) {
      if (hits.size() <= topK) {
        return List.copyOf(hits);
      }
      return List.copyOf(hits.subList(0, topK));
    }
  }
}
