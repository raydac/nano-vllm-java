package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.prompts.RagPrompts;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Retrieval-augmented chat over an {@link LLM}: {@link RagIndex} → prompt → generate.
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
 * when on (default for Gemma). No-hit turns always isolate so prior corpus answers cannot latch.
 * Thinking is off by default so small max-token budgets are not spent on {@code <think>} blocks.
 * Grounded turns also clamp sampling temperature. Not thread-safe.
 *
 * <p>{@link #open(ChatSession, RagIndex)} reuses that session's {@link LLM} (rewrite and Gemma
 * policy stay enabled).
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
    this.isolateGeneration = this.llm.tokenizer().isGemmaChat();
    this.chat.enableThinking(false);
  }

  public static RagSession open(final LLM llm, final RagIndex index) {
    requireNonNull(llm, "llm");
    return new RagSession(llm, new ChatSession(llm), index);
  }

  public static RagSession open(final LLM llm, final RagIndex index, final int maxTokens) {
    requireNonNull(llm, "llm");
    return new RagSession(llm, ChatSession.open(llm, maxTokens), index);
  }

  public static RagSession open(final ChatSession chat, final RagIndex index) {
    requireNonNull(chat, "chat");
    return new RagSession(chat.llm(), chat, index);
  }

  public RagSession topK(final int topK) {
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    this.topK = topK;
    return this;
  }

  public RagSession maxContextChars(final int maxContextChars) {
    if (maxContextChars < 64) {
      throw new IllegalArgumentException("maxContextChars must be >= 64");
    }
    this.maxContextChars = maxContextChars;
    return this;
  }

  public static String formatUserMessage(
    final List<RagHit> hits,
    final String question,
    final int maxContextChars
  ) {
    return UserMessage.format(hits, question, maxContextChars);
  }

  /**
   * RAG defaults to thinking off so the token budget goes to the grounded answer.
   * Re-enable for plain Qwen-style chain-of-thought if desired.
   */
  public RagSession enableThinking(final boolean enableThinking) {
    this.chat.enableThinking(enableThinking);
    return this;
  }

  public RagSession sampling(final SamplingParams samplingParams) {
    this.baseSampling = requireNonNull(samplingParams, "samplingParams");
    this.chat.sampling(this.baseSampling);
    return this;
  }

  /**
   * When retrieval returns no passages, use at least this many new tokens (capped by
   * {@link SamplingParams#maxTokens()} from {@link #sampling} when lower). Default {@code 384}.
   * Grounded turns still use the {@link #sampling} budget (e.g. short answers on Gemma).
   */
  public RagSession maxTokensWhenNoHits(final int maxTokens) {
    if (maxTokens < 1) {
      throw new IllegalArgumentException("maxTokens must be >= 1");
    }
    this.maxTokensWhenNoHits = maxTokens;
    return this;
  }

  public RagSession timeout(final Duration timeout) {
    this.chat.timeout(timeout);
    return this;
  }

  public RagSession streamTo(final PrintStream thinkOut, final PrintStream answerOut,
                             final boolean color) {
    this.chat.streamTo(thinkOut, answerOut, color);
    return this;
  }

  public RagSession listen(final LlmListener listener) {
    this.chat.listen(listener);
    return this;
  }

  public RagSession diagnostics(final Consumer<String> diagnostics) {
    this.chat.diagnostics(diagnostics);
    return this;
  }

  public List<RagHit> lastHits() {
    return this.lastHits;
  }

  public ChatSession chat() {
    return this.chat;
  }

  public void clear() {
    this.chat.clear();
    this.lastHits = List.of();
    this.anchorQuery = "";
    this.lastSource = "";
    this.lastRetrievalQuery = "";
  }

  public static String formatUserMessage(final List<RagHit> hits, final String question) {
    return UserMessage.format(hits, question, Integer.MAX_VALUE);
  }

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
    this.chat.sampling(new SamplingParams(
      temperature,
      maxTokens,
      base.ignoreEos(),
      base.topK(),
      base.topP()));
  }

  /**
   * Convenience: {@link #send(String)} then {@link ChatReply#answer()}.
   *
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
   * not earlier assistant replies — avoids tiny-model latch on prior answers. Defaults on for Gemma.
   * No-hit turns always isolate regardless of this flag.
   */
  public RagSession isolateGeneration(final boolean isolateGeneration) {
    this.isolateGeneration = isolateGeneration;
    return this;
  }

  /**
   * Generates a grounded reply for {@code question} (rewrite / retrieve / prompt / chat).
   *
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

    private static final SamplingParams REWRITE_SAMPLING = new SamplingParams(0.1f, 48);
    private static final java.util.regex.Pattern SEARCH_PREFIX =
      java.util.regex.Pattern.compile("(?i)^\\s*(?:search\\s*:\\s*)?");
    private static final java.util.regex.Pattern WRAP_QUOTES =
      java.util.regex.Pattern.compile("^[\"'`]+|[\"'`]+$");

    private QueryRewrite() {
    }

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

    static Optional<String> parse(final String rawModelText) {
      if (rawModelText == null || rawModelText.isBlank()) {
        return Optional.empty();
      }
      String answer = com.igormaznitsa.nanollvm.chat.ChatReply.parse(rawModelText)
        .answer().strip();
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

    static Optional<String> rewrite(
      final LLM llm,
      final String priorContext,
      final String followUp
    ) {
      requireNonNull(llm, "llm");
      String user = userMessage(priorContext, followUp);
      List<com.igormaznitsa.nanollvm.chat.ChatMessage> turn =
        List.of(com.igormaznitsa.nanollvm.chat.ChatMessage.user(user));
      String prompt = llm.tokenizer().applyChatTemplate(
        com.igormaznitsa.nanollvm.chat.ChatMessages.toTemplateMaps(turn), true, false);
      String raw = llm.generate(List.of(prompt), REWRITE_SAMPLING).getFirst().text();
      return parse(raw);
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

    static boolean shortFollowUp(final String question) {
      return PreparedRag.tokenize(question).size() < SHORT_FOLLOW_UP_MAX_TOKENS;
    }

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
     * @param rewritten rewritten query, or {@code null} when rewrite yielded nothing usable
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

    static boolean updatesAnchorFromQuestion(final String question) {
      return !shortFollowUp(question);
    }

    static String anchorExpandedQuery(final String question, final String anchor) {
      if (anchor == null || anchor.isBlank() || !shortFollowUp(question)) {
        return question;
      }
      return anchor + '\n' + question;
    }

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
