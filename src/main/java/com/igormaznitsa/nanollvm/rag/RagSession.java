package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import java.io.PrintStream;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Retrieval-augmented chat over an {@link LLM}: {@link RagIndex} → prompt → generate.
 *
 * <p>History stores the original user text; the model sees a context-augmented last turn.
 * Short follow-ups with a prior turn are rewritten by an isolated LLM call into standalone
 * search keywords (or no retrieve when the model returns {@code NONE}). Short first turns and
 * longer standalone questions use the raw text for BM25. Without an LLM, retrieval falls back
 * to concatenating the previous longer user turn. Among competitive hits, shorter passages
 * are preferred so the prompt stays dense on any corpus. {@link #isolateGeneration(boolean)}
 * omits prior assistant answers from grounded (hit) generates only; no-hit turns keep
 * history so conversational follow-ups still work. Thinking is off by default so small
 * max-token budgets are not spent on {@code <think>} blocks. Grounded turns also clamp
 * sampling temperature. Not thread-safe.
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
    this.llm = llm;
    this.chat = requireNonNull(chat, "chat");
    this.index = requireNonNull(index, "index");
    this.isolateGeneration = llm != null && llm.tokenizer().isGemmaChat();
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
    return new RagSession(null, chat, index);
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

  /**
   * When {@code true}, grounded turns (retrieval hits) see only the RAG-augmented user message,
   * not earlier assistant replies — avoids tiny-model latch on prior answers. Defaults on for Gemma.
   * Turns with no hits keep full chat history so follow-ups (“fix the method above”) still work.
   */
  public RagSession isolateGeneration(final boolean isolateGeneration) {
    this.isolateGeneration = isolateGeneration;
    return this;
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

    this.lastHits = this.retrieve(q);
    this.updateAnchorAfterRetrieve(q);
    if (!this.lastHits.isEmpty()) {
      this.lastSource = this.lastHits.getFirst().chunk().source();
    }

    boolean compact = this.llm != null && this.llm.tokenizer().isGemmaChat();
    String prompt = RagPrompt.format(this.lastHits, q, this.maxContextChars, compact);
    this.applyTurnSampling();
    boolean isolate = this.isolateGeneration && !this.lastHits.isEmpty();
    return this.chat.sendPrepared(q, prompt, isolate);
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

  private List<RagHit> retrieve(final String question) {
    Optional<String> retrievalQuery = this.resolveRetrievalQuery(question);
    if (retrievalQuery.isEmpty()) {
      this.lastRetrievalQuery = "";
      return List.of();
    }
    this.lastRetrievalQuery = retrievalQuery.get();
    int pool = Math.max(this.topK * 4, this.topK);
    List<RagHit> candidates = this.index.retrieve(this.lastRetrievalQuery, pool);
    List<RagHit> grounded = RagRetrieval.preferCompactPassages(candidates, pool);
    if (RagRetrieval.needsAnchor(question) && !this.lastSource.isEmpty()) {
      return RagRetrieval.preferPriorSource(grounded, this.lastSource, this.topK);
    }
    return grounded.size() <= this.topK
      ? List.copyOf(grounded)
      : List.copyOf(grounded.subList(0, this.topK));
  }

  private void updateAnchorAfterRetrieve(final String question) {
    if (this.llm != null && RagRetrieval.needsRewrite(question)) {
      if (!this.lastHits.isEmpty() && !this.lastRetrievalQuery.isBlank()) {
        this.anchorQuery = this.lastRetrievalQuery;
      }
      return;
    }
    if (RagRetrieval.shouldUpdateAnchor(question)) {
      this.anchorQuery = question;
      return;
    }
    if (this.llm == null && this.anchorQuery.isBlank() && !this.isOutsideCorpus(question)) {
      this.anchorQuery = question;
    }
  }

  private Optional<String> resolveRetrievalQuery(final String question) {
    if (this.llm != null && RagRetrieval.needsRewrite(question) && !this.anchorQuery.isBlank()) {
      return RagQueryRewrite.rewrite(this.llm, this.anchorQuery, question);
    }
    if (this.isOutsideCorpus(question)) {
      return Optional.empty();
    }
    if (this.llm != null) {
      return Optional.of(question);
    }
    return Optional.of(RagRetrieval.retrievalQuery(question, this.anchorQuery));
  }

  private boolean isOutsideCorpus(final String question) {
    return switch (this.index) {
      case PreparedRag prepared -> prepared.bm25().isOutsideCorpus(question);
      case Bm25Index bm25 -> bm25.isOutsideCorpus(question);
      default -> false;
    };
  }
}
