package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.LLM;
import com.igormaznitsa.nanollvm.SamplingParams;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.chat.ChatSession;

import java.io.PrintStream;
import java.time.Duration;
import java.util.List;
import java.util.function.Consumer;

/**
 * Retrieval-augmented chat over an {@link LLM}: {@link RagIndex} → prompt → generate.
 *
 * <p>History stores the original user text; the model sees a context-augmented last turn.
 * Short follow-ups expand retrieval with the previous longer user turn and may prefer the
 * same source document (structural continuity). {@link #isolateGeneration(boolean)} omits
 * prior assistant answers from the generate template. Thinking is off by default so small
 * max-token budgets are not spent on {@code <think>} blocks. Not thread-safe.
 *
 * <pre>{@code
 * PreparedRag rag = RagFactory.make(Path.of("docs"));
 * ChatReply reply = llm.rag(rag).topK(2).send("What is the capital of France?");
 * }</pre>
 */
public final class RagSession {

  private final LLM llm;
  private final ChatSession chat;
  private final RagIndex index;
  private int topK = 4;
  private int maxContextChars = 3500;
  private boolean isolateGeneration;
  private List<RagHit> lastHits = List.of();
  private String anchorQuery = "";
  private String lastSource = "";

  private RagSession(LLM llm, ChatSession chat, RagIndex index) {
    this.llm = llm;
    this.chat = requireNonNull(chat, "chat");
    this.index = requireNonNull(index, "index");
    this.isolateGeneration = llm != null && llm.tokenizer().isGemmaChat();
    this.chat.enableThinking(false);
  }

  public static RagSession open(LLM llm, RagIndex index) {
    requireNonNull(llm, "llm");
    return new RagSession(llm, new ChatSession(llm), index);
  }

  public static RagSession open(LLM llm, RagIndex index, int maxTokens) {
    requireNonNull(llm, "llm");
    return new RagSession(llm, ChatSession.open(llm, maxTokens), index);
  }

  public static RagSession open(ChatSession chat, RagIndex index) {
    return new RagSession(null, chat, index);
  }

  public RagSession topK(int topK) {
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be > 0");
    }
    this.topK = topK;
    return this;
  }

  public RagSession maxContextChars(int maxContextChars) {
    if (maxContextChars < 64) {
      throw new IllegalArgumentException("maxContextChars must be >= 64");
    }
    this.maxContextChars = maxContextChars;
    return this;
  }

  /**
   * When {@code true}, each generate call sees only the RAG-augmented user turn (plus empty/default
   * system seed), not earlier assistant replies. Defaults on for Gemma chat tokenizers.
   */
  public RagSession isolateGeneration(boolean isolateGeneration) {
    this.isolateGeneration = isolateGeneration;
    return this;
  }

  /**
   * RAG defaults to thinking off so the token budget goes to the grounded answer.
   * Re-enable for plain Qwen-style chain-of-thought if desired.
   */
  public RagSession enableThinking(boolean enableThinking) {
    this.chat.enableThinking(enableThinking);
    return this;
  }

  public RagSession sampling(SamplingParams samplingParams) {
    this.chat.sampling(samplingParams);
    return this;
  }

  public RagSession timeout(Duration timeout) {
    this.chat.timeout(timeout);
    return this;
  }

  public RagSession streamTo(PrintStream thinkOut, PrintStream answerOut, boolean color) {
    this.chat.streamTo(thinkOut, answerOut, color);
    return this;
  }

  public RagSession diagnostics(Consumer<String> diagnostics) {
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
  }

  public ChatReply send(String question) {
    requireNonNull(question, "question");
    String q = question.strip();
    if (q.isEmpty()) {
      throw new IllegalArgumentException("question must not be blank");
    }

    this.lastHits = this.retrieve(q);
    if (RagRetrieval.shouldUpdateAnchor(q)) {
      this.anchorQuery = q;
    }
    if (!this.lastHits.isEmpty()) {
      this.lastSource = this.lastHits.getFirst().chunk().source();
    }

    boolean compact = this.llm != null && this.llm.tokenizer().isGemmaChat();
    String prompt = RagPrompt.format(this.lastHits, q, this.maxContextChars, compact);
    return this.chat.sendPrepared(q, prompt, this.isolateGeneration);
  }

  public String ask(String question) {
    return this.send(question).answer();
  }

  private List<RagHit> retrieve(String question) {
    String retrievalQuery = RagRetrieval.retrievalQuery(question, this.anchorQuery);
    int pool = Math.max(this.topK * 4, this.topK);
    List<RagHit> candidates = this.index.retrieve(retrievalQuery, pool);
    if (RagRetrieval.needsAnchor(question) && !this.lastSource.isEmpty()) {
      return RagRetrieval.preferPriorSource(candidates, this.lastSource, this.topK);
    }
    return candidates.size() <= this.topK
        ? List.copyOf(candidates)
        : List.copyOf(candidates.subList(0, this.topK));
  }
}
