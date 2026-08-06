package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

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
 * Multi-turn chat over an {@link LLM}: history, template, truncation, reply parsing.
 * Not thread-safe; use one session per conversation thread.
 */
public final class ChatSession {

  private final LLM llm;
  private final List<ChatMessage> history;
  private SamplingParams samplingParams;
  private Duration timeout = Duration.ZERO;
  private PrintStream thinkOut;
  private PrintStream answerOut;
  private boolean color;
  private Boolean enableThinking;
  private Consumer<String> diagnostics = message -> {
  };

  public ChatSession(final LLM llm) {
    this(llm, llm.defaultSampling());
  }

  public ChatSession(final LLM llm, final SamplingParams samplingParams) {
    this.llm = requireNonNull(llm, "llm");
    this.samplingParams = requireNonNull(samplingParams, "samplingParams");
    this.history = new ArrayList<>(llm.newConversation());
  }

  public static ChatSession open(final LLM llm, final int maxTokens) {
    return new ChatSession(llm, SamplingDefaults.forTokenizer(llm.tokenizer(), maxTokens));
  }

  public ChatSession sampling(final SamplingParams samplingParams) {
    this.samplingParams = requireNonNull(samplingParams, "samplingParams");
    return this;
  }

  public SamplingParams samplingParams() {
    return this.samplingParams;
  }

  public ChatSession timeout(final Duration timeout) {
    this.timeout = timeout == null ? Duration.ZERO : timeout;
    return this;
  }

  public ChatSession streamTo(final PrintStream thinkOut, final PrintStream answerOut,
                              final boolean color) {
    this.thinkOut = thinkOut;
    this.answerOut = answerOut;
    this.color = color;
    return this;
  }

  /**
   * Enables or disables thinking-scratchpad invitation for this session.
   *
   * <p>When {@code false}, the chat template seeds an empty {@code <think></think>} so Qwen-style
   * models skip long chain-of-thought (important for RAG token budgets). {@code null}/unset keeps
   * the default from {@link com.igormaznitsa.nanollvm.tokenizer.Tokenizer#invitesThinking()}.
   */
  public ChatSession enableThinking(final boolean enableThinking) {
    this.enableThinking = enableThinking;
    return this;
  }

  public ChatSession diagnostics(final Consumer<String> diagnostics) {
    this.diagnostics = diagnostics == null ? message -> {
    } : diagnostics;
    return this;
  }

  public List<ChatMessage> history() {
    return List.copyOf(this.history);
  }

  public void clear() {
    this.history.clear();
    this.history.addAll(this.llm.newConversation());
  }

  public ChatReply send(final String userText) {
    requireNonNull(userText, "userText");
    return this.sendPrepared(userText, userText);
  }

  /**
   * Adds {@code historyUserText} to the conversation history, but builds the model prompt as if
   * the last user turn were {@code modelUserText} (used by RAG to keep a short history while
   * injecting retrieved context for generation only).
   */
  public ChatReply sendPrepared(final String historyUserText, final String modelUserText) {
    return this.sendPrepared(historyUserText, modelUserText, false);
  }

  /**
   * Sends a turn that stores one user text in history but may generate from a different prepared
   * user string.
   *
   * @param isolateGeneration when {@code true}, the model sees only the system seed (if any) plus
   *                          the prepared user turn — prior assistant answers are kept in history
   *                          for the app, but not fed into this generate (avoids tiny-model latch)
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
    StreamPrinter printer = this.newPrinter();
    ChatReply reply = this.generateTurn(printer, modelUser, isolateGeneration);

    if (gemmaChat && ChatPrompts.isSetupBoilerplate(reply.answer())) {
      ChatMessages.scrubSetupBoilerplateTurns(this.history);
      this.diagnostics.accept("(setup boilerplate — retrying without filler history)");
      printer = this.newPrinter();
      reply = this.generateTurn(printer, modelUser, isolateGeneration);
      if (ChatPrompts.isSetupBoilerplate(reply.answer())) {
        reply = new ChatReply("", "Hello! What would you like to know?", false);
        this.diagnostics.accept("(setup boilerplate — used plain greeting fallback)");
      }
    }

    return this.finishTurn(reply, printer);
  }

  private StreamPrinter newPrinter() {
    if (this.thinkOut == null || this.answerOut == null) {
      return null;
    }
    return new StreamPrinter(this.thinkOut, this.answerOut, this.color);
  }

  private ChatReply generateTurn(
      final StreamPrinter printer,
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
          if (printer != null) {
            printer.update(AssistantParts.parse(tokenizer.decode(streamedIds, gemmaChat)));
          }
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

  private ChatReply finishTurn(final ChatReply reply, final StreamPrinter printer) {
    String answer = reply.answer().strip();
    String thinking = reply.thinking();
    boolean thinkOpen = reply.thinkOpen();

    if (this.shouldSalvageAnswer(answer, thinking)) {
      answer = AssistantParts.salvageFromThinking(thinking);
      this.diagnostics.accept(thinkOpen
          ? "(reply recovered from unclosed thinking)"
          : "(reply recovered from thinking; model omitted or truncated visible answer)");
    }
    if (answer.isBlank()) {
      answer = "Sorry — I couldn't form a reply. Please try again.";
      this.diagnostics.accept("(empty reply — used fallback)");
    }

    ChatReply finished = new ChatReply(thinking, answer, false);
    if (printer != null) {
      printer.update(new AssistantParts(finished.thinking(), finished.answer(), false));
      printer.closeTurn();
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
}
