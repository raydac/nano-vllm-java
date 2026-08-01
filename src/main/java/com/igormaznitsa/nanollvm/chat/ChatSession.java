package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.LLM;
import com.igormaznitsa.nanollvm.SamplingDefaults;
import com.igormaznitsa.nanollvm.SamplingParams;
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
  private Consumer<String> diagnostics = message -> {
  };

  public ChatSession(LLM llm) {
    this(llm, llm.defaultSampling());
  }

  public ChatSession(LLM llm, SamplingParams samplingParams) {
    this.llm = requireNonNull(llm, "llm");
    this.samplingParams = requireNonNull(samplingParams, "samplingParams");
    this.history = new ArrayList<>(llm.newConversation());
  }

  public static ChatSession open(LLM llm, int maxTokens) {
    return new ChatSession(llm, SamplingDefaults.forTokenizer(llm.tokenizer(), maxTokens));
  }

  public ChatSession sampling(SamplingParams samplingParams) {
    this.samplingParams = requireNonNull(samplingParams, "samplingParams");
    return this;
  }

  public ChatSession timeout(Duration timeout) {
    this.timeout = timeout == null ? Duration.ZERO : timeout;
    return this;
  }

  public ChatSession streamTo(PrintStream thinkOut, PrintStream answerOut, boolean color) {
    this.thinkOut = thinkOut;
    this.answerOut = answerOut;
    this.color = color;
    return this;
  }

  public ChatSession diagnostics(Consumer<String> diagnostics) {
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

  public ChatReply send(String userText) {
    requireNonNull(userText, "userText");
    String user = userText.strip();
    if (user.isEmpty()) {
      throw new IllegalArgumentException("userText must not be blank");
    }

    this.history.add(ChatMessage.user(user));
    Tokenizer tokenizer = this.llm.tokenizer();
    ChatMessages.truncateHistory(
        this.history, tokenizer, this.llm.config().maxModelLen(), this.samplingParams.maxTokens());

    boolean gemmaChat = tokenizer.isGemmaChat();
    StreamPrinter printer = this.newPrinter();
    ChatReply reply = this.generateTurn(printer);

    if (gemmaChat && ChatPrompts.isSetupBoilerplate(reply.answer())) {
      ChatMessages.scrubSetupBoilerplateTurns(this.history);
      this.diagnostics.accept("(setup boilerplate — retrying without filler history)");
      printer = this.newPrinter();
      reply = this.generateTurn(printer);
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

  private ChatReply generateTurn(StreamPrinter printer) {
    Tokenizer tokenizer = this.llm.tokenizer();
    boolean gemmaChat = tokenizer.isGemmaChat();
    boolean enableThinking = !gemmaChat;
    String prompt = tokenizer.applyChatTemplate(
        ChatMessages.toTemplateMaps(this.history), true, enableThinking);

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

  private ChatReply finishTurn(ChatReply reply, StreamPrinter printer) {
    String answer = reply.answer().strip();
    String thinking = reply.thinking();
    boolean thinkOpen = reply.thinkOpen();

    if (answer.isBlank() && !thinking.isBlank()) {
      answer = AssistantParts.salvageFromThinking(thinking);
      this.diagnostics.accept(thinkOpen
          ? "(reply recovered from unclosed thinking)"
          : "(reply recovered from thinking; model omitted visible answer)");
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
}
