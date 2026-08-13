package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatHistory;
import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatMessages;
import com.igormaznitsa.nanollvm.chat.ChatReply;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Runs configured {@link LlmAdvisor}s on the same engine (one batched {@link LLM#generate}), then
 * mixes replies via {@link LlmAdvisorMixer}.
 */
final class AdvisorRunner {

  private static final int MAX_ADVISOR_TOKENS = 256;
  private static final float ADVISOR_TEMPERATURE = 0.3f;

  private AdvisorRunner() {
  }

  /**
   * When the LLM has no advisors, returns {@code modelUserText} unchanged. Otherwise consults each
   * advisor with {@code priorDialog} plus the prepared user turn, then mixes notes.
   */
  public static AdvisorEnrichment enrich(
    final LLM llm,
    final String modelUserText,
    final List<ChatMessage> priorDialog,
    final SamplingParams mainSampling
  ) {
    requireNonNull(llm, "llm");
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(priorDialog, "priorDialog");
    requireNonNull(mainSampling, "mainSampling");

    List<LlmAdvisor> advisors = llm.advisors();
    if (advisors.isEmpty()) {
      return AdvisorEnrichment.passthrough(modelUserText);
    }

    List<ChatMessage> prior = List.copyOf(priorDialog);
    SamplingParams sampling = advisorSampling(mainSampling);
    List<String> prompts = advisors.stream()
      .map(advisor -> promptForAdvisor(llm, advisor, prior, modelUserText, sampling))
      .toList();
    List<LLM.GenerationOutput> outputs = llm.generate(prompts, sampling);
    List<AdvisorResponse> responses = IntStream.range(0, advisors.size())
      .mapToObj(i -> new AdvisorResponse(
        advisors.get(i).name(),
        parseAnswer(llm, outputs.get(i))))
      .toList();

    List<String> noteTexts = responses.stream()
      .map(AdvisorResponse::text)
      .toList();
    List<String> salvageNotes = AdvisorPrompt.selectNotesForMix(modelUserText, noteTexts);

    ChatHistory history = ChatHistory.of(prior);
    String mixed = llm.advisorMixer().mixPrompt(llm, responses, history, modelUserText);
    return new AdvisorEnrichment(mixed, responses, salvageNotes);
  }

  private static String promptForAdvisor(
    final LLM llm,
    final LlmAdvisor advisor,
    final List<ChatMessage> priorDialog,
    final String modelUserText,
    final SamplingParams sampling
  ) {
    List<ChatMessage> turn = new ArrayList<>(
      AdvisorPrompt.dialogTurn(advisor.prompt(), priorDialog, modelUserText));
    ChatMessages.truncateHistory(
      turn,
      llm.tokenizer(),
      llm.config().maxModelLen(),
      sampling.maxTokens(),
      false,
      llm.thinkTags());
    return llm.tokenizer().applyChatTemplate(
      ChatMessages.toTemplateMaps(turn),
      true,
      false,
      llm.thinkTags().open(),
      llm.thinkTags().close());
  }

  private static String parseAnswer(final LLM llm, final LLM.GenerationOutput output) {
    Tokenizer tokenizer = llm.tokenizer();
    String raw = output.text();
    if (raw == null || raw.isBlank()) {
      raw = tokenizer.decode(output.tokenIds(), tokenizer.skipSpecialTokensOnChatDecode());
    }
    String answer = ChatReply.parse(raw, llm.thinkTags()).answer().strip();
    return AdvisorPrompt.usableNote(answer, llm.advisorNoteFilter());
  }

  private static SamplingParams advisorSampling(final SamplingParams main) {
    int maxTokens = Math.min(main.maxTokens(), MAX_ADVISOR_TOKENS);
    float temperature = Math.min(main.temperature(), ADVISOR_TEMPERATURE);
    if (temperature <= 1e-10f) {
      temperature = ADVISOR_TEMPERATURE;
    }
    return new SamplingParams(temperature, maxTokens, false, main.topK(), main.topP());
  }
}
