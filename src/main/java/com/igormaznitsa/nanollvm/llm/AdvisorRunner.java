package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.AssistantParts;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.List;

/**
 * Runs configured LLM advisors on the same engine (no history, no UI stream). Advisor notes
 * appear on the thinking stream; only grounded, non-conflicting notes are mixed into the main
 * user text as unverified hints.
 */
public final class AdvisorRunner {

  private static final int MAX_ADVISOR_TOKENS = 256;
  private static final float ADVISOR_TEMPERATURE = 0.3f;

  private AdvisorRunner() {
  }

  /**
   * When the LLM has no advisor prompts, returns {@code modelUserText} unchanged. Otherwise
   * consults each advisor and mixes their notes into the main prompt.
   */
  public static AdvisorEnrichment enrich(
    final LLM llm,
    final String modelUserText,
    final SamplingParams mainSampling
  ) {
    requireNonNull(llm, "llm");
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(mainSampling, "mainSampling");

    List<String> rolePrompts = llm.advisorPrompts();
    if (rolePrompts.isEmpty()) {
      return AdvisorEnrichment.passthrough(modelUserText);
    }

    boolean compact = llm.tokenizer().isGemmaChat();
    List<String> prompts = rolePrompts.stream()
      .map(role -> AdvisorPrompt.isolated(llm.tokenizer(), role, modelUserText))
      .toList();
    SamplingParams sampling = advisorSampling(mainSampling);
    List<String> answers = llm.advisorMode() == AdvisorMode.PARALLEL
      ? runParallel(llm, prompts, sampling)
      : runSequential(llm, prompts, sampling);
    List<String> notes = answers.stream()
      .map(answer -> answer == null ? "" : answer.strip())
      .toList();
    List<String> mixNotes = AdvisorPrompt.selectNotesForMix(modelUserText, notes);
    return new AdvisorEnrichment(
      AdvisorPrompt.mix(modelUserText, mixNotes, compact),
      notes,
      mixNotes);
  }

  private static List<String> runParallel(
    final LLM llm,
    final List<String> prompts,
    final SamplingParams sampling
  ) {
    return llm.generate(prompts, sampling).stream()
      .map(output -> parseAnswer(llm.tokenizer(), output))
      .toList();
  }

  private static List<String> runSequential(
    final LLM llm,
    final List<String> prompts,
    final SamplingParams sampling
  ) {
    return prompts.stream()
      .map(prompt -> parseAnswer(llm.tokenizer(),
        llm.generate(List.of(prompt), sampling).getFirst()))
      .toList();
  }

  private static String parseAnswer(final Tokenizer tokenizer, final LLM.GenerationOutput output) {
    String raw = output.text();
    if (raw == null || raw.isBlank()) {
      raw = tokenizer.decode(output.tokenIds(), tokenizer.isGemmaChat());
    }
    String answer = AssistantParts.parse(raw).answer().strip();
    if (answer.isEmpty() || ChatPrompts.isSetupBoilerplate(answer)) {
      return "";
    }
    return answer;
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
