package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.AssistantParts;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.List;

/**
 * Runs configured LLM subagents on the same engine (no history, no UI stream) and mixes their
 * answers into the prepared user text for the main generate.
 */
public final class SubagentRunner {

  private static final int MAX_SUBAGENT_TOKENS = 256;
  private static final float SUBAGENT_TEMPERATURE = 0.3f;

  private SubagentRunner() {
  }

  /**
   * When the LLM has no subagent prompts, returns {@code modelUserText} unchanged. Otherwise
   * consults each subagent and appends advisor notes.
   */
  public static SubagentEnrichment enrich(
    final LLM llm,
    final String modelUserText,
    final SamplingParams mainSampling
  ) {
    requireNonNull(llm, "llm");
    requireNonNull(modelUserText, "modelUserText");
    requireNonNull(mainSampling, "mainSampling");

    List<String> rolePrompts = llm.subagentPrompts();
    if (rolePrompts.isEmpty()) {
      return SubagentEnrichment.passthrough(modelUserText);
    }

    boolean compact = llm.tokenizer().isGemmaChat();
    List<String> prompts = rolePrompts.stream()
      .map(role -> SubagentPrompt.isolated(llm.tokenizer(), role, modelUserText))
      .toList();
    SamplingParams sampling = subagentSampling(mainSampling);
    List<String> answers = llm.subagentMode() == SubagentMode.PARALLEL
      ? runParallel(llm, prompts, sampling)
      : runSequential(llm, prompts, sampling);
    List<String> notes = SubagentPrompt.usableNotes(answers);
    return new SubagentEnrichment(
      SubagentPrompt.mix(modelUserText, notes, compact),
      notes);
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

  private static SamplingParams subagentSampling(final SamplingParams main) {
    int maxTokens = Math.min(main.maxTokens(), MAX_SUBAGENT_TOKENS);
    float temperature = Math.min(main.temperature(), SUBAGENT_TEMPERATURE);
    if (temperature <= 1e-10f) {
      temperature = SUBAGENT_TEMPERATURE;
    }
    return new SamplingParams(temperature, maxTokens, false, main.topK(), main.topP());
  }
}
