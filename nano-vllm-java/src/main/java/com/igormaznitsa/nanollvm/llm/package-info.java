/**
 * Inference engine API: {@link LLM}, {@link Config}, {@link SamplingParams}, {@link GenerationStats}.
 * Status and chat text share {@link com.igormaznitsa.nanollvm.chat.LlmListener}.
 * <p>
 * Load weights via {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} into a shared {@link com.igormaznitsa.nanollvm.models.LlmModel},
 * then {@link LLM#builder(com.igormaznitsa.nanollvm.models.LlmModel)}. Chat helpers live in {@code chat}; retrieval in {@code rag}.
 * Optional advisors: {@link LLM.Builder#advisors(LlmAdvisorMixer, LlmAdvisor...)}.
 * Default sampling: {@link LLM.Builder#sampling(SamplingParams)} or {@link SamplingDefaults#neutral()}.
 * {@link LLM.GenerationOutput#stats()} carries engine-measured prompt/completion tokens and generate wall time.
 * <p>
 * A single {@link LLM} instance is not safe for concurrent generation; {@link LLM#cancel()} may interrupt
 * an in-flight generate. Library errors are in {@link com.igormaznitsa.nanollvm.exceptions}.
 */

package com.igormaznitsa.nanollvm.llm;
