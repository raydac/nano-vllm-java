/**
 * Inference engine API: {@link LLM}, runtime {@link Config}, {@link EngineIo}, and {@link SamplingParams}.
 * <p>
 * Load weights via {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} into a shared {@link com.igormaznitsa.nanollvm.models.LlmModel},
 * then {@link LLM#builder(com.igormaznitsa.nanollvm.models.LlmModel)} (or path convenience
 * {@link LLM#builder(java.nio.file.Path)}). Chat helpers live in {@code chat}; retrieval in {@code rag}.
 * <p>
 * A single {@link LLM} instance is not safe for concurrent generation; {@link LLM#cancel()} may interrupt
 * an in-flight generate. Library errors are in {@link com.igormaznitsa.nanollvm.exceptions}.
 */

package com.igormaznitsa.nanollvm.llm;
