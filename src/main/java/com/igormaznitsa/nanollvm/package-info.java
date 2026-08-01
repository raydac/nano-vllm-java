/**
 * Public library surface for embedding nano-vllm-java.
 * <p>
 * Prefer {@link LLM#builder(java.nio.file.Path)}, {@link EngineIo}, {@link SamplingParams},
 * and {@link com.igormaznitsa.nanollvm.chat.ChatSession}. Engine internals live under {@code engine},
 * {@code layers}, and {@code tensor}.
 * <p>
 * A single {@link LLM} instance is not safe for concurrent generation; call sequentially
 * or use separate instances. {@link LLM#cancel()} may interrupt an in-flight generate.
 */

package com.igormaznitsa.nanollvm;
