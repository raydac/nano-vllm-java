/**
 * Public library surface for embedding nano-vllm-java.
 * <p>
 * Prefer {@link ModelFactory#make(java.nio.file.Path)} for a shared immutable {@link Model},
 * then {@link LLM#builder(Model)} (or path convenience {@link LLM#builder(java.nio.file.Path)}),
 * {@link EngineIo}, {@link SamplingParams}, {@link com.igormaznitsa.nanollvm.chat.ChatSession},
 * and text RAG via {@link com.igormaznitsa.nanollvm.rag.RagFactory} /
 * {@link com.igormaznitsa.nanollvm.rag.PreparedRag} / {@link LLM#rag}.
 * Engine internals live under {@code engine}, {@code layers}, {@code tensor}, {@code models},
 * and {@code internal} (weight loader, safetensors, inference Context).
 * Exported {@code utils} is limited to path helpers ({@code BundledModels}/{@code BundledRag}),
 * {@code Json}, and {@code NanoVllmProps}.
 * <p>
 * A single {@link LLM} instance is not safe for concurrent generation; call sequentially
 * or use separate instances sharing one {@link Model}. Share one
 * {@link com.igormaznitsa.nanollvm.rag.PreparedRag} the same way across LLMs.
 * {@link LLM#cancel()} may interrupt an in-flight generate.
 */

package com.igormaznitsa.nanollvm;
