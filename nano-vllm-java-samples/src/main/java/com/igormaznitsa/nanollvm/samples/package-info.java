/**
 * Runnable demos ({@link HelloWorld}, {@link NextTokenHelloWorld}, {@link LogTriageHelloWorld},
 * {@link AdvisorRagHelloWorld}, {@link RagTunerHelloWorld}, {@link EmbeddingsHelloWorld},
 * {@link Example}, {@link Bench}).
 * Helpers for local {@code models/} and {@code rag/} live in {@code samples.utils}.
 * Launch from the repo root with
 * {@code mvn -pl nano-vllm-java-samples -q exec:java} (default main {@link Example}:
 * model menu, RAG mode, advisor count, then a line-oriented chat, embed, or few-shot classify session).
 */

package com.igormaznitsa.nanollvm.samples;
