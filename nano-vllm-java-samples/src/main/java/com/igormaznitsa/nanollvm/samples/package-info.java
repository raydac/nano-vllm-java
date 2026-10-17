/**
 * Runnable demos ({@link HelloWorld}, {@link NextTokenHelloWorld}, {@link LogTriageHelloWorld},
 * {@link AdvisorRagHelloWorld}, {@link RagTunerHelloWorld}, {@link EmbeddingsHelloWorld},
 * {@link TranscribeHelloWorld}, {@link SynthesizeHelloWorld}, {@link VoiceReplyHelloWorld},
 * {@link LanguageIdHelloWorld}, {@link JevHelloWorld}, {@link Example}, {@link Bench},
 * {@link KernelChatBench}).
 * Helpers for local {@code models/} and {@code rag/} live in {@code samples.utils}.
 * Launch from the repo root with
 * {@code mvn -pl nano-vllm-java-samples -q exec:java} (default main {@link Example}:
 * model menu, then chat (RAG mode, advisor count) or a Jev decision loop, or embed, classify, transcribe, or synthesize).
 */

package com.igormaznitsa.nanollvm.samples;
