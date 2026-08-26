/**
 * Library module root. Start here, then open the type that matches what you want to do.
 *
 * <p><b>Load a checkpoint</b> — {@link com.igormaznitsa.nanollvm.models.LlmModelFactory}
 * ({@code make(path)} or {@code open(path).…make()}). Every kind then goes through
 * {@link com.igormaznitsa.nanollvm.llm.LLM#builder}: chat, or typed
 * {@link com.igormaznitsa.nanollvm.llm.LLM#generate} for embeddings, Whisper, Piper, and raw
 * text continuation.
 *
 * <p><b>Talk to the model</b> — {@link com.igormaznitsa.nanollvm.llm.LLM}: {@code chat()} for a
 * conversation, {@code chatOnce} for one question,
 * {@link com.igormaznitsa.nanollvm.llm.LLM#generate(com.igormaznitsa.nanollvm.models.LlmInput, com.igormaznitsa.nanollvm.models.LlmModality)}
 * for embeddings / speech / TTS / raw continuation, {@code rag} to answer from your documents.
 * Reply style (length, randomness) is
 * {@link com.igormaznitsa.nanollvm.llm.SamplingParams}. Session extras (timeout, streaming,
 * history cap) live on {@link com.igormaznitsa.nanollvm.chat.ChatSession}.
 *
 * <p><b>Index documents</b> — {@link com.igormaznitsa.nanollvm.rag.RagFactory} then
 * {@code llm.rag(index)}. Chunk size is {@link com.igormaznitsa.nanollvm.rag.RagLoadOptions};
 * how many passages are stuffed into the prompt is {@code RagSession.maxContextChars}.
 *
 * <p>Errors are {@link com.igormaznitsa.nanollvm.exceptions}. JVM flags are
 * {@link com.igormaznitsa.nanollvm.utils.NanoLlvmProps}. Runnable demos live in the separate
 * Maven module {@code nano-vllm-java-samples}.
 */

package com.igormaznitsa.nanollvm;
