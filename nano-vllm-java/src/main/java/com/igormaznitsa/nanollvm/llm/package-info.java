/**
 * Inference engine API: bind a loaded {@link com.igormaznitsa.nanollvm.models.LlmModel} to an
 * {@link LLM}, then generate, chat, or retrieve.
 *
 * <p><b>Load vs run.</b> Weights live on a shareable {@link com.igormaznitsa.nanollvm.models.LlmModel}
 * from {@link com.igormaznitsa.nanollvm.models.LlmModelFactory}. Each {@link LLM} is one engine
 * (KV cache, scheduler, matmul). Close the engine first, then the model. Embedding checkpoints
 * use {@code LlmModel.embed}; {@link LLM#builder} rejects them.
 *
 * <p><b>Which call.</b> {@link LLM#chat()} / {@link LLM#chatOnce(String)} apply the tokenizer chat
 * template and parse thinking/answer. {@link LLM#complete(String)} continues a raw string with no
 * template. {@link LLM#generate} / {@link LLM#generateTokenIds} are the batch primitives (optional
 * timeout, progress, token stream). {@link LLM#rag} wraps chat with a {@link com.igormaznitsa.nanollvm.rag.RagIndex}.
 *
 * <p><b>Knobs (where to look).</b> How long / how random a reply is:
 * {@link SamplingParams} / {@link SamplingDefaults} (or {@link LLM.Builder#sampling(SamplingParams)}). How much
 * conversation the engine can hold, CPU workers, and memory:
 * {@link LLM.Builder} ({@code maxModelLen}, {@code cpuThreads}, {@code kvHeapFraction}).
 * Per-conversation extras: {@link com.igormaznitsa.nanollvm.chat.ChatSession}. {@link Config} is
 * the sealed copy of those engine limits after {@code build()} — applications rarely construct it.
 * Optional {@link LlmAdvisor}s run as one extra batched generate before the main reply; mix with
 * {@link LlmAdvisorMixer}. See {@link LLM} “If you want…” for a goal → method map.
 *
 * <p><b>I/O and errors.</b> Construction is silent ({@link com.igormaznitsa.nanollvm.chat.LlmListeners#silent()});
 * CLI tools pass {@link com.igormaznitsa.nanollvm.chat.LlmListeners#toSystem()}. One {@link LLM}
 * must not generate concurrently; {@link LLM#cancel()} aborts an in-flight run.
 * Failures are {@link com.igormaznitsa.nanollvm.exceptions} types. {@link GenerationStats} on
 * {@link LLM.GenerationOutput} / {@link com.igormaznitsa.nanollvm.chat.ChatReply} is wall time
 * of the enclosing generate.
 *
 * <p><b>Process control.</b> Default parallel matmul uses a process-wide daemon pool
 * ({@code nanollvm-matmul-*}). Servers should call
 * {@link LLM.Builder#matmulExecutor(java.util.concurrent.ExecutorService)} (caller-owned),
 * {@link LLM.Builder#dedicatedMatmulPool()} (engine-owned, closed with the {@code LLM}), or
 * {@link LLM.Builder#disableMultiCpu()}. See {@link LLM} “Servers, threads, and memory”.
 */

package com.igormaznitsa.nanollvm.llm;
