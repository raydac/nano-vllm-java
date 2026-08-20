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
 * <p><b>Knobs.</b> {@link SamplingParams} / {@link SamplingDefaults} control next-token draws
 * (temperature, top-k, top-p). {@link Config} is the sealed engine layout (context length, KV
 * pages, stop ids) built with the {@code LLM}. Optional {@link LlmAdvisor}s run as one extra
 * batched generate before the main reply; mix with {@link LlmAdvisorMixer}.
 *
 * <p><b>I/O and errors.</b> Construction is silent ({@link com.igormaznitsa.nanollvm.chat.LlmListeners#silent()});
 * CLI tools pass {@link com.igormaznitsa.nanollvm.chat.LlmListeners#toSystem()}. One {@link LLM}
 * must not generate concurrently; {@link LLM#cancel()} aborts an in-flight run.
 * Failures are {@link com.igormaznitsa.nanollvm.exceptions} types. {@link GenerationStats} on
 * {@link LLM.GenerationOutput} / {@link com.igormaznitsa.nanollvm.chat.ChatReply} is wall time
 * of the enclosing generate.
 */

package com.igormaznitsa.nanollvm.llm;
