# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 1.1.0-SNAPSHOT

### Changed
- Maven layout is now multi-module: parent `nano-vllm-java-pom`, library `nano-vllm-java`, demos
  `nano-vllm-java-samples`. Sample mains are no longer packaged in the library JAR; run demos with
  `mvn -pl nano-vllm-java-samples exec:java` from the repository root.

### Added
- **ONNX folder weight import** (Tier A): `LlmModelFactory.make(folder)` loads `config.json` + tokenizer + `.onnx` (root or `onnx/`) like safetensors — Qwen3 / Gemma3 / Llama chat and BERT embeddings; no ONNX Runtime.
- **Llama** causal architecture (`LlamaForCausalLM`) for HF safetensors and ONNX (Tiny-LLM-ONNX base demo;
  SmolLM2-135M-Instruct-ONNX chat demo via `models/download-smollm2-135m-instruct-onnx.sh`).
- Transparent GGUF BERT embedding support (e.g. GTE-small): load via `LlmModelFactory.make` and call `LlmModel.embed(...)` with text or token ids; `LLM.builder` rejects embedding-only models; `Example` menu option runs an embedding REPL.
- Dense / hybrid RAG: `DenseRagIndex`, `HybridRagIndex`, and `RagFactory.withEmbeddings(PreparedRag, LlmModel)` (BM25 + embedding cosine via RRF). `Example` offers a RAG-mode menu (none / BM25 / dense / hybrid) after choosing a chat model.
- RAG classpath documents: `RagFactory.makeResource` / `Builder.addResource` (absolute ClassLoader path or `Class.getResourceAsStream` resolution); source labels use `classpath:…`.
- GGML dequant for Q3_K and IQ4_NL (needed by common small embedding GGUF quants).
- Stream / classpath model load: `ModelFileId` + `ModelFileSource`, `LlmModelFactory.make(source)`, and `fromClasspath` / `fromClasspathGguf` helpers (bytes stay in heap; no disk cache). Filesystem `make(Path)` unchanged.

### Fixed
- Generation stops at `maxModelLen` (and clamps `maxTokens` to remaining context) so short-context models such as Tiny-LLM-ONNX no longer crash RoPE past `max_position_embeddings`.
- ONNX load skips non-float graph constants (e.g. INT64) and scalar initializers so transformers.js exports like SmolLM2 Instruct load cleanly; unknown / float8 / nibble weight types fail with an explicit error instead of silent ignore.
- Decode stops early on degenerate token loops (exact repeated blocks, long same-token streaks, or overused n-grams) so tiny models cannot fill the whole `maxTokens` budget with the same paragraph; Example caps compact ONNX demos (SmolLM2 / Tiny) to 256 new tokens in chat and RAG.

## [1.0.0] — 2026-08-09

First public release of the **nano-vllm-java** CPU inference library
(JPMS module `com.igormaznitsa.nanollvm`, Maven coordinates `com.igormaznitsa:nano-vllm-java:1.0.0`).

### Added
- Pure Java 21+ offline LLM inference: continuous batching, paged KV cache, Hugging Face safetensors and GGUF weights (Qwen3, Gemma3, LFM2).
- Shared models via `LlmModelFactory.make` and per-engine `LLM.Builder` / `LLM` (chat, one-shot, completion, cancel, timeout, generation stats).
- `LlmModel` is `AutoCloseable`: close engines first, then the shared model, to release weight resources; closed models and engines reject further use (`isClosed()` on both).
- Chat sessions with history limits, listeners, optional advisors and mixers, plus lexical BM25 RAG (`RagFactory` / `RagSession`).
- Process-wide `ResourceLimits` for file, PDF, corpus, JSON, GGUF, safetensors, and history budgets (overridable per process or per corpus).
- Configuration knobs including `kvHeapFraction` for heap-based KV auto-sizing, CPU matmul thread control, and `nanollvm.*` / `NANOLLVM_*` system properties and environment variables.
- Documented RAG/advisor trust boundary and optional suppression of prepared-prompt debug events (`emitDebugPrompts(false)`).
- Samples: minimal Gemma `HelloWorld`, log-triage demo, interactive `Example`, and `Bench`.

### Changed
- Explicit `.cpuThreads(N)` / `.disableMultiCpu()` wins over `-Dnanollvm.cpu.threads`; sequential mode creates no matmul executor.
