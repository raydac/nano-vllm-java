# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 1.1.0-SNAPSHOT

### Changed
- Interactive `Example` demo is a **line-oriented terminal** app: model menu, RAG mode
  (none / BM25 / dense / hybrid), then advisor count (`0`–`3`). Enter selects the first item
  (downloaded models first; Qwen3-0.6B preferred for chat quality). If no checkpoint is on disk, the demo exits
  with download instructions. Embedding models skip to an embed REPL. Prepared-prompt debug
  (`debug>` on stderr) is off unless you pass `--debug`. Read `samples.Example` top to bottom —
  each mode is a named method.
- Library chat/sampling/RAG defaults are architecture-marker driven, not product-tuned: `Tokenizer.ChatFormat`
  (ChatML / turn-based / plain), neutral `SamplingDefaults`, no Qwen EOS id fallback, no default
  unknown-arch→Qwen3, no Gemma-only session retry or RAG isolate. Demo policies (system prompts,
  turn-based top-k, unusable-answer recovery, advisor setup-boilerplate filter) live in
  `nano-vllm-java-samples` (`SampleChatPrompts`, `Example`, `HelloWorld`).
- Advisor demo role text, shared advisor instructions, Greek name catalog, and advisor-aware
  system add-on are samples-only (`SampleAdvisorPrompts`). The library uses caller-supplied
  `LlmAdvisor` name/prompt only, plus structural note-mixing helpers.
- Demo advisor role strings and advisor-aware system add-on moved to samples
  (`SampleAdvisorPrompts`); library no longer appends advisor prose to system prompts.
- Library chat defaults no longer inject model-family system prose (Qwen `<think>` rules, plain
  assistant text). `ChatPrompts.systemFor` is always empty; demos set policy via
  `SampleChatPrompts` in `nano-vllm-java-samples`. 1.0 boolean chat shims
  (`ChatMessages.newConversation(boolean)`, `scrubSetupBoilerplateTurns`,
  `ChatPrompts.systemFor(boolean)` / `systemFor(Tokenizer, boolean)`) are removed in this
  unreleased line rather than kept as `@Deprecated` — use `newConversation(String)`,
  `scrubMatchingAssistantTurns`, and `systemFor(Tokenizer)` / `LLM.Builder.systemPrompt`.
- Maven layout is now multi-module: parent `nano-vllm-java-pom`, library `nano-vllm-java`, demos
  `nano-vllm-java-samples`. Sample mains are no longer packaged in the library JAR; run demos with
  `mvn -pl nano-vllm-java-samples exec:java` from the repository root.

### Added
- `ModelSupport` / `UnsupportedModelException`: exact architecture detection (no substring `qwen`→Qwen3),
  a user-facing support catalog, and fail-fast load errors for look-alike families (Qwen2, Qwen3.5/Fara,
  Gemma 2, Mistral, vision, GGUF Qwen/Llama, HF BERT safetensors). `-Dnanollvm.arch` cannot override a
  different family. `LLM.builder` / `LlmModel.embed` misuse messages include the same catalog.
- `Tokenizer.ChatFormat` / `isTurnBasedChat()` / `skipSpecialTokensOnChatDecode()` (product-named
  chat helpers such as {@code isGemmaChat} removed; use format/architecture APIs).
- RMSNorm offset scale flag renamed to {@code onePlusWeight} (math convention, not a product name).
- `ChatSession.recoverUnusableAnswers` / `unusableAnswer` / `unusableAnswerFallback` (opt-in).
- `LLM.Builder.advisorNoteFilter` so apps can drop demo setup fillers before advisor mix.
- **ONNX folder weight import** (Tier A): `LlmModelFactory.make(folder)` loads `config.json` + tokenizer + `.onnx` (root or `onnx/`) like safetensors — Qwen3 / Gemma3 / Llama chat and BERT embeddings; no ONNX Runtime.
- **Llama** causal architecture (`LlamaForCausalLM`) for HF safetensors and ONNX (Tiny-LLM-ONNX base demo;
  SmolLM2-135M-Instruct-ONNX chat demo via `models/download-smollm2-135m-instruct-onnx.sh`).
- Transparent GGUF BERT embedding support (e.g. GTE-small): load via `LlmModelFactory.make` and call `LlmModel.embed(...)` with text or token ids; `LLM.builder` rejects embedding-only models; `Example` menu option runs an embedding REPL.
- Dense / hybrid RAG: `DenseRagIndex`, `HybridRagIndex`, and `RagFactory.withEmbeddings(PreparedRag, LlmModel)` (BM25 + embedding cosine via RRF). `Example` offers a RAG-mode menu (none / BM25 / dense / hybrid) after choosing a chat model, then an advisor-count menu (`0`–`3`).
- RAG classpath documents: `RagFactory.makeResource` / `Builder.addResource` (absolute ClassLoader path or `Class.getResourceAsStream` resolution); source labels use `classpath:…`.
- GGML dequant for Q3_K and IQ4_NL (needed by common small embedding GGUF quants).
- Stream / classpath model load: `ModelFileId` + `ModelFileSource`, `LlmModelFactory.make(source)`, and `fromClasspath` / `fromClasspathGguf` helpers (bytes stay in heap; no disk cache). Filesystem `make(Path)` unchanged.

### Fixed
- Windows model download scripts (`.ps1` / `.cmd`) resolve their install dir via `$PSScriptRoot`,
  write with absolute paths (no longer change the caller’s working directory), prefer `curl.exe`,
  stay ASCII-only for Windows PowerShell 5.1 encoding, and the Gemma script also honors
  `HF_HOME\token`.
- Lexical RAG off-topic detection no longer treats conversational fillers (`what` / `think` /
  `about` / …) as topic evidence, so queries like “what do you think about BMW?” stay outside
  fairy-tale corpora even when those glue words appear in the documents.
- ChatML models without `<think>` vocab tokens no longer set `Tokenizer.invitesThinking()`; think
  invitation is vocab-gated for HF and GGUF loads alike. Library system prompts stay empty
  regardless (demo policies live in samples).
- Generation stops at `maxModelLen` (and clamps `maxTokens` to remaining context) so short-context models such as Tiny-LLM-ONNX no longer crash RoPE past `max_position_embeddings`.
- ONNX load skips non-float graph constants (e.g. INT64) and scalar initializers so transformers.js exports like SmolLM2 Instruct load cleanly; unknown / float8 / nibble weight types fail with an explicit error instead of silent ignore.
- Decode stops early on degenerate token loops (exact repeated blocks, long same-token streaks, or overused n-grams) so tiny models cannot fill the whole `maxTokens` budget with the same paragraph; Example caps compact ONNX demos (SmolLM2 / Tiny) to 256 new tokens in chat and RAG.
- `BundledModels.find` accepts absolute filesystem paths (it no longer strips a leading `/` before the absolute-path check).

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
