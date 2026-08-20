# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 1.1.1-SNAPSHOT

### Added
- `LLM.Builder.dedicatedMatmulPool()`: a bounded matmul thread pool owned by that engine and shut
  down on `close()`, so a server need not join the process-wide `nanollvm-matmul-*` pool or hand
  the library a foreign `ExecutorService`. Combining it with `matmulExecutor` fails at `build()`.

- Optional download scripts for [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
  (`models/download-multilingual-e5-small.sh` / `.ps1` / `.cmd` → `models/multilingual-e5-small/`, ONNX fp32 ~470 MB).
  Hugging Face Unigram SentencePiece tokenizers load from `tokenizer.json`; embedding wrap accepts XLM-R
  `<s>` / `</s>` as well as BERT `[CLS]` / `[SEP]`. Optimum-style BERT ONNX (anonymous `onnx::MatMul_*`
  weights) remaps those MatMul aliases through the BERT schema, same as named HF tensors.
  `EmbeddingsHelloWorld` defaults to this folder and prefixes non-retrieval text with `query: `.
  Unigram now applies the Hugging Face precompiled charsmap (needed for many non-ASCII XLM-R / E5
  strings). Folders without `tokenizer.json` load SentencePiece `tokenizer.model`. WordPiece,
  WordLevel, and character models are recognized from `tokenizer.json` `model.type`.
  `Tokenizer.fromSentencePiece` builds a tokenizer from protobuf bytes.

- Load-time RAG tuners on `RagFactory.Builder.addProcessor`: skip files, supply custom document
  text, or rewrite extracted strings before chunking. Several tuners run in order.
  Sample `RagTunerHelloWorld` extracts a bundled Project Gutenberg EPUB of Karel Čapek's *R.U.R.*
  (JDK zip + StAX, no epub4j / xpp3), unpacks gte-small for faster CPU embedding, redraws a single
  in-place percent/ETA bar while indexing, and asks questions from that play. `DenseRagIndex.of` / `HybridRagIndex.of` accept
  a per-passage embed callback and an optional caller `Executor` for parallel indexing (sequential
  when omitted).

### Removed
- Built-in `PdfTextExtractor`. Folder walks no longer pick up `.pdf` by default. Index PDFs (or
  other binaries) with `RagTuner.extracting`, same pattern as the EPUB sample. `ResourceLimits`
  drops PDF inflate / ToUnicode cmap caps (`maxPdfInflateBytes`, `maxCmapRangeSpan`,
  `maxCmapEntries`).

### Fixed
- Closing a model, engine, or weight reader now drops file buffers, KV pages, rotary tables, and
  the last shared matmul pool instead of pinning them until process exit. GGUF and safetensors files
  ≤ 2 GiB are copied into heap so `close()` can reclaim them; shards larger than that still use a
  positioned file channel. Late unpack releases packed bytes when no other engine is using the model.

### Changed
- Dense RAG query embedding stays concurrent (each `LlmModel.embed` uses a fresh step context).
  Index-time passage embedding is sequential unless the caller supplies an `Executor`.
- Extra CPU cores now work on long prompts, not only GEMM: independent attention heads, rotary
  embeddings, token embedding gathers, and Gemma QAT activation scaling run on the same matmul
  pool as linear layers when `cpuThreads > 1`.
- `LlmModel` is a sealed API type. The factory still returns `LlmModel`; the transformer graph
  and engine lease live on a hidden implementation (no static access registry).

## [1.1.0] — 2026-08-16

Public release of **nano-vllm-java** `1.1.0` (Maven coordinates `com.igormaznitsa:nano-vllm-java:1.1.0`).
ONNX weight import, Llama and Gemma 4 text chat, BERT embeddings, Qwen3 GGUF, dense/hybrid RAG, and related API cleanup.

### Changed
- Prepared-prompt `TEXT_DEBUG` events are off unless you call `ChatSession.emitDebugPrompts(true)`
  or `RagSession.emitDebugPrompts(true)` (the previous default dumped mixed RAG/advisor prompts to
  any listener / `streamTo` sink).
- CPU inference kernels are faster on the decode path: linear layers reuse the activation
  vector across outputs (GEMV), Vector-API dots use independent accumulators, and residual /
  MLP / RMSNorm / attention value mix run through SIMD instead of scalar Java loops. Paged KV
  attention reads cache slots in place instead of copying each page into a dense tensor.
- Construct sampling knobs only with `SamplingParams.builder()` (or `SamplingDefaults` / withers).
  Convenience `new SamplingParams()`, two-arg, and three-arg constructors are removed so they cannot
  skip named knobs or disagree with builder defaults (those shortcuts used top-p 0.9).
  `SamplingParams` is a class with a private constructor, same Builder contract as `LLM` /
  `LlmAdvisor`.
- Construct an engine only with `LLM.builder(model).build()`. The `new LLM(model)` shortcut is
  removed so closed and embedding checkpoints cannot skip builder checks.
- Interactive `Example` demo is a **line-oriented terminal** app: model menu, RAG mode
  (none / BM25 / dense / hybrid), then advisor count (`0`–`3`). Enter selects the first item
  (downloaded models first; Qwen3-0.6B preferred for chat quality). If no checkpoint is on disk, the demo exits
  with download instructions. Embedding models skip to an embed REPL. Prepared-prompt debug
  (`debug>` on stderr) is off unless you pass `--debug`. Read `samples.Example` top to bottom —
  each mode is a named method.
- GGUF and ONNX weight loads now redraw the same in-place percent/ETA bar as Hugging Face
  safetensors (current tensor on one line) instead of a tensor list or a late “assembled” summary.
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
  release rather than kept as `@Deprecated` — use `newConversation(String)`,
  `scrubMatchingAssistantTurns`, and `systemFor(Tokenizer)` / `LLM.Builder.systemPrompt`.
- Maven layout is now multi-module: parent `nano-vllm-java-pom`, library `nano-vllm-java`, demos
  `nano-vllm-java-samples`. Sample mains are no longer packaged in the library JAR; run demos with
  `mvn -pl nano-vllm-java-samples exec:java` from the repository root.

### Added
- Fluent load, sampling, and call shortcuts: `LlmModelFactory.open(path).listen(…).unpackParameters().thinkTags(…).chatSpecials(…).make()`
  (existing `make` / `fromClasspath*` remain); `SamplingParams.builder()` and withers; `SamplingDefaults.neutral()`;
  `LLM.Builder.sampling` / `stopTokenIds`; `chatOnce` / `complete` max-token overloads; `generate(…, Duration)` /
  seq-aware `LLM.TokenEvent` callbacks; `ChatSession.seed` / `maxTokens` / `send(text, params)`;
  `ChatReply.parse(raw, llm)` using the model's think tags. `ChatSession` streaming also emits
  `TEXT_RAW` (unparsed tokenizer decode, think tags and chat specials kept) alongside parsed
  thinking/answer.
- `ModelSupport` / `UnsupportedModelException`: exact architecture detection (no substring `qwen`→Qwen3),
  a user-facing support catalog, and fail-fast load errors for look-alike families (Qwen2, Qwen3.5/Fara,
  Gemma 2, Gemma 4 vision/audio-only variants, Mistral, other VLMs, GGUF Llama/Gemma, HF BERT safetensors). `-Dnanollvm.arch` cannot override a
  different family. `LLM.builder` / `LlmModel.embed` misuse messages include the same catalog. Gemma 4 **text** (including QAT mobile) is a supported chat graph.
- `Tokenizer.ChatFormat` / `isTurnBasedChat()` / `skipSpecialTokensOnChatDecode()` (product-named
  chat helpers such as {@code isGemmaChat} removed; use format/architecture APIs).
- RMSNorm offset scale flag renamed to {@code onePlusWeight} (math convention, not a product name).
- `ChatSession` / `RagSession` opt-in `recoverUnusableAnswers` / `unusableAnswer` /
  `unusableAnswerFallback`; `RagSession` also exposes `maxHistoryMessages` and `emitDebugPrompts`
  so RAG chats do not have to drop through `.chat()`.
- `LLM.Builder.advisorNoteFilter` so apps can drop demo setup fillers before advisor mix.
- **ONNX folder weight import** (Tier A): `LlmModelFactory.make(folder)` loads `config.json` + tokenizer + `.onnx` (root or `onnx/`) like safetensors — Qwen3 / Gemma3 / Llama chat and BERT embeddings; no ONNX Runtime.
- Optional download scripts for Gemma 4 E2B QAT mobile (`models/download-gemma4-e2b-qat-mobile.sh` / `.ps1` / `.cmd` → `models/Gemma4-E2B-IT-QAT-Mobile/`, ~2.3 GB). Hugging Face folders with `model_type` `gemma4` / `gemma4_text` now load as **text-only chat** (packed QAT int2/4/8, per-layer embeddings, KV sharing). Vision and audio towers in the same checkpoint are skipped. Safetensors shards larger than 2 GiB are read via `FileChannel` (Java mmap stays limited to 2 GiB).
- **Llama** causal architecture (`LlamaForCausalLM`) for HF safetensors and ONNX (Tiny-LLM-ONNX base demo;
  SmolLM2-135M-Instruct-ONNX chat demo via `models/download-smollm2-135m-instruct-onnx.sh`).
  Sample `NextTokenHelloWorld` encodes a seed and prints the next sampled tokens plus the continued
  text (default Tiny-LLM-ONNX).
- Transparent GGUF BERT embedding support (e.g. GTE-small): load via `LlmModelFactory.make` and call `LlmModel.embed(...)` with text or token ids; `LLM.builder` rejects embedding-only models; `Example` menu option runs an embedding REPL. Sample `EmbeddingsHelloWorld` prints the vector dim, a preview, and cosine vs the same / related / unrelated text (default `models/gte-small.Q2_K.gguf`).
- **Qwen3 GGUF chat:** `LlmModelFactory.make(pathToQwen3.gguf)` loads a self-contained `general.architecture=qwen3` file (embedded tokenizer + packed weights). Load is split into container transport (GGUF / HF safetensors / ONNX) and a per-family architecture processor that binds config/schema, fills weights, and builds the graph. Qwen2, MoE, VL, and Gemma/Llama GGUF stay rejected.
- Dense / hybrid RAG: `DenseRagIndex`, `HybridRagIndex`, and `RagFactory.withEmbeddings(PreparedRag, LlmModel)` (BM25 + embedding cosine via RRF). `Example` offers a RAG-mode menu (none / BM25 / dense / hybrid) after choosing a chat model, then an advisor-count menu (`0`–`3`). Sample `AdvisorRagHelloWorld` shows one custom advisor (Alex) plus BM25 over `rag/` on Gemma3-270M.
- RAG classpath documents: `RagFactory.makeResource` / `Builder.addResource` (absolute ClassLoader path or `Class.getResourceAsStream` resolution); source labels use `classpath:…`.
- GGML dequant for the remaining GGUF weight types: K-quants (`Q2_K`, `Q5_K`, `Q8_K`), legacy (`Q4_1`, `Q5_0`, `Q5_1`, `Q8_1`, `Q1_0`, `Q2_0`), IQ (`IQ1_S`/`M`, `IQ2_*`, `IQ3_*`, `IQ4_XS`), ternary (`TQ1_0`/`TQ2_0`), MXFP4 / NVFP4, and integer/F64 tensors. `Q3_K` / `IQ4_NL` were already present. File recipes like `Q4_K_M` still mix those GGML types (not a separate dtype).
- Stream / classpath model load: `ModelFileId` + `ModelFileSource`, `LlmModelFactory.make(source)`, and `fromClasspath` / `fromClasspathGguf` helpers (bytes stay in heap; no disk cache). Filesystem `make(Path)` unchanged.
- Custom chat scratchpad markers via `ThinkTags` in `LlmModelFactory.make(…, Map)` under
  `LlmModel.OPTION_THINK_TAGS` (default remains `<think>` / `</think>`), and chat-markup search
  strings via `ChatSpecials` under `LlmModel.OPTION_CHAT_SPECIALS` (defaults cover ChatML, Gemma
  turn markers, Llama stops, and the default think pair). Omitted keys are filled with those
  defaults on the frozen options map. `ChatSession.thinkTags` / `RagSession.thinkTags` override the
  scratchpad pair for one conversation. Parse, ChatML skip-seed, and history truncation use the
  same think pair when both markers are in vocab.

### Fixed
- Weightless RMSNorm (no affine scale, used on Gemma 4 shared-KV V) no longer NPEs on the fused
  residual path `forward(x, residual)`.
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
