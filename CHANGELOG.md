# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 1.3.0-SNAPSHOT

### Added
- Load-time extras via typed keys: `LlmModelFactory.open(path).optionalData(key, value).make()`.
  Unknown keys are kept and ignored by graphs that do not read them. Empty extras stay off
  `LlmModel.options()` so existing think-tag / chat-special maps are unchanged. Piper voices
  use `LlmOptionalData.ESPEAK_DATA` for the espeak-ng-data directory (default
  `{model}/espeak-ng-data`). A missing or incomplete folder is ignored; Piper still
  synthesizes. When the folder includes `dictsource` (`*_list` / `*_rules`) for the
  voice language, Piper reads those files for G2P instead of the built-in letter tables.
  Suffix and prefix `S`/`P` rules retranslate the stem; digits use the list's number
  fragments (`_1`, `_2X`, `_0C`, …). Russian `*_rules` honor the A/B/C/F/G/H/Y letter
  groups (so `и` after a hard consonant is `ы`, not after every consonant) and then
  palatalize before `е`/`и`, reduce unstressed `о`/`е`, and place stress from
  compiled `{lang}_dict` / `$1`–`$7` list flags (otherwise espeak's Russian
  syllable-count guess, so two-syllable words like `это`/`мама` are not
  end-stressed). That stops Irina stressing every unknown word on the final
  syllable, which sounded Slavic but not Russian. If `dictsource` is missing but compiled
  `phontab` and `{lang}_dict` are present, listed words still come from those files.

- Piper text-to-speech from a voice folder (`*.onnx` + `*.onnx.json`). Load with
  `LlmModelFactory.make`, then `LLM.builder(model).build()` and `LLM.synthesize` for
  uncompressed WAV bytes (PCM16 LE mono) — write the array only if you need a file.
  `LlmModel.synthesize` remains a sequential shortcut.
  Official Piper ONNX exports (including ru_RU-irina-medium) load: the phoneme table exported as
  `sid`, WaveNet kernels named `onnx::Conv_*`, and HiFi-GAN vocoder residual blocks. Conv and
  ConvTranspose geometry (stride, padding, dilation, groups, output padding) comes from the ONNX
  graph instead of vocoder-family guesses, so voices are not locked to Lessac/Irina kernel sizes.
  Irina uses ResBlock2 (`convs.0`/`convs.1`) with those graph dilations, so output is speech rather
  than a metallic buzz. Reverse synthesis follows VITS: all residual couplings with channel flips,
  WaveNet skip paths, and stochastic duration (rational-quadratic conv flows).
  espeak-ng-data is optional (compiled `phontab` / `{lang}_dict` are used when
  `dictsource` is absent). Download scripts
  install `lang/` plus `dictsource/` (`*_list` / `*_rules`) and, when a system
  espeak-ng-data is present, compiled `phontab` / `phondata` / `{lang}_dict` so Russian
  lexical stress is available. The dictionary is the G2P when
  present. Otherwise a letter-to-sound fallback is used: Russian uses espeak phones for ш/ж/ы (`ʃ`/`ʒ`/`y`), so those
  consonants stay audible (academic IPA `ʂ`/`ʐ`/`ɨ` are unused by this voice). Piper voices map the voiced velar to
  IPA `ɡ` (not ASCII `g`); G2P emits `ɡ` so Russian `г` is spoken instead of dropped. English Lessac G2P maps
  espeak `@2`/`@5` to schwa (no leaked digit `2`), and an en-us post-pass turns British-leaning phones into Piper
  rhotic IPA (`ɹ`/`ɚ`/`ɑː`) with content-word stress so pangrams stay close to official Piper.
  Russian letter G2P also destresses word-final obstruents, assimilates voicing across clusters
  (including prepositions), and applies common orthoepic rewrites (жи/ши, -ого/-его, что, -ться)
  while still emitting Piper espeak phones. Latin letter G2P uses a small English lexicon plus
  letter/digraph rules so a US English Piper voice still works without dictionaries. Optional
  `models/download-piper-en-lessac-medium.sh` / `.ps1` / `.cmd` (Lessac medium + espeak-ng-data)
  and `models/download-piper-ru-irina-medium.sh` / `.ps1` / `.cmd` (Irina medium + espeak-ng-data).
  Sample `SynthesizeHelloWorld` prefers Lessac when both folders exist; `Example` lists both
  voices and the TTS session suggests `Hello world` / `Привет, мир`.
  `LLM.builder` is the runtime for every graph kind: Piper uses the same `cpuThreads` / matmul
  pool as chat, and 1-D conv / conv-transpose split independent output channels across that pool.
  Non-chat engines skip KV paging (`numKvcacheBlocks` is 0), so Piper no longer fails to build
  on a large heap (the chat auto-sizer treated a missing transformer as 1-byte pages and overflowed).

- Whisper speech-to-text from Hugging Face safetensors (`openai/whisper-*`). Load with
  `LlmModelFactory.make`, then `LLM.builder(model).build()` and `LLM.transcribe` on uncompressed
  WAV bytes, a WAV file, or 16 kHz-resampled PCM (no file required when the payload is already
  in memory). Optional language is a `Locale` (`null` / `Locale.ROOT` = auto; region ignored;
  ISO aliases such as `jv` map to Whisper's `jw` token). `LlmModel.transcribe` remains a sequential shortcut. The builder
  uses the same CPU matmul pool as chat (Linear, attention, and stem convs). CTranslate2 /
  faster-whisper `model.bin` folders, Whisper GGUF, and Whisper ONNX are refused. Optional
  `models/download-whisper-base.sh` / `.ps1` / `.cmd` (~290 MB) and `download-whisper-tiny.sh`
  (~150 MB). Sample `TranscribeHelloWorld`; `Example` lists Whisper in the menu and opens a WAV
  transcribe session.

- Optional download scripts for [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
  (`models/download-xlm-roberta-base.sh` / `.ps1` / `.cmd` → `models/xlm-roberta-base/`, ONNX fp32
  saved as `onnx/model.onnx`, ~1.9 GB). BERT-encoder embeddings (`bert` / `roberta` / `xlm-roberta`,
  and the same graph under those family names) load from GGUF or ONNX via `LLM.builder` then
  `LLM.embed` (or `LlmModel.embed` as a sequential shortcut) — the
  library keys off architecture, not a named Hub checkpoint. ONNX BERT weight names drop whatever
  module sits in front of `embeddings.` / `encoder.` (not a fixed prefix list).
  `ModelSupport.isEmbeddingCheckpoint` classifies a folder or GGUF from `config.json` / metadata
  without loading weights. The `Example` demo lists the optional xlm-roberta-base download in its
  menu, picks any BERT encoder under `models/` for dense/hybrid RAG (smallest by weight file), and
  can few-shot classify on encoder vectors (centered prototypes: teach `label | text`, then predict).
  `RagFactory.withEmbeddings` can report per-passage embed progress (and run those embeds on a
  caller `Executor`). DistilBERT / ALBERT / DeBERTa / ELECTRA stay unsupported; safetensors
  BERT-family folders are still rejected.

### Fixed
- Chat answers no longer keep a typed ChatML lookalike `<|im_ended|>` at the end of a turn
  (`ChatSpecials.DEFAULT`). Generation still stops only on real EOS ids such as `<|im_end|>`.

- Model download scripts treat HTTP 416 on resume as "already complete", so re-running a script
  after a sidecar such as `config.json` finished no longer fails (`curl: (22) ... 416`).

## [1.2.0] — 2026-08-22

Public release of **nano-vllm-java** `1.2.0` (Maven coordinates `com.igormaznitsa:nano-vllm-java:1.2.0`).
SentencePiece and extra tokenizer families, RAG load tuners, engine-owned matmul pool, deterministic sampling, checkpoint modalities, and `close()` resource reclaim.

### Added
- `LlmModel.modalities()` (and `inputModalities()` / `outputModalities()`): input and output
  content types as `LlmModality` (`TEXT`, `IMAGE`, `AUDIO`, `VIDEO`, `EMBEDDING`). Values follow
  the checkpoint config (Gemma 4 QAT mobile declares text+image+audio+video in). Embedding
  encoders are text→embedding. `LlmModel.usableModalities()` is what this library actually runs
  (text→text chat, or text→embedding); vision/audio towers are still skipped at load.
  `LlmModel.toString()` includes `modalities=` and `usable=` when they differ. The `Example`
  demo prints checkpoint modalities after load, plus the runtime line when they are not the same.

- RAG user turns with retrieved passages now start with `RagPrompts.GROUNDING` so small chat models
  are told to answer from those lines only and not invent a book or play.

- `HybridRagIndex.of(RagIndex…)` fuses any two or more indexes with RRF (nested hybrids flatten).
  BM25+dense via `RagFactory.withEmbeddings` is unchanged. To mix disk folders, classpath files,
  and inline strings, add them on one `RagFactory.builder()` — `HybridRagIndex` does not concatenate
  corpora. `Builder.addFolders` walks several directories in one call.

- `LLM.Builder.deterministic()` (also `SamplingParams` / `ChatSession` / `RagSession`): same prompt
  always picks the highest-logit token (`topK = 1`, nucleus off). `temperature(0)` stays rejected.

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
  (JDK zip + StAX, no epub4j / xpp3), keeps the short OPF title (before a subtitle slash) as a
  Markdown heading so later chunks still name the play, indexes with BM25, and asks questions from that play with
  `LLM.Builder.deterministic()` so repeats pick the same tokens. `DenseRagIndex.of` / `HybridRagIndex.of` accept
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
  pool as linear layers when `cpuThreads > 1`. Causal attention splits that work head-major so
  each worker covers a full query range (later tokens attend more keys; query-major chunks
  left the last worker with most of the prefill).
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
