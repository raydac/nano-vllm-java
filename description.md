# How a Language Model Works

### An introductory guide to this Java inference engine (Nano-vLLM)

This guide is written for a reader with general academic habits — careful definitions, structured argument, and a
willingness to learn technical vocabulary — but **without** assuming prior machine-learning or systems coursework.

You do not need fluency in Java. Linear algebra appears where it clarifies structure (vectors, matrices, shapes); each
term is defined on first use. Occasional metaphors remain only as scaffolding, never as substitutes for the concepts.

The subject is a small program that **loads a pretrained model** and runs **inference** on an ordinary CPU, without
CUDA or other native backends: continuing text or short chat with a **causal** language model; (**since 1.1.0**)
encoding sentences to vectors with a **BERT-family embedding** GGUF or ONNX; (**since 1.3.0**) transcribing speech
with **Whisper** safetensors; and synthesizing speech with **Piper** ONNX voices. Java package / JPMS module:
`com.igormaznitsa.nanollvm`. This guide matches development line **1.3.0-SNAPSHOT** (after released **1.2.0**,
2026-08-22).

Where a topic has a standard paper or format specification, a short **Further reading** note lists links. Those are
optional depth; the exposition here is self-contained. A curated **learning path** (what to read in which order) is
**chapter 21**; **chapter 20** is a flat bookmark index of the same family of links.

Where a topic has a home in this library, an **In the code** note names the Java types and methods that implement it
(package `com.igormaznitsa.nanollvm`). Those notes are signposts — **chapter 16** collects the full map, samples, and
file paths.

---

## Table of contents

1. [What this is really about](#1-what-this-is-really-about)
2. [The one trick: guessing the next piece of text](#2-the-one-trick-guessing-the-next-piece-of-text)
3. [Cutting language into pieces the machine can count](#3-cutting-language-into-pieces-the-machine-can-count)
4. [Loading a model: opening the library box](#4-loading-a-model-opening-the-library-box)
5. [`config.json` — the blueprint field by field](#5-configjson--the-blueprint-field-by-field)
6. [`tokenizer.json` — the dictionary file field by field](#6-tokenizerjson--the-dictionary-file-field-by-field)
7. [Tensors and `*.safetensors` — parameters on disk](#7-tensors-and-safetensors--parameters-on-disk)
7a. [GGUF — quantized single-file models (Qwen3 and LFM2)](#7a-gguf--quantized-single-file-models-qwen3-and-lfm2)
7b. [BERT embedding GGUFs — sentence vectors (**since 1.1.0**)](#7b-bert-embedding-ggufs--sentence-vectors-since-110)
7c. [ONNX weight folders and Llama (**since 1.1.0**)](#7c-onnx-weight-folders-and-llama-since-110)
7d. [Whisper speech-to-text (**since 1.3.0**)](#7d-whisper-speech-to-text-since-130)
7e. [Piper text-to-speech (**since 1.3.0**)](#7e-piper-text-to-speech-since-130)
8. [Attention: kinds of looking-back, and how they work](#8-attention-kinds-of-looking-back-and-how-they-work)
9. [The thinking process: how it is organized and how it works with the model](#9-the-thinking-process-how-it-is-organized-and-how-it-works-with-the-model)
10. [Tensors, embeddings, and the arithmetic of inference](#10-tensors-embeddings-and-the-arithmetic-of-inference) —
    **math catalog** (linear, RMSNorm, attention, RoPE, MLP, sampling)
11. [Choosing a word: not always the most obvious one](#11-choosing-a-word-not-always-the-most-obvious-one)
12. [Why the program keeps a notebook of the past](#12-why-the-program-keeps-a-notebook-of-the-past)
13. [Serving several conversations without chaos](#13-serving-several-conversations-without-chaos)
14. [Chat versus finishing a sentence](#14-chat-versus-finishing-a-sentence)
15. [A full walk-through: “What is 2+2?”](#15-a-full-walk-through-what-is-22)
16. [Where it lives in the code (classes, methods, samples)](#16-where-it-lives-in-the-code-classes-methods-samples)
17. [Text RAG: documents beside the model](#17-text-rag-documents-beside-the-model)
18. [Word list](#18-word-list)
19. [Honest limits](#19-honest-limits)
20. [External reading index](#20-external-reading-index)
21. [Suggested literature (learning path)](#21-suggested-literature-learning-path)

---

## 1. What this is really about

Imagine you open a book that has been **already written and bound**. You are not writing that book. You are only
**reading from it** to help you invent the next line of a story or the next reply in a dialogue.

That is what this program does.

- Someone else (large labs, long training runs) produced the “book” — the **model**.
- This project only **opens** that book and **uses** it.
- It does not learn new world knowledge while you chat. It applies what is already stored in the files.

In one sentence:

> **You give it text; it repeatedly chooses a plausible next scrap of text until the answer feels finished.**

The rest of this guide unpacks that sentence without assuming prior ML coursework.

**In the code:** front door is `LLM` / `LLM.Builder`; interactive demo is `samples.Example`; linear mains include
`HelloWorld`, `NextTokenHelloWorld`, `LogTriageHelloWorld`, `AdvisorRagHelloWorld`,
`RagTunerHelloWorld`, `EmbeddingsHelloWorld`, `TranscribeHelloWorld`, `SynthesizeHelloWorld` (chapter 16).

---

## 2. The one trick: guessing the next piece of text

A language model is not a database that “looks up” Paris when you mention France — though it often *behaves* as if it
had looked something up.

Its native skill is narrower and stranger:

> Looking at everything written so far, score **every** possible next scrap in its vocabulary, then pick one.

So if the text so far is:

> The capital of France is

the model does not “search Europe.” It asks, for tens of thousands of possible next pieces (“ Paris”, “ Lyon”, “ a”, “
the”, …): *how strongly does this continuation fit?* Then it chooses.

That choice becomes part of the text. Then it asks again. And again.

```text
  “The capital of France is”
           │
           ▼
   score every possible next scrap
           │
           ▼
   pick one  →  “ Paris”
           │
           ▼
  “The capital of France is Paris”
           │
           ▼
   score again → pick → …
```

**Training** (teaching the model on huge amounts of text) already happened elsewhere.  
**Inference** (using the finished model to produce new text) is what this project does — like performing a play from a
finished script, not writing the script.

**Further reading:** the modern stack grows from the Transformer architecture —
[Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762); a line-by-line walkthrough
is [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

Two families of Hugging Face checkpoints were the original teaching crates (different “editions” of the book, same kind
of reading process): **Qwen3** and **Gemma3**. **Since 1.1.0** the same causal path also loads **Llama**-style graphs
(including small ONNX demos) and **Gemma 4 text** (QAT mobile) — vision and audio towers in that crate are skipped.
The same Qwen3 graph also loads from a **GGUF** file (**since 1.1.0**). A further GGUF path loads **LFM2**
(hybrid short-convolution + GQA). Sentence embeddings use **BERT**-family GGUF or ONNX (**since 1.1.0**, chapter 7b).
**Since 1.3.0**, the same factory also loads **Whisper** speech-to-text (Hugging Face safetensors, chapter **7d**) and
**Piper** text-to-speech (ONNX voice folder, chapter **7e**). You usually need not care which; the program detects
which files you pointed it at.

**In the code:** architecture pick is `ArchitectureProcessors.of` → `Qwen3ForCausalLM`,
`Gemma3ForCausalLM`, `Gemma4ForCausalLM`, `LlamaForCausalLM`, or `Lfm2ForCausalLM`; GGUF entry is `LlmModelFactory` →
`models.llmcontainer.GgufTransport` (container catalog) → `models.llmarch.ArchitectureProcessor.bind` / `fill` / `create`;
ONNX folders use `models.llmcontainer.OnnxTransport` → the same architecture processor (chapter **7c**); Whisper uses
`SpeechArchitecture` / `WhisperForAsr` (chapter **7d**); Piper uses `SynthesisArchitecture` / `PiperForTts`
(chapter **7e**); one next-token
step is `Transformer.step` → `CausalLM.forward` / `computeLogits` → `Sampler.forward` (chapter 16). Linear demo:
`samples.NextTokenHelloWorld` (Tiny-LLM-ONNX) encodes a seed and prints the next sampled tokens.

---

## 3. Cutting language into pieces the machine can count

Computers do not “see” letters the way we do. Before any clever reading happens, your sentence is broken into
**tokens**: small chunks of text — sometimes a whole word, sometimes part of a word, sometimes punctuation.

Think of a **special dictionary** that knows not only whole words but also common fragments (“ing”, “tion”, pieces of
names). Encoding means: look up the sentence in that dictionary and write down a list of **numbers** (the page numbers
of those scraps). Decoding means: turn the numbers back into readable text.

```text
  “Hello”  →  [numbers]  →  inside the model  →  [more numbers]  →  “Hi there”
```

Why bother? Because the model’s entire inner life is **arithmetic on lists of numbers**. Tokens are the bridge between
human language and that arithmetic. The next bridge after ids is the **embedding** table: each id becomes a vector of
length `hidden_size` (chapter 10).

A separate file, the **tokenizer**, holds that dictionary and the rules for chopping text. Chat models also store a
**template**: stage directions such as “this line is the user,” “this line is the assistant,” so the model is not
confused about who is speaking. Without those markers, a dialogue looks like an undifferentiated blob of prose.
**Chapter 6** opens `tokenizer.json` field by field.

**In the code:** `tokenizer.Tokenizer` — `fromPretrained` / `fromGguf` / `fromSentencePiece` (**since 1.2.0**),
`encode`, `decode`, `applyChatTemplate` (chapter 16). Hugging Face folders load `tokenizer.json`, else
SentencePiece `tokenizer.model`, else a tiny `config.json` fallback.

**Further reading:** subword BPE was popularized for translation in
[Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909);
byte-level BPE is described with GPT-2 in OpenAI’s
[*Language Models are Unsupervised Multitask
Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Chat
wrapping is documented in Hugging Face’s
[chat templating guide](https://huggingface.co/docs/transformers/chat_templating).


---

## 4. Loading a model: opening the library box

“Loading a model” sounds like plugging in a mind. It is closer to **unpacking a crate** and arranging its contents on a
desk until the program can use them.

### Where the crate comes from

You (or a download script in this project’s `models/` folder) fetch weights from a public model hub. This port accepts
**three** crate shapes — do not mix their stories:

| Crate shape | What you point at | Typical cargo |
|-------------|-------------------|---------------|
| **Hugging Face folder** | A directory | `config.json` + tokenizer (`tokenizer.json`, else SentencePiece `tokenizer.model` **since 1.2.0**) + `*.safetensors` **or** supported `*.onnx` (Qwen3 / Gemma3 / **Gemma 4 text** / **Llama**; BERT embeddings from ONNX; **Whisper** speech safetensors **since 1.3.0**) — **since 1.1.0** for ONNX and Gemma 4. **Piper** voices are a folder of `*.onnx` + `*.onnx.json` (**since 1.3.0**, chapter **7e**). |
| **Single GGUF file** | One `.gguf` | Metadata + embedded tokenizer + quantized weights (**Qwen3** or **LFM2** chat, or **BERT** embeddings — not Gemma/Llama GGUF) |
| **Stream / classpath** (**since 1.1.0**) | A `ModelFileSource` (or `fromClasspath*` helpers) | Same roles as above, but bytes come from streams into **heap** (no disk cache); `make(Path)` stays direct disk I/O |

**Format restrictions are real:** each crate shape accepts only a **subset** of hub files (dtypes, architectures,
file-name filters). Chapters **7** / **7a** / **7c** / **7d** / **7e** spell out what loads and what fails.

Hugging Face folder example:

```text
models/Qwen3-0.6B/          (example)
  config.json               ← blueprint of the building
  tokenizer.json            ← dictionary + chat stage directions
  model-….safetensors       ← the heavy crates of learned numbers
  (sometimes more shards, license notes, extras…)
```

Until the chosen crate is **read into memory and wired into empty structure**, the program cannot answer anything.
Loading is that wiring.

### Three kinds of cargo (do not mix them up)

**1. The blueprint (`config.json` or Piper `*.onnx.json`)**  
A list of measurements: how many stacked “reading rooms” (layers), how wide each stream of numbers is, how many
attention heads, how long a passage may be, which recipe (Qwen vs Gemma vs Whisper vs Piper), and similar.

This is like an architect’s plan. It does **not** contain opinions about France or arithmetic. It only tells the program
how large the furniture must be. **Chapter 5** explains every field this project reads, with real Qwen and Gemma
examples.

**2. The dictionary (`tokenizer.json`, or `tokenizer.model`, and friends)**  
How human text becomes token numbers and back, plus special markers for “user” and “assistant.”  
This is the spelling system and the stage-direction language — still not the “knowledge.” **Chapter 6** details
`tokenizer.json` (and friends such as `tokenizer_config.json`); folders without JSON load SentencePiece
`tokenizer.model` (**since 1.2.0**).

**3. The learned numbers (`*.safetensors` or supported `*.onnx`)**  
The big cargo. Millions or billions of numbers shaped by training. These are what make one model sound different from
another. **Chapter 7** (safetensors) and **chapter 7c** (ONNX **since 1.1.0**) explain layouts, dtypes, and what is
*not* accepted. GGUF packs the same role into one file (**chapter 7a**). Whisper uses the safetensors crate
(chapter **7d**). Piper uses ONNX initializers plus a sidecar JSON (chapter **7e**).

On disk they are often stored in a compact form (half-precision, GGUF block quants, or packed QAT). Hugging Face **float**
crates are **widened to ordinary float32 in RAM**. **Gemma 4 text QAT** safetensors stay **packed** (int2/4/8 + SRQ) and
dequantize on use, like GGUF. GGUF chat/embedding weights **stay packed** by default and dequantize a
row during matmul (chapter 7a). Easy to teach; still hungry for memory (activations and KV are float32 either way).

### What loading does, step by step

Think of a librarian preparing a reading desk:

1. **Read the blueprint**  
   The program opens `config.json` (Hugging Face folder) or GGUF metadata (`qwen3.*` / `lfm2.*` keys) and learns the
   measurements.

2. **Build empty furniture**  
   It constructs the stack of reading rooms, empty weight shelves, empty embedding card-index, empty final scoring
   table — all the right *shapes*, still filled with zeros or placeholders.

3. **Open the heavy crates**  
   Each `.safetensors` file begins with a catalog (tensor name → where the bytes live). The loader walks that catalog.
   A `.gguf` file carries the same catalog idea in binary form (chapter 7a).

4. **Match names to shelves**  
   A name like “layer 7’s output mix” must land in layer 7’s output-mix shelf. Hugging Face Qwen folders often ship
   fused Query/Key/Value packs that this project **merges into one wider shelf** while loading. **GGUF Qwen3** keeps
   those tensors **unfused** (`blk.N.attn_q` / `attn_k` / `attn_v`) so packed quants stay packed. Some models **share**
   the card-index with the final scoring table (one physical shelf, two jobs).

5. **Skip what does not belong**  
   Extra files or unrecognized names are ignored. Only registered shelves get filled.

6. **Open the dictionary**  
   Tokenizer files (or the GGUF-embedded vocab) are loaded separately. They never become part of the neural arithmetic;
   they stand at the door, translating.

Those six steps seal an immutable **`LlmModel`** (weights + tokenizer + blueprint). Opening a runtime **engine** is a
second, separate act:

7. **Lay out blank notebooks** (when you build an `LLM` for **chat**)  
   Memory pages for Keys and Values during conversation are **created empty** on each chat `LLM`. They are not downloaded
   knowledge; they are scratch paper for that engine’s current text. Share one `LlmModel` across many engines; each
   engine gets its own notebooks. Embedding, Whisper, and Piper engines skip KV paging (`numKvcacheBlocks` is 0) and
   still share the same CPU matmul pool.

8. **Optional warm-up** (also on `LLM.Builder`, **off by default**)  
   A tiny pretend question may run once so the first real answer is not also paying start-up costs. Call
   `.warmup()` if you want it; `LlmModelFactory.make` never warms up by itself.

After load, the **learned shelves stay fixed**. Chat does not rewrite the model files. Only the notebooks and temporary
worksheets change while answering. Close order: close each **`LLM` first**, then the shared **`LlmModel`**.

**In the code:** `LlmModelFactory.make` (Path, `ModelFileSource`, or `fromClasspath*`) seals an immutable `LlmModel`.
`LlmModel.toString()` prints kind, modalities, architecture, container, sizes, and packed/dense/qat (safe after close).
`LlmModel.modalities()` follows the checkpoint (`LlmModality` enum). Chat graphs are at least
text→text; Gemma 4 QAT mobile also declares image, audio, and video input. Embedding encoders are
text→embedding. Whisper is audio→text; Piper is text→audio. `LlmModel.usableModalities()` is what this library
actually runs (vision/audio towers on Gemma 4 stay skipped). Optional `Map` load options (**since 1.1.0**) include `LlmModel.OPTION_THINK_TAGS` (`ThinkTags`) and
`LlmModel.OPTION_CHAT_SPECIALS` (`ChatSpecials`; both frozen on the model, library defaults when omitted).
**Since 1.3.0**, `LlmModelFactory.open(path).optionalData(key, value)` stores typed extras (Piper
`LlmOptionalData.ESPEAK_DATA`). Empty extras stay off `options()`.
A non-silent `LlmListener` sees the same in-place percent/ETA bar while **safetensors**, **GGUF**, or **ONNX** weights
are poured (`models.llmcontainer.LoadProgress`). Causal graphs use `ArchitectureProcessor.createCausal`; embedding GGUFs
use `ArchitectureProcessor.createEmbedding`; Whisper uses `createSpeech`; Piper uses `createSynthesis`. Each
`LLM.builder(model).build()` is the runtime for every kind — chat allocates a KV arena via `Transformer`;
embed / transcribe / synthesize skip KV and still use the engine matmul pool
(chapter 16).

```java
try (LlmModel model = LlmModelFactory.open(Path.of("models/Qwen3-0.6B")).make();
     LLM llm = LLM.builder(model).maxModelLen(2048).build()) {
  String answer = llm.chat(256).send("What is 2+2?").answer();
} // close LLM first, then LlmModel (try-with-resources reverse order)
```

**Further reading:** Hub layout and `from_pretrained`-style folders are covered in Hugging Face
[Transformers docs](https://huggingface.co/docs/transformers); this Java port is inspired
by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and the serving ideas in
[vLLM / PagedAttention](https://arxiv.org/abs/2309.06180).

```text
  disk crate                    memory after LlmModelFactory.make
  ─────────                    ────────────────────────────────
  blueprint          ──►       sizes & recipe chosen
  dictionary         ──►       encode / decode ready
  weight crates      ──►       full shelves (fixed)
  (nothing yet)      ──►       (KV notebooks appear later, per LLM)
```

### What each shelf is *for* when answers begin

| Shelf of numbers          | Everyday job while answering                                          |
|---------------------------|-----------------------------------------------------------------------|
| Embedding card-index      | Turn each token number into a rich “portrait” (a long list of traits) |
| Norm scales               | Keep those portraits from becoming absurdly loud or tiny              |
| Query / Key / Value mixes | Prepare three views of each place in the text for attention           |
| Attention output mix      | Blend several parallel “glances” back into one stream                 |
| Feed-forward (MLP) block  | A large private rewrite of each portrait — much of the model’s bulk   |
| Final vocabulary scorer   | Turn the last portrait into “how good is each next token?”            |

If loading failed halfway, you would have a building with missing furniture: some rooms empty, answers nonsense or
crashes. A successful **model** load means: **every expected shelf has its numbers**, dictionary ready. A successful
**engine** open means notebooks are allocated for that `LLM`.

**In the code (shelves):** `ContainerTransport` (`SafetensorsTransport` / `OnnxTransport` / `GgufTransport`) +
`ArchitectureProcessor` (bind catalog → fill `WeightBag` → create graph). Notebooks via per-`LLM` `KvCacheArena` bound into `internal.Context` for
`Attention` (chapter 16).

### Why the first load feels slow and heavy

- The crates are large (often gigabytes).
- Every number may be expanded to a fuller format in RAM.
- The program must touch essentially the whole model once before it can answer.

That cost is paid **at open time**, not on every word (the notebooks exist precisely so later words are cheaper).

---

## 5. `config.json` — the blueprint field by field

The file `config.json` sits in the model folder. It is ordinary JSON: named fields and values. It does **not** store the
learned knowledge. It stores the **measurements and recipe** so this program can build empty furniture of the right size
and choose Qwen-like vs Gemma-like behaviour.

This chapter explains the fields **this project actually reads** (via `Config.HfConfig`), what they mean in plain
language, and how they show up in real models such as Qwen3-0.6B and Gemma3-270M. Extra keys that appear in hub files
but are ignored here are listed at the end.

### What the file looks like (tiny sketch)

```json
{
   "model_type": "qwen3",
   "architectures": [
      "Qwen3ForCausalLM"
   ],
   "vocab_size": 151936,
   "hidden_size": 1024,
   "num_hidden_layers": 28,
   "num_attention_heads": 16,
   "num_key_value_heads": 8,
   "head_dim": 128,
   "intermediate_size": 3072,
   "max_position_embeddings": 40960,
   "rms_norm_eps": 1e-06,
   "hidden_act": "silu",
   "tie_word_embeddings": true,
   "rope_theta": 1000000,
   "torch_dtype": "bfloat16"
}
```

A Gemma file adds things like `layer_types`, `sliding_window`, `hidden_activation`, `query_pre_attn_scalar`, and
`rope_local_base_freq`.

### How to read the tables below

- **Field** — the JSON key as written on disk (snake_case).
- **Means** — everyday meaning.
- **Used for** — what this Java engine does with it.
- **If missing** — the default or fallback this loader applies.

---

### Identity: which building plan is this?

| Field           | Means                                                                                 | Used for                                              | If missing                                         |
|-----------------|---------------------------------------------------------------------------------------|-------------------------------------------------------|----------------------------------------------------|
| `model_type`    | Short family name (`qwen3`, `gemma3_text`, `gemma4`, `llama`, …)                    | Exact match via `ModelSupport` (Qwen3 / Gemma3 text / Gemma 4 text / Llama; GGUF `qwen3` / `lfm2`; embeddings BERT encoder: `bert` / `roberta` / `xlm-roberta`) | Unsupported families fail with `UnsupportedModelException` and a support catalog. Optional `-Dnanollvm.arch=…` only when it **matches** the checkpoint. |
| `architectures` | List of class-style names from Hugging Face (`Qwen3ForCausalLM`, `LlamaForCausalLM`, …) | Same detection if `model_type` is unclear; `*ForConditionalGeneration` / vision classes are rejected | Optional                                           |

You can override causal detection with `-Dnanollvm.arch=qwen3`, `gemma3`, `gemma4`, `llama`, or `lfm2` **only when that id matches the checkpoint**. A forced id cannot turn Qwen2 / Qwen3.5 / vision models into a supported graph. Look-alike names are rejected (`qwen3_5` is not `qwen3`; `gemma2` is not `gemma3`; `gemma4` is its own text graph, not Gemma 3). BERT-encoder families (`bert`, `roberta`, `xlm-roberta`) load as embeddings from GGUF or ONNX (**since 1.3.0** for RoBERTa / XLM-R); DistilBERT / ALBERT / DeBERTa / ELECTRA are still rejected. Whisper is `whisper` from Hugging Face safetensors (**since 1.3.0**). Piper is detected from `*.onnx` + `*.onnx.json`. The error lists what this library actually loads (`ModelSupport.CATALOG`).
Embedding GGUFs use a separate detector (`bert` / `roberta` / `xlm-roberta`). **Since 1.3.0**, `LLM.builder(model)` is the runtime for every kind — chat, `embed`, `transcribe`, and `synthesize`. `LlmModel.embed` / `transcribe` / `synthesize` remain sequential shortcuts (no engine pool). ONNX folders use the
same `config.json` detection (chapter **7c**).

**Examples from real folders**

- Qwen3-0.6B: `"model_type": "qwen3"`, `"architectures": ["Qwen3ForCausalLM"]`
- Gemma3-270M: `"model_type": "gemma3_text"`, `"architectures": ["Gemma3ForCausalLM"]`
- Gemma 4 E2B QAT mobile: `"model_type": "gemma4"` / `"gemma4_text"` — text chat in this port (vision/audio towers skipped; see **Input and output modalities** below)
- SmolLM2 / Tiny-LLM ONNX: `"model_type": "llama"`, `"architectures": ["LlamaForCausalLM"]`
- xlm-roberta-base ONNX: `"model_type": "xlm-roberta"`, `"architectures": ["XLMRobertaForMaskedLM"]` — one BERT-encoder example via ONNX (**since 1.3.0**); any `bert` / `roberta` / `xlm-roberta` checkpoint with mapped names works the same
- whisper-base: `"model_type": "whisper"` — audio→text (**since 1.3.0**, chapter **7d**)
- Piper Lessac / Irina: folder of `*.onnx` + `*.onnx.json` — text→audio (**since 1.3.0**, chapter **7e**)

---

### Input and output modalities (**since 1.2.0**)

After load, `LlmModel.modalities()` is **what the checkpoint file declares**, not a hard-coded family list.
Detection looks at **keys on the root of `config.json`** (GGUF-mapped configs do not ship these keys):

| Input type | Declared when the root object has | Typical files |
|------------|-----------------------------------|---------------|
| **text** | Always, for every chat graph | Qwen3, Gemma3, Llama, LFM2, Gemma 4 |
| **image** | `vision_config` **or** `image_token_id` | Gemma 4 QAT mobile |
| **audio** | `audio_config` **or** `audio_token_id` | Gemma 4 QAT mobile |
| **video** | `video_config` **or** `video_token_id` | Gemma 4 QAT mobile (`video_token_id`) |

Chat **output** is always **text**. BERT-style embedding files skip that scan and report **text → embedding**.
Whisper reports **audio → text**. Piper reports **text → audio**.

`LlmModel.usableModalities()` is the narrower set this Java engine actually runs: **text → text** for chat,
**text → embedding** for BERT, **audio → text** for Whisper, or **text → audio** for Piper. Gemma 4’s extra towers stay skipped (`Gemma4QatLoader` drops `vision_tower` /
`audio_tower` weights). The `Example` demo prints the checkpoint pair after load, then a second “this library runs”
line when the two differ.

**In the code:** `Config.HfConfig` sets `imageConfigPresent` / `audioConfigPresent` / `videoConfigPresent` from those
keys; `LlmModalities.ofCheckpoint` builds the declared pair; `LlmModel.usableModalities()` is `TEXT_TO_TEXT`,
`TEXT_TO_EMBEDDING`, `AUDIO_TO_TEXT`, or `TEXT_TO_AUDIO`.

---

### Size of the dictionary and the stream

| Field               | Means (short)                                                           | Used for                                    | If missing                  |
|---------------------|-------------------------------------------------------------------------|---------------------------------------------|-----------------------------|
| `vocab_size`        | How many distinct token ids the model knows                             | Width of embedding table and LM head (rows) | Treated as 0 → broken model |
| `hidden_size`       | Dimension $H$ of each token’s hidden state (residual stream width)      | Embedding dim; attention/MLP I/O width      | 0 → broken                  |
| `intermediate_size` | How wide the **temporary expansion** is inside each layer’s MLP rewrite | Gate/up and down projection sizes           | 0 → broken                  |

### What `hidden_size` really is

Each token position carries a **fixed-length hidden state** — a vector in $\mathbb{R}^{H}$ where $H$ = `hidden_size`.
Papers often call this *d_model* or the *model dimension*.

- After the embedding lookup, token id `42` becomes a vector of length `hidden_size`.
- Attention mixes information across positions, but each position still leaves with a vector of the **same** length.
- Residuals add updates onto that same-width stream.
- At the end, that vector is scored against every vocabulary row to produce next-token logits.

So `hidden_size` is the **width of the residual stream**: the embedding dimension and the working width of every
transformer block’s main path.

It is **not**:

- the number of layers (that is `num_hidden_layers`);
- the dictionary size (that is `vocab_size`);
- how many tokens of context you may use (that is `max_position_embeddings` / your `maxModelLen`).

```text
  token "cat"  →  [ n₁, n₂, n₃, … , n_H ]
                   ◄────── hidden_size H ──────►

  same width after attention, after MLP, after every layer
```

Larger `hidden_size` → higher capacity per position, and more memory/compute. Smaller → leaner.

### What `intermediate_size` really is

Inside **each** layer, after attention, the MLP / feed-forward block does **not** stay at width `hidden_size` the whole
time. It typically:

1. **Expands** the hidden state from `hidden_size` to `intermediate_size` (often about 2×–4× wider),
2. Applies a nonlinearity / gate (SiLU, GELU, …),
3. **Shrinks** back to `hidden_size` before joining the residual stream again.

So `intermediate_size` is the width of that **temporary expanded representation** inside the MLP only. The residual
stream stays at `hidden_size`.

```text
  hidden state (width H)
        │
        ▼
  expand to intermediate_size I   ◄── temporary wide MLP representation
        │
     gate / activate
        │
  shrink back to H
        │
        ▼
  add onto residual (still width H)
```

Why expand? A wider intermediate representation gives the layer more capacity to recombine features before returning to
the residual stream. Most of a decoder layer’s **parameter count** often sits in these expand/shrink matrices — that is
why `intermediate_size` matters so much for file size and RAM.

### How the two fit together

|                            | `hidden_size` (H)                                        | `intermediate_size` (I)                            |
|----------------------------|----------------------------------------------------------|----------------------------------------------------|
| Lives where?               | Everywhere: embeddings, attention, residuals, final norm | Only inside each layer’s MLP                       |
| Stays for the whole stack? | Yes — same H from first embed to last layer              | No — temporary per layer                           |
| Role                       | Residual-stream / embedding dimension                    | Expanded MLP width                                 |
| Typical relation           | Baseline                                                 | Often roughly 2×–4× H (a design choice, not a law) |

**Real values** (from the example model folders this project documents — not quality rankings):

|                         | Qwen3-0.6B       | Gemma3-270M        |
|-------------------------|------------------|--------------------|
| `vocab_size`            | 151936           | 262144             |
| `hidden_size` (H)       | **1024**         | **640**            |
| `intermediate_size` (I) | **3072** (= 3×H) | **2048** (≈ 3.2×H) |

So Qwen’s residual width is larger (1024 vs 640), and both use an MLP intermediate width about three times $H$.

Gemma’s larger vocabulary and smaller hidden size are different design trade-offs — not “more intelligence” by
themselves.

---

### Depth and attention geometry

| Field                 | Means                                   | Used for                                                           | If missing                                     |
|-----------------------|-----------------------------------------|--------------------------------------------------------------------|------------------------------------------------|
| `num_hidden_layers`   | How many stacked reading rooms (layers) | Loop that builds layer 0 … N−1                                     | 0 → empty stack                                |
| `num_attention_heads` | How many Query glances run in parallel  | Multi-head attention width                                         | 0 → broken                                     |
| `num_key_value_heads` | How many Key/Value notebook groups      | GQA: Queries share KV groups when this is smaller than Query heads | Defaults to `num_attention_heads` (full MHA)   |
| `head_dim`            | Length of one head’s vector             | Attention math; QKV slice sizes                                    | If absent: `hidden_size / num_attention_heads` |

**GQA check:** if `num_key_value_heads` < `num_attention_heads`, several Query heads share one KV group  
(chapter 8).

**Real values**

|                       | Qwen3-0.6B | Gemma3-270M |
|-----------------------|------------|-------------|
| `num_hidden_layers`   | 28         | 18          |
| `num_attention_heads` | 16         | 4           |
| `num_key_value_heads` | 8          | 1           |
| `head_dim`            | 128        | 256         |

Qwen here: 16 Queries / 8 KV → groups of 2. Gemma here: 4 Queries / 1 KV → extreme sharing (MQA-like GQA).

---

### How long a passage may be

| Field                     | Means                                             | Used for                                                                                                | If missing                      |
|---------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------|
| `max_position_embeddings` | Official maximum context length the recipe claims | Caps RoPE tables; also caps this engine’s `maxModelLen` (the smaller of builder setting and this value) | Default **32768** in the loader |

**Real values:** Qwen3-0.6B → 40960; Gemma3-270M → 32768.

Your machine may not afford the full length in RAM; the builder’s `maxModelLen` can choose a smaller desk.

---

### Normalization and activations

| Field               | Means                                                                    | Used for                                                           | If missing       |
|---------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------|------------------|
| `rms_norm_eps`      | Tiny ε added under the square root in RMSNorm so division never explodes | Every RMSNorm                                                      | Default `1e-6`   |
| `hidden_act`        | Name of MLP gate activation (older / Qwen-style key)                     | Qwen expects `silu`                                                | Default `"silu"` |
| `hidden_activation` | Alternate activation name (Gemma-style key)                              | Preferred over `hidden_act` when present (`effectiveActivation()`) | Optional         |

**Effective rule in this project:** use `hidden_activation` if set; else `hidden_act`; else SiLU.

**Real values:** Qwen uses `"hidden_act": "silu"`. Gemma uses `"hidden_activation": "gelu_pytorch_tanh"` (a smooth
GELU-like curve in the MLP gate).

---

### Sharing the card index with the final scorer

| Field                 | Means                                                           | Used for                                                     | If missing                                                    |
|-----------------------|-----------------------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------|
| `tie_word_embeddings` | If true, embedding table and LM head **share** the same numbers | Saves a huge matrix; load path may skip a separate `lm_head` | Default **false** when the field is absent |

**Real values:** Qwen3-0.6B sets `true`. Gemma3-270M often omits the field and omits a separate `lm_head` tensor; the
`gemma3` graph still reuses embeddings when `lm_head` is missing on disk.

---

### Attention extras

| Field                   | Means                                         | Used for                                                                | If missing                  |
|-------------------------|-----------------------------------------------|-------------------------------------------------------------------------|-----------------------------|
| `attention_bias`        | Whether QKV projections add a bias vector     | Qwen path: if false, uses Q/K RMSNorm instead of bias                   | Default `false`             |
| `query_pre_attn_scalar` | Number used to scale attention scores (Gemma) | `attentionScale = 1 / sqrt(scalar)` (or `head_dim` if scalar absent/≤0) | 0 → fall back to `head_dim` |

**Real values:** both sample models set `attention_bias` false. Gemma3-270M sets `query_pre_attn_scalar` to 256 (same as
its `head_dim` here).

---

### RoPE — rotary positional embedding (how order enters attention)

**Name note:** RoPE stands for *Rotary Position Embedding*. It does **not** “rotate a position index.” It **rotates
pairs of numbers inside the Query and Key vectors**, using an angle that depends on each token’s position
(`0, 1, 2, …`). After that rotation, attention scores between two tokens depend on their **relative** distance as well
as their content.

Without some positional signal, “dog bites man” and “man bites dog” would look too alike to the match of Query against
Key. Older models often **added** a learned vector per position to the token embedding. RoPE instead leaves the token
embedding alone and twists Q and K inside each attention layer.

```text
  token at position p
       │
       ▼
  build Q, K, V from hidden state     (linear projections)
       │
       ▼
  RoPE: rotate Q and K by angle(p)    (V usually unchanged)
       │
       ▼
  attention scores ← Q · Kᵀ / scale   (relative positions now matter)
```

| Field                  | Means                                                     | Used for                                                                  | If missing               |
|------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------|--------------------------|
| `rope_theta`           | Base frequency for the rotation angles (global / default) | Build cos/sin tables for RoPE on Q and K                                  | Default `1000000`        |
| `rope_local_base_freq` | Separate base for **local / sliding** layers (Gemma)      | Sliding layers use this instead of `rope_theta`                           | Default `10000`          |
| `rope_scaling`         | Optional object describing long-context RoPE tricks       | If it contains `rope_theta`, that value can override the base (Qwen path) | `null` / absent → ignore |

Larger `rope_theta` is often used for longer contexts (slower angle growth with distance). Local Gemma layers with a
smaller base keep nearby positions distinct when the sliding window is short.

**In this library:** `Norms.RotaryEmbedding` precomputes a cos/sin cache of shape roughly
`[max_position_embeddings, head_dim]` from `rope_theta` (or the local base). Each attention block calls
`RotaryEmbedding#forward(positions, q, k)` **before** `Attention#forward`. See also chapter 8 and the embeddings section
in chapter 10.

**Real values:** both use `rope_theta` ≈ 1 000 000; Gemma also sets `rope_local_base_freq` to 10000; `rope_scaling` is
null in both samples.

---

### Sliding window and mixed layer types (Gemma)

| Field            | Means                                                            | Used for                                             | If missing                                                                                                       |
|------------------|------------------------------------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `sliding_window` | How many recent tokens a local layer may look back               | Window width for sliding attention                   | `0` or null → no windowing unless layer types say otherwise                                                      |
| `layer_types`    | Per-layer list: e.g. `"sliding_attention"` vs `"full_attention"` | `isSlidingLayer(i)` decides window + which RoPE base | If absent but `sliding_window` > 0: compat fallback — full attention every 6th layer (1-based), else sliding |

**Real Gemma3-270M:** `sliding_window` 512; `layer_types` alternates five sliding layers then one full, repeating across
18 layers.

**Real Qwen3-0.6B:** `sliding_window` null / unused here; `use_sliding_window` false — treated as global causal
attention in this port.

---

### Storage hint on disk

| Field         | Means                                                            | Used for                                                                        | If missing                 |
|---------------|------------------------------------------------------------------|---------------------------------------------------------------------------------|----------------------------|
| `torch_dtype` | Preferred training/storage dtype name (`bfloat16`, `float16`, …) | Informational in this port; tensors are still converted to Java float32 on load | Default `"float16"` string |

It does **not** keep this engine in BF16 math. It only describes how the hub file was typically stored.

---

### Fields you may see but this project does not drive inference with

Hub `config.json` files often contain more keys. Examples from the sample models that are **not** structural inputs to
the Java forward pass here:

| Field                                                | Typical meaning elsewhere            | Here                                                                                         |
|------------------------------------------------------|--------------------------------------|----------------------------------------------------------------------------------------------|
| `bos_token_id` / `eos_token_id` / `pad_token_id`     | Special token ids                    | EOS/stop come mainly from the **tokenizer**; these config ids are not the engine’s stop list |
| `attention_dropout`                                  | Training dropout                     | Ignored at inference                                                                         |
| `initializer_range`                                  | Weight init scale for training       | Ignored (weights come from safetensors)                                                      |
| `use_cache`                                          | Hint to cache KV in other frameworks | This engine always uses its own paged KV                                                     |
| `transformers_version`                               | Which library wrote the file         | Ignored                                                                                      |
| `use_sliding_window` / `max_window_layers`           | Alternate sliding metadata           | Logic uses `sliding_window` + `layer_types` (and Gemma fallback)                             |
| `attn_logit_softcapping` / `final_logit_softcapping` | Score capping in some Gemma variants | Not applied in this port                                                                     |
| `use_bidirectional_attention`                        | Non-causal mode                      | Must stay false for generation; not a supported mode here                                    |
| `_sliding_window_pattern`                            | Documentation of the 6-layer pattern | Fallback uses modulo 6 if `layer_types` missing                                              |

If a field is absent from the tables above, assume: **safe to leave as the hub shipped it**; this loader simply does not
consult it.

---

### Two worked readings

#### Qwen3-0.6B (compressed reading)

> Family qwen3; 28 layers; residual width **1024** (`hidden_size`); each MLP briefly widens to
> **3072** (`intermediate_size`, 3×); vocabulary ~152k; 16 Query heads with 8 KV groups
> (GQA); head size 128; SiLU MLP; embeddings tied to LM head; RoPE base 1e6; context claim up to 40960; weights stored
> as BF16 on disk, run as float32 here.

#### Gemma3-270M (compressed reading)

> Family gemma3_text; 18 layers; residual width **640**; MLP intermediate **2048** (~3.2×) with GELU-tanh gate;
> vocabulary ~
> 262k; 4 Query
> heads sharing 1 KV head; head size 256; attention scaled by `query_pre_attn_scalar` 256; many layers only look back
> 512 tokens, every sixth layer looks globally; local RoPE base 10k, global 1e6; expect tied embeddings.

---

### Blueprint vs engine knobs (do not confuse)

| Comes from `config.json` / GGUF metadata  | Comes from this program’s builder / runtime                                         |
|-------------------------------------------|-------------------------------------------------------------------------------------|
| Layer count, widths, heads, RoPE, windows | `maxModelLen` (capped by blueprint), KV page size, number of KV pages, batch limits |
| Which recipe (Qwen3 / Gemma3 / Gemma 4 text / Llama / LFM2 / BERT) | `-Dnanollvm.arch=…` override (causal); embedding GGUFs detected separately; ONNX folders use same HF `config.json` (not Gemma 4) |
| `torch_dtype` hint                        | Always float32 activations; GGUF weights may stay packed until unpack               |
| (not in blueprint)                        | `kvHeapFraction` (default **0.25**) — share of heap used to size the KV arena       |
| (not in blueprint)                        | CPU matmul threads / optional caller `ExecutorService`; warmup on/off               |

The blueprint says what the **model is**. The builder says how hard you ask your **machine** to run it.

**In the code:** `Config.HfConfig.load` fills the blueprint; `LLM.Builder` / `Config` set runtime knobs such as
`maxModelLen`, `kvHeapFraction`, KV pages, batch limits, and matmul threading (chapter 16). Process-wide parser /
corpus caps live in exported `utils.ResourceLimits` (**since 1.0.0**).

**Further reading:** configuration objects in the Python ecosystem —
[Hugging Face `PretrainedConfig`](https://huggingface.co/docs/transformers/main/en/main_classes/configuration); model
cards on the Hub usually ship the same `config.json` this chapter describes.


---

## 6. `tokenizer.json` — the dictionary file field by field

If `config.json` is the architect’s plan for the neural building, **`tokenizer.json`** is the **spelling book**: how
human text becomes token numbers and back. It is also JSON, but its job is linguistic plumbing, not layer widths.

This project loads Hugging Face folders with `Tokenizer.fromPretrained`. Companion files often sit beside
the tokenizer:

| File                     | Role here                                                                  |
|--------------------------|----------------------------------------------------------------------------|
| `tokenizer.json`         | Vocab, merges, normalizer / pre-tokenizer / decoder pipeline, added tokens |
| `tokenizer.model`        | SentencePiece protobuf when `tokenizer.json` is absent (**since 1.2.0**; `ModelFileId.TOKENIZER_MODEL`) |
| `tokenizer_config.json`  | Often holds `chat_template`, `eos_token`, `pad_token` names                |
| `generation_config.json` | May list extra `eos_token_id` values used as stop ids                      |

**GGUF path (chapter 7a / 7b):** there may be no separate `tokenizer.json`. The same public `Tokenizer` type is built
from embedded `tokenizer.ggml.*` metadata via `Tokenizer.fromGguf` / `GgufTokenizerSource` — BPE for many chat exports,
WordPiece for BERT embedding GGUFs. This chapter’s field-by-field tour is still the HF JSON shape; GGUF just stores a
subset of the same ideas inside one file.

Without a usable tokenizer, the model cannot turn your sentence into the ids the embedding table expects.

### Top-level shape

A Hugging Face / 🤗 Tokenizers export typically looks like:

```json
{
   "version": "1.0",
   "truncation": null,
   "padding": null,
   "added_tokens": [
      {
         "id":
         …,
         "content": "…",
         "special": true
         …
      }
      …
   ],
   "normalizer": {
      "type": "…"
      …
   },
   "pre_tokenizer": {
      "type": "…"
      …
   },
   "post_processor": {
      "type": "…"
      …
   },
   "decoder": {
      "type": "…"
      …
   },
   "model": {
      "type": "BPE",
      "vocab": {
         "token-string": id
         …
      },
      "merges": [
         "a b",
         "c d"
         …
      ],
      "byte_fallback": true/false,
      "unk_token": "…"
      …
   }
}
```

### Top-level fields

| Field            | Means                                                           | Used in this project?                                                  |
|------------------|-----------------------------------------------------------------|------------------------------------------------------------------------|
| `version`        | Format version of the tokenizer export                          | Informational                                                          |
| `truncation`     | Optional max-length truncation policy                           | Not applied as a separate engine policy here                           |
| `padding`        | Optional pad-to-length policy                                   | Not a generate-time padder here                                        |
| `added_tokens`   | Extra tokens beyond the base vocab (chat markers, `<think>`, …) | **Yes** — merged into vocab; specials noted                            |
| `normalizer`     | Text cleanup before splitting (Unicode form, replace rules, …)  | Inspected to detect style (e.g. Gemma Replace)                         |
| `pre_tokenizer`  | How raw text is first split into pieces                         | Detect Metaspace vs byte-level; drives encode path                     |
| `post_processor` | Template wrapping for encode in some setups                     | Present in file; chat wrapping here mostly uses `chat_template` / code |
| `decoder`        | How token strings become final text                             | Style detection; decode path (byte-level vs metaspace/`▁`)             |
| `model`          | The core BPE (or other) model: vocab + merges                   | **Central** — vocab map and merge ranks                                |

### `added_tokens[]` — special and extra scraps

Each entry is usually an object:

| Subfield                                           | Means                                                                   |
|----------------------------------------------------|-------------------------------------------------------------------------|
| `id`                                               | Integer id in the vocabulary                                            |
| `content`                                          | Exact string form (e.g. `<\|im_start\|>`, `<think>`, `<start_of_turn>`) |
| `special`                                          | If true, treated as a control token more than ordinary text             |
| `lstrip` / `rstrip` / `single_word` / `normalized` | Fine behaviour flags from the Tokenizers library                        |

**This loader:** copies `content` → `id` into the live vocab; marks many specials (including `<think>` /
`</think>`, Gemma turn markers, bos/eos) so they can be handled carefully when decoding.

**Scale:** Qwen3-0.6B sample has on the order of **tens** of added tokens; Gemma3-270M may list **thousands**
(including many special / multimodal leftovers even in a text build).

### `model` — the codec heart

| Subfield                                                  | Means                                                | Role                                                          |
|-----------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------|
| `type`                                                    | `"BPE"`, `"Unigram"`, `"WordPiece"`, `"WordLevel"`, or `"Char"` | Codec family chosen at load (**since 1.2.0** for Unigram / WordLevel / Char) |
| `vocab`                                                   | Map **token string → integer id**                    | Encode lookup and decode reverse map                          |
| `merges`                                                  | Ordered list of pair merges (`"a b"` or `["a","b"]`) | Rank: earlier merge = higher priority when compressing pieces |
| `byte_fallback`                                           | If unknown pieces fall back to byte tokens           | Affects robustness on odd characters                          |
| `unk_token`                                               | String for unknown (if used)                         | Style-dependent                                               |
| `dropout`, `fuse_unk`, `ignore_merges`, prefixes/suffixes | Training / advanced BPE options                      | Mostly unused at simple inference                             |

**How large are the vocab and merge lists? (example models in this project)**

The tables above name fields; the numbers below show **scale** for the two model directories this guide uses as running
examples — Qwen3-0.6B and Gemma3-270M under `models/`. They are not “quality scores.” They answer: *how many distinct
token strings does the tokenizer know, and how many BPE merge rules does it apply when compressing text into those
tokens?*

| What you are counting     | Why it matters                                                               | Qwen3-0.6B | Gemma3-270M |
|---------------------------|------------------------------------------------------------------------------|------------|-------------|
| Entries in `model.vocab`  | Size of the string↔id dictionary used by encode/decode                       | ~151 643   | ~262 144    |
| Entries in `model.merges` | Number of ordered pair-merge rules (earlier = higher priority when encoding) | ~151 387   | ~514 906    |

Gemma’s larger vocab and much longer merge list mean a finer-grained dictionary (more special/reserved strings as well
as ordinary pieces). Qwen’s lists are smaller but still on the order of 10⁵ entries — typical for modern chat models.

**Do not confuse these with `config.json`:** `vocab_size` there can be **slightly larger** than `model.vocab` (padding /
reserved embedding rows). The **embedding matrix** is sized from `config.json`; the **tokenizer’s** live dictionary
comes from `tokenizer.json`. If those disagree badly, encode/decode and the LM head no longer line up.

### Encoding styles this project supports

#### A. GPT-2-style byte-level BPE (typical Qwen)

Everyday story:

1. Optionally normalize (e.g. Unicode NFC).
2. Split text with a GPT-2-like regex into rough chunks.
3. Map each byte to a printable stand-in character (byte encoder).
4. Repeatedly merge adjacent pairs according to `merges` ranks until nothing cheaper remains.
5. Look up each final piece in `vocab` → ids.

Decode reverses the map and turns byte stand-ins back into UTF-8 text. The `decoder` is often `"ByteLevel"`.

#### B. Metaspace / `▁` BPE (typical Gemma)

Everyday story:

1. Normalizer may **replace** spaces with a special mark `▁` (U+2581), or a Metaspace pre-tokenizer may prepend `▁`.
2. Split into pieces; run BPE merges.
3. Decode turns `▁` back into spaces / word boundaries.

The loader chooses **METASPACE_BPE** vs **GPT2_BYTE_BPE** by inspecting pre-tokenizer/decoder nodes and vocab clues
(Gemma chat markers, etc.).

#### C. WordPiece / GGUF-embedded (BERT embeddings, **since 1.1.0**)

BERT-family embedding GGUFs often set `tokenizer.ggml.model=bert`. Hugging Face `tokenizer.json` may also set
`model.type` to `"WordPiece"` (HF `##` pieces). Encoding is **greedy longest-match WordPiece**, not the BPE merge
loop above. Apps never pick a codec enum — style is chosen inside `Tokenizer` when the model loads
(`ChatFormat` is the public chat-layout enum). Details: chapter 7b.

#### D. Unigram SentencePiece, WordLevel, and character models (**since 1.2.0**)

- **Unigram** (XLM-RoBERTa / multilingual E5 `tokenizer.json`): Viterbi segmentation; Hugging Face precompiled
  **charsmap** from the normalizer blob is applied when present. Embedding wrap accepts `<s>`/`</s>` as well as
  `[CLS]`/`[SEP]`.
- **WordLevel** / **Char**: lookup or per-character ids from `tokenizer.json` `model.type`.
- **SentencePiece `tokenizer.model`:** used when `tokenizer.json` is missing. `Tokenizer.fromSentencePiece(bytes, …)`
  builds the same codec from in-memory protobuf.

### How encode / decode use the file (organization)

```text
  your string
      │
      ▼
  normalizer / pre-tokenizer rules (from JSON)
      │
      ▼
  BPE merges (ordered list in model.merges)
      │
      ▼
  vocab lookup → list of ids  ──►  model embeddings
      │
      ▼
  … generation …
      │
      ▼
  ids → vocab reverse → decoder rules → readable string
```

Chat templates (often from `tokenizer_config.json`) wrap roles **before** encode, so the model sees the markers it was
trained with (`<|im_start|>`, `<start_of_turn>`, …).

### What is *not* inside `tokenizer.json`

- Learned attention / MLP weights (those are `.safetensors`).
- Layer counts and head geometry (those are `config.json`).
- Runtime KV notebooks.

It is only the **bridge language ↔ numbers**.

### A fair one-sentence summary

> **`tokenizer.json` stores the vocabulary, the merge or Unigram recipe, and the text-cleanup pipeline so strings become
> the integer ids the embedding table understands — and back again. Folders without that JSON use SentencePiece
> `tokenizer.model` instead (**since 1.2.0**).**

**In the code:** `Tokenizer.fromPretrained` / `fromGguf` / `fromSentencePiece` (**since 1.2.0**) read vocab + optional
chat template / stop ids; `ChatFormat` (ChatML / turn-based / plain) and `invitesThinking()` follow markers in
template/vocab (not product names); encode/decode and `applyChatTemplate` live on `Tokenizer` (chapter 16). Codecs
are package-private behind that public type.

**Further reading:** the JSON pipeline (normalizer → pre-tokenizer → model → decoder) is the
[Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) design; API overview of the `Tokenizer` class is
[here](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer).


---

## 7. Tensors and `*.safetensors` — parameters on disk

The large files named `model.safetensors` or `model-00001-of-00003.safetensors` hold the **learned parameters** of the
model. This project reads them with `models.llmcontainer.SafetensorsReader` and copies them into the empty parameter buffers allocated from
`config.json`. Before the file format, it helps to fix what a **tensor** is — because every weight, every activation,
and every intermediate result in chapters 8–11 is one.

### What a tensor is

In this engineering context (and in deep-learning libraries generally), a **tensor** is a **multidimensional array of
numbers**, together with a **shape** that says how those numbers are organized.

| Rank (number of axes) | Usual name   | Example shape      | Example role                                    |
|-----------------------|--------------|--------------------|-------------------------------------------------|
| 0                     | scalar       | `()`               | Rare as a stored weight; one number             |
| 1                     | vector       | `[H]`              | Hidden state of one token; RMSNorm scale vector |
| 2                     | matrix       | `[V, H]`, `[I, H]` | Embedding table; linear-layer weight            |
| 3+                    | higher-order | `[L, B, H]`, …     | Batched activations, caches (layout varies)     |

**Shape** is an ordered list of positive integers. The product of the dimensions is the **number of elements**
(`numel`). Storage is almost always a **flat contiguous buffer** in row-major order: the last index changes fastest.

Two tensors with the same `numel` can be **reshaped** without moving values (reinterpret the axes). Two tensors with
different `numel` cannot.

**Weights vs activations.** Parameters loaded from disk (embedding matrix, projection matrices, norm scales) are
**weights**. Arrays produced during a forward pass (hidden states, attention scores, logits) are **activations**. Both
are tensors; only weights persist across requests as part of the model.

**Dtype.** On disk a tensor may be stored as BF16, F16, F32, or F64. After load, this educational CPU port widens
everything to Java `float` (IEEE float32). Shape is unchanged; only element width grows. Other safetensors dtypes
(int / uint / bool / …) **fail the load**.

Formal detail and the Java type live in **chapter 10**. The rest of this chapter is the **on-disk container** that
delivers named weight tensors into RAM.

### Supported formats / restrictions (safetensors)

| Allowed | Not in this port |
|---------|------------------|
| Element dtypes **`F32`**, **`F16`**, **`BF16`**, **`F64`** → float32 weights | Other catalog dtypes except Gemma 4 QAT packed int2/4/8 (+ SRQ) |
| HF folder causal graphs: **Qwen3**, **Gemma3**, **Gemma 4 text** (QAT mobile, packed), **Llama** (when `config.json` + names match) | **LFM2** from an HF safetensors directory (use **GGUF**, ch. 7a); Gemma 4 **GGUF** / **MoE** / vision-audio generation |
| Same folder may also use ONNX instead (ch. **7c**, not Gemma 4); if **both** `*.safetensors` and `*.onnx` exist, **safetensors wins** | Full HF **BERT** safetensors directory load (BERT embeddings: GGUF ch. 7b, or ONNX BERT when names map) |
| Activations / KV always float32 after load | Keeping BF16/F16 math at runtime |

### Why the safetensors format exists

Older workflows used opaque pickles. **Safetensors** is a simple, memory-map–friendly layout:

1. A small binary length prefix.
2. A JSON **catalog** naming every tensor (name → dtype, shape, byte range).
3. A raw **payload** of little-endian numeric bytes.

No hidden code execution — only data.

### On-disk layout (byte by byte)

```text
  offset 0        8 bytes     header length N  (uint64 little-endian; this reader uses the low 32 bits)
  offset 8        N bytes     UTF-8 JSON header (catalog)
  offset 8+N      …           raw tensor bytes (payload)
```

The JSON header maps **tensor name → descriptor**. Optional key `"__metadata__"` holds free-form strings (e.g.
`{"format":"pt"}`) and is skipped when loading weights.

### Each tensor’s catalog entry

Example from a real Qwen file:

```json
"model.layers.0.mlp.gate_proj.weight": {
   "dtype": "BF16",
   "shape": [
      3072,
      1024
   ],
   "data_offsets": [
      628623360,
      634914816
   ]
}
```

| Field               | Means                                                                                       |
|---------------------|---------------------------------------------------------------------------------------------|
| **name** (JSON key) | Hugging Face parameter path — must match what the model graph expects (or a packed rewrite) |
| `dtype`             | Element type on disk: `F32`, `F16`, `BF16`, `F64` (this reader)                             |
| `shape`             | List of dimensions, e.g. `[out, in]` for a weight matrix                                    |
| `data_offsets`      | `[start, end)` **relative to the start of the payload** (not including the 8+N header)      |

Byte length of one tensor ≈ `end - start`. For BF16, that is roughly `2 × product(shape)` bytes.

### Dtypes: what the letters mean

| `dtype` | On disk      | What this Java port does                   |
|---------|--------------|--------------------------------------------|
| `F32`   | 32-bit float | Copy into `float[]`                        |
| `F16`   | 16-bit float | Expand each value to Java `float`          |
| `BF16`  | bfloat16     | Expand to Java `float` (shift/expand bits) |
| `F64`   | 64-bit float | Narrow to Java `float`                     |

**Important:** after load, **float** safetensors compute as **float32** in this teaching engine. A BF16 file still
becomes a large F32 resident image in RAM. **Exception (since 1.1.0):** Gemma 4 **text QAT mobile** keeps packed
int2/4/8 (+ SRQ) weights in heap and dequantizes during matmul — same idea as GGUF packed mode, not a full F32 expand.
Shards larger than 2 GiB are read with `FileChannel` (files ≤ 2 GiB are copied into a heap buffer so `close()` can drop them). Gemma 4 **GGUF**, **MoE**
(`enable_moe_block`), and vision/audio generation are still unsupported; towers in the QAT-mobile crate are skipped.

**Samples:** both Qwen3-0.6B and Gemma3-270M ship **all** listed tensors as `BF16` in the inspected files.

### What the contents *are* (names and roles)

Each catalog name identifies which parameter buffer it fills. Typical patterns:

| Name pattern                                                                  | Tensor content (shape intuition)                                            |
|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `model.embed_tokens.weight`                                                   | Token embedding matrix `[vocab_size, hidden_size]` — one row per token id   |
| `model.layers.i.input_layernorm.weight`                                       | RMSNorm scale vector for layer *i* (before attention)                       |
| `model.layers.i.self_attn.q_proj.weight`                                      | Query projection matrix (HF may store Q/K/V separately)                     |
| `model.layers.i.self_attn.k_proj.weight`                                      | Key projection matrix                                                       |
| `model.layers.i.self_attn.v_proj.weight`                                      | Value projection matrix                                                     |
| `model.layers.i.self_attn.o_proj.weight`                                      | Output projection after attention                                           |
| `model.layers.i.self_attn.q_norm.weight` / `k_norm.weight`                    | Optional per-head Q/K norms                                                 |
| `model.layers.i.post_attention_layernorm.weight`                              | Norm before MLP                                                             |
| `model.layers.i.mlp.gate_proj.weight` / `up_proj.weight` / `down_proj.weight` | Gated MLP matrices                                                          |
| `model.layers.i.pre_feedforward_layernorm.weight` / `post_…`                  | Extra Gemma norms                                                           |
| `model.norm.weight`                                                           | Final RMSNorm before scoring                                                |
| `lm_head.weight`                                                              | LM head `[vocab_size, hidden_size]` — often absent when embeddings are tied |

**Counts in samples:** Qwen3-0.6B ≈ **311** tensors in one file; Gemma3-270M ≈ **236** (tied head → often no separate
`lm_head` tensor).

### Shards (several files)

Large models may split weights across `model-00001-of-0000N.safetensors`. The loader **lists all `*.safetensors`**,
opens them in sorted order, and plans every matching tensor. Each file has its **own** header + payload; names must not
collide across shards for the same parameter.

### How this project loads a tensor

```text
  1. Read header → names, dtypes, shapes, byte ranges
  2. Match name → canonical param (rewrite q_proj/k_proj/v_proj → fused qkv_proj, etc.)
  3. read payload slice → convert to float32 Tensor
  4. Assemble into WeightBag (merge packed shards), then ArchitectureProcessor.createCausal builds the graph
```

Unknown names are skipped. Missing **required** names fail the load.

### What is *not* in `.safetensors`

- Tokenizer strings and merges.
- Chat templates.
- KV cache / attention state (allocated empty at runtime).
- RoPE cos/sin tables (computed from `rope_theta` in config).

Only **trained parameters** (and sometimes buffers stored as tensors).

### Size intuition

Rough on-disk payload ≈ Σ over tensors of `bytes_per_element × numel`.

Example: one BF16 embedding `[151936, 1024]` is already hundreds of megabytes before the layers. After expansion to F32
in RAM, expect roughly **~2×** that payload for resident weights alone — plus KV pages and activations.

### Summary

> **A `.safetensors` file is a named collection of tensors: a JSON index of shapes and byte ranges, followed by a raw
> numeric payload. Float crates are widened to float32; Gemma 4 text QAT stays packed. Names assemble into a
> `WeightBag` used to construct the immutable model graph.**

**In the code:** `models.llmcontainer.SafetensorsTransport` lists shards and headers; `models.llmcontainer.SafetensorsReader` reads payloads; `ArchitectureProcessor.bind` + `fill` (`ModelLoader` / `Gemma4QatLoader`) match names
(including packed `q_proj`/`k_proj`/`v_proj` → `qkv_proj`) and merges into `WeightBag`; `models.llmcontainer.LoadProgress` redraws
the same percent/ETA bar used by GGUF and ONNX when a listener is attached; `ArchitectureProcessor.createCausal` builds
the graph (chapter 16). Conceptual tensor definitions: chapter 10.

**Further reading:** format overview in the
[Safetensors documentation](https://huggingface.co/docs/safetensors); binary layout notes in the
[safetensors GitHub README](https://github.com/safetensors/safetensors).


---

<a id="7a-gguf--quantized-single-file-models-qwen3-and-lfm2"></a>
<a id="7a-gguf-and-lfm2--quantized-single-file-models"></a>
## 7a. GGUF — quantized single-file models (Qwen3 and LFM2)

Chapter 7 described **safetensors**: a JSON catalog plus a raw float/BF16 payload. **GGUF** is the other on-disk
container this project can open (popular with llama.cpp). One file holds **typed metadata**, an embedded **tokenizer**,
and **GGML-typed weight blocks** (often quantized). This engine supports **GGUF v2/v3** for:

- architecture **`qwen3`** — dense Qwen3 text **chat** (**since 1.1.0**; same graph as the Hugging Face folder; GGUF
  keeps unfused `blk.N.attn_*` / `ffn_*` tensors so packed quants stay packed);
- architecture **`lfm2`** — hybrid short-convolution + GQA **chat** models (this section);
- architecture **`bert`** — sentence-embedding encoders (**chapter 7b**, **since 1.1.0**).

Load is three layers for every container: **transport** opens the file or folder (`GgufTransport`, `SafetensorsTransport`, `OnnxTransport`) and reports a catalog (metadata + tensor names). **Architecture processing** (`ArchitectureProcessor`) then decodes the family (`qwen3` / `gemma3` / `gemma4` / `llama` / `lfm2` / `bert`). Chat families share a causal template (`CausalArchitecture`: Hugging Face bind by default, required `createCausal`); BERT uses `EmbeddingArchitecture`. Processors bind config + schema, fill weights, and build the graph. Gemma 4 still overrides fill (QAT safetensors only) and LFM2 overrides bind/fill (GGUF only) — they are not subclasses of Gemma 3 or of each other. Unsupported families (Qwen2, Gemma/Llama GGUF, MoE, VL) still fail before weights are copied, with
`ModelSupport.CATALOG`.

There is **no** in-repo download for a Qwen3 GGUF. Place any `general.architecture=qwen3` file yourself (for example
from [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)). The script
`models/download-qwen3-0.6b.sh` still fetches the Hugging Face **safetensors folder**. LFM2 GGUF does have a script
(`models/download-lfm2.5-2.6b-gguf.sh`).

For causal GGUF chat it keeps weights **packed** in RAM by default and **dequantizes rows/tiles during matmul and embedding**.
Each `Linear` / LM-head binds a `LinearKernel` at construction (`dense-f32` with decode-1, or packed with a
GGML-type-fixed dequant lambda). Token embeddings bind a matching `EmbeddingKernel`.
`GgufDequant.dequantizeRange` writes block quants **straight into the caller float buffer** (a one-block scratch
only when a range starts mid-block).

**Three unpack stories (do not conflate them):**

| Mode | How you ask | What happens to weight RAM |
|------|-------------|----------------------------|
| **Packed (default)** | `LlmModelFactory.make(path)` | Quant blocks stay in heap; dequant on each matmul/embed use |
| **Unpack at load** | `make(path, io, true)` | file bytes → float32 while loading; **no** packed heap copy of those tensors |
| **Late unpack** | `LLM.Builder.allowUnpackParameters()` on a packed model | Materializes float32; **releases packed payloads** when no other engine still holds them |

```java
// Qwen3 chat GGUF (since 1.1.0) — bring your own file; no models/download-qwen3-*.gguf script
Path qwen3Gguf = Path.of("/opt/models/qwen3.gguf");
try (LlmModel model = LlmModelFactory.make(qwen3Gguf);
     LLM llm = LLM.builder(model).build()) {
  String answer = llm.chat(256).send("Hello").answer();
}

Path lfm2 = Path.of("models/LFM2.5-2.6B-Q4_K_M.gguf");

// Packed default (smallest weight RAM; dequant on matmul)
try (LlmModel model = LlmModelFactory.make(lfm2);
     LLM llm = LLM.builder(model)
         // optional: .allowUnpackParameters()  // late float32 for this engine
         .build()) {
  String answer = llm.chat(256).send("Hello").answer();
}

// Unpack at load instead (no packed heap copy of those tensors):
// LlmModelFactory.make(lfm2, LlmListeners.silent(), true);
```

Activations and the KV cache remain float32 either way. A ~1.7 GB Q4_K_M 2.6B file stays near that size for packed
weights; default `-Xmx16g` in `.mvn/jvm.config` is still useful headroom for KV / scratch.

### Why GGUF exists (vs safetensors)

| | `.safetensors` (ch. 7) | `.gguf` (this section) |
|--|------------------------|-------------------------|
| Typical source | Hugging Face snapshot directory | Single llama.cpp-style export |
| Catalog | UTF-8 JSON header | Binary KV metadata + tensor info table |
| Element types | `F32` / `F16` / `BF16` / … | GGML types: floats **and** block quants (`Q4_K`, …) |
| Tokenizer | Separate `tokenizer.json` | Often embedded (`tokenizer.ggml.*`) |
| This port | Widen to float32 | Keep packed by default; **dequant on matmul**; `make(..., true)` or `allowUnpackParameters()` → float32 |

No pickle, no executable code — only structured data. This reader memory-maps the file (current limit: **≤ 2 GiB** map).

### On-disk layout (byte by byte)

Little-endian throughout:

```text
  offset 0        4 bytes     magic "GGUF"  (uint32 0x46554747)
  offset 4        4 bytes     version       (2 or 3 supported here)
  offset 8        8 bytes     tensor_count  (uint64)
  offset 16       8 bytes     kv_count      (uint64)
  …               …           kv_count × (key string, value type, value)
  …               …           tensor_count × (name, n_dims, dims[], ggml_type, relative_offset)
  …               pad         align to general.alignment (default 32)
  …               …           tensor payload bytes (quant blocks or floats)
```

Strings are `uint64 length` + UTF-8 bytes. Each metadata **value** has a type tag (uint32, float32, string, array, …).
Each **tensor info** records:

| Field | Means |
|-------|--------|
| **name** | GGML / export name (e.g. `blk.0.attn_q.weight`, `token_embd.weight`) — not always HF dotted paths |
| `n_dims` / `dims[]` | Axis lengths in **GGML order** (often reverse of HF `[out, in]`) |
| `ggml_type` | Element / block layout id (see table below) |
| `relative_offset` | Byte offset from the start of the **aligned payload**, not from file start |

**Shape note:** when this reader builds a Java `Tensor`, **2D dims are reversed** to Hugging Face style
`[out, in]` / embedding `[vocab, dim]`. Higher-rank dims are reversed the same way. That mismatch is a common source of
“wrong matmul shape” bugs if skipped.

### GGML dtypes this port dequantizes

Row dequant follows llama.cpp `ggml-quants.c`. File names like **Q4_K_M** / **Q5_K_S** are recipes that **mix** these
`ggml_type` ids (not extra dtypes).

| Group | `ggml_type` ids | Notes |
|-------|-----------------|-------|
| Floats | `0` F32, `1` F16, `30` BF16, `28` F64 | Expand / copy to Java `float` |
| Integers | `24` I8, `25` I16, `26` I32, `27` I64 | Rare as weights; widened to float |
| Legacy quants | `2` Q4_0, `3` Q4_1, `6` Q5_0, `7` Q5_1, `8` Q8_0, `9` Q8_1, `41` Q1_0, `42` Q2_0 | 32- or 64/128-elem blocks |
| K-quants | `10` Q2_K, `11` Q3_K, `12` Q4_K, `13` Q5_K, `14` Q6_K, `15` Q8_K | 256-elem superblocks |
| IQ | `16`–`19`, `20` IQ4_NL, `21`–`23`, `29` IQ1_M | Lookup grids from ggml-common.h |
| Ternary / MX | `34` TQ1_0, `35` TQ2_0, `39` MXFP4, `40` NVFP4 | |

Weights stay packed; activations use float32 after per-row dequant inside the bound `LinearKernel` /
`EmbeddingKernel` (via `GgufDequant.dequantizeRange`).

### Supported formats / restrictions (GGUF)

| Allowed | Not in this port |
|---------|------------------|
| GGUF **v2 / v3**; heap payload **≤ ~2 GiB** | Larger files / other GGUF major versions |
| Architectures **`qwen3`** / **`lfm2`** (chat) and **`bert`** (embeddings **since 1.1.0**) | **Qwen2**, **Gemma / Llama** GGUF, MoE / VL, or other exports |
| GGML weight types in the table above | Removed ggml layouts (`Q4_2`/`Q4_3`, SIMD-repack `Q4_0_4_4` / `IQ4_NL_4_4`, …) |
| Packed default; optional unpack at load / late unpack | ONNX or safetensors wrapped *inside* a `.gguf` |

### What else lives in GGUF metadata

Useful keys this loader / tokenizer read (names vary slightly by export):

| Key pattern | Role |
|-------------|------|
| `general.architecture` | Must be `qwen3` or `lfm2` for chat (`bert` → chapter 7b) |
| `lfm2.*` / `qwen3.*` / `*.block_count`, `*.embedding_length`, … | Sizes → `Config.HfConfig` |
| `qwen3.attention.key_length` (or `value_length`) | Qwen3 `head_dim` — **not** `hidden / heads` (0.6B is 128, not 64) |
| `lfm2.attention.head_count_kv` | Often a **per-layer array** (0 on conv layers) — use max > 0 as GQA kv heads |
| `tokenizer.ggml.tokens` / `.merges` | Embedded vocab |
| `tokenizer.chat_template` | Optional; many LFM2 files **omit** it even when ChatML specials exist in vocab |

### LFM2 graph (why architecture matters)

**LFM2** is a *hybrid* stack: most layers are gated **short convolutions** (small rolling state per sequence), and a
subset are full **grouped-query attention**. A GGUF that only matched Qwen/Gemma layer shapes would still be wrong —
the graph must know which layers are `conv` vs `full_attention`.

### Chat packaging (easy to get wrong)

Many `lfm2` GGUF exports omit an embedded `chat_template` string even though the vocab still contains ChatML markers
(`<|im_start|>`, `<|im_end|>`). This port therefore:

- Detects **`Tokenizer.ChatFormat`** from template/vocab markers (**ChatML** / turn-based / plain) — not from product names.
- Sets `Tokenizer.invitesThinking()` from vocab (`<think>` + `</think>` present), for HF and GGUF loads alike.
  Custom pairs go on `LlmModelFactory.open(path).thinkTags(tags).make()`
  (frozen on the model; inherited by every `LLM` sharing the checkpoint);
  skip-seed then uses `Tokenizer.invitesThinking(open, close)`. `ChatSession.thinkTags` overrides one conversation.
  Chat markup searched in decoded answers is `ChatSpecials` (`OPTION_CHAT_SPECIALS`; default ChatML / Gemma / Llama specials plus the default think pair).
- Keeps library system text empty (`ChatPrompts.systemFor`); demos set policy via samples `SampleChatPrompts`.
- When thinking is disabled and the vocab invites the configured tags, ChatML may pre-insert an empty open/close
  pair so the model skips a long scratchpad (token-budget control — not a second brain).

The UI can still split `<think>` / answer if the model emits those tags. (Sense C details: chapter 9.)

Example ChatML prompt fragment this tokenizer builds:

```text
<|im_start|>system
You are a helpful assistant. …
<|im_end|>
<|im_start|>user
hello
<|im_end|>
<|im_start|>assistant
```

### CPU matmul threads

Dense `MatmulRuntime` / `FloatKernels.dot` can split the output axis across workers. Extra cores also speed
**long-prompt** work on the same pool: independent attention heads, RoPE tokens, embedding-row gathers, and Gemma
QAT activation scaling (`cpuThreads > 1`). Nested pool work stays sequential so a fixed pool cannot deadlock.
Each **`LLM`** owns its own runtime. Defaults: `Runtime.availableProcessors()` (`LLM.Builder.cpuThreads(N)` /
`.allCpuThreads()` / `.disableMultiCpu()`, or `-Dnanollvm.cpu.threads=N`). Builder thread count wins over the
system property. `.disableMultiCpu()` / `cpuThreads(1)` use a **sequential** runtime (no worker pool — the calling
thread only). An optional `.matmulExecutor(ExecutorService)` is never shut down by `LLM.close()`.
**Since 1.2.0**, `.dedicatedMatmulPool()` creates a bounded pool of `cpuThreads` daemons that `LLM.close()` shuts
down (cannot combine with `matmulExecutor`) — use this in a server instead of joining the process-wide
`nanollvm-matmul-*` default. This helps multi-core decode; GGUF weight RAM stays near packed size (KV /
activations are still float32).

### Summary

> **A `.gguf` file is a little-endian catalog of metadata + named GGML tensors, followed by an aligned payload of
> float or block-quantized bytes. This port maps the file, keeps supported types packed (reversing 2D dims to HF
> layout), and dequantizes rows in place during matmul / embedding into the same float32 activation path as
> safetensors. Chat graphs are `qwen3` and `lfm2`; the same container also loads BERT embedding graphs (chapter 7b).**

**In the code (application entry):** `LlmModelFactory.make` / `make(…, true)` / `LLM.Builder.allowUnpackParameters()`;
tokenizer via the sealed `LlmModel`. **Internal map:** `models.llmcontainer.GgufTransport` (catalog) → `ArchitectureProcessor.bind`
(adapt metadata → `HfConfig` + schema) → `ArchitectureProcessor.fill` (payloads via `GgufReader` / `GgufDequant`);
weight-load status uses the same `models.llmcontainer.LoadProgress` bar as safetensors and ONNX;
`models.internal.PackedWeight` / `Qwen3ForCausalLM` (unfused GGUF linears) / `Lfm2ForCausalLM`;
`tensor.LinearKernel` / `EmbeddingKernel`; short-conv state in
`engine.ConvStateArena` via `Transformer` / `internal.Context`; chat defaults via `ChatPrompts.systemFor(Tokenizer)`
(always empty — demos use `SampleChatPrompts`) and vocab-gated `Tokenizer.invitesThinking()`; matmul via
`MatmulRuntime` (per `LLM`).

**Further reading:** [GGUF format notes](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md); Qwen3 GGUF example
[Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF); Liquid
[LFM2 blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models);
example weights [LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF).


---

## 7b. BERT embedding GGUFs — sentence vectors (**since 1.1.0**)

Chapter 7a showed how **causal** chat models can live in a `.gguf` file. **Since 1.1.0**, the same GGUF loader also
accepts a different kind of network: a **BERT-family encoder** whose job is not to write the next token, but to turn a
finished sentence into one **dense vector** (an embedding) you can compare with other sentences.

The demo weight people usually try for a **tiny English GGUF** is **gte-small** (`models/gte-small.Q2_K.gguf`, via
`models/download-gte-small-gguf.sh`). The linear sample `EmbeddingsHelloWorld` defaults to **multilingual-e5-small**
ONNX (`models/download-multilingual-e5-small.sh`) — same `LlmModel.embed` API, different tokenizer and **task prefixes**
(below).

### What “BERT” means here (without the training lore)

**BERT** (Bidirectional Encoder Representations from Transformers) is an architecture for **reading** text, not
**continuing** it.

| | Causal chat LM (Qwen / Gemma / Llama / LFM2) | BERT-style embedding encoder |
|---|---|---|
| Question it answers | “What token comes **next**?” | “What is a good **vector summary** of this whole input?” |
| Attention | **Causal** — no peeking at future positions | **Bidirectional** — each token may look left **and** right |
| Typical API here | `LLM.builder(model)` → `chat` / `generate` | `LLM.builder(model)` → `embed` (or `LlmModel.embed` as a sequential shortcut) |
| Output | Token ids / text | One L2-normalized vector of length `hidden_size` |

People also say “BERT” for many descendants that keep that encoder shape (including small **sentence embedding** models
such as GTE). In this port, GGUF metadata that looks like BERT loads as architecture `bert` and builds
`BertForEmbedding`.

**Humanities picture:** a causal LM is a writer who may only see the draft so far. A BERT encoder is a reader who may
reread the whole finished card before writing a single summary number-line for the card box (chapter 17 dense RAG).

### How this project loads and runs it

1. **Same factory as chat models:** `LlmModelFactory.make(pathToGguf)`.
2. **Detect:** `model.isEmbeddingModel()` is true; `architectureName()` is typically `bert`.
3. **Open an engine:** `LLM.builder(model).build()` then `llm.embed(text)` — same CPU pool as chat
   (**since 1.3.0**). `LlmModel.embed` remains a sequential shortcut (no engine pool; that is what dense RAG indexing uses).
4. **Encode:** `float[] v = llm.embed("hello world");` (or `embed(List)` / already-tokenized ids).
5. **Compare:** vectors are **L2-normalized**, so cosine similarity is just the **dot product**.

```java
try (LlmModel model = LlmModelFactory.make(Path.of("models/gte-small.Q2_K.gguf"));
     LLM llm = LLM.builder(model).build()) {
  float[] a = llm.embed("Paris is the capital of France.");
  float[] b = llm.embed("What city is France's capital?");
  // cosine ≈ sum_i a[i]*b[i]
}
```

**In the code:** `LlmModelFactory` → `ArchitectureProcessor.createEmbedding` → `BertForEmbedding`; public surface `LLM.embed` / `LlmModel.embed`;
samples `EmbeddingsHelloWorld` (default multilingual-e5-small ONNX), `Example` menu items for gte-small, E5, and xlm-roberta-base.

### Task prefixes (`query:` / `passage:`) — library vs checkpoint

`LlmModel.embed` does **not** add any task tag. It tokenizes the string you pass (and wraps BERT `[CLS]` / `[SEP]`, or
XLM-RoBERTa `<s>` / `</s>` when those are the specials). Prefixes such as `query: ` are **not** a requirement of the
BERT graph or of this engine.

Some embedding checkpoints were **trained** with those tags inside the text. The **E5** family
([intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)) expects:

| Prefix | When to use it |
|--------|----------------|
| `query: …` | A search question, **or** any non-retrieval sentence (similarity, clustering, classification) |
| `passage: …` | A document you want to retrieve against a query |

Without them the encoder still runs; the vectors sit off the space E5 learned, so cosine scores are weaker and less
comparable to published E5 numbers. **gte-small** (and most BERT GGUF demos here) do **not** use this convention —
pass the raw sentence.

`samples.EmbeddingsHelloWorld` prepends `query: ` when the model path looks like E5. Pass
`models/gte-small.Q2_K.gguf` as the first argument if you want the tiny English GGUF without a prefix.

### Inside one `embed` call (pipeline)

```text
text
  → tokenize (WordPiece, or Unigram SentencePiece for XLM-R / E5)
  → wrap [CLS] / [SEP] when present, else <s> / </s>
  → token + position + token-type embeddings
  → LayerNorm
  → N × bidirectional transformer block
        (self-attention over the full sequence + GELU FFN; post-LN style)
  → mean pool over sequence length  →  one vector of size hidden_size
  → L2 normalize
  → float[]
```

Weights may stay **GGUF-packed** and dequantize on use, like other GGUF graphs (chapter 7a). Supported quants for
common small embedding files include paths that needed **Q3_K** / **IQ4_NL** dequant (**since 1.1.0**).

**Tokenizer:** GGUF `tokenizer.ggml.model=bert` → WordPiece / greedy longest-match (chosen inside `Tokenizer` when the
GGUF loads — not a public enum apps set), often with lowercase and metaspace-style pieces.

**In the code (graph — internal):** `models.internal.BertForEmbedding`; blocks use `layers.BidirectionalAttention`
(not the causal `Attention` used for chat); norms via `Norms.LayerNorm`; pooling and L2 live at the end of `encode`.
**Public surface:** `LlmModel.isEmbeddingModel()` / `embed(…)` / `modalities()` (`text->embedding`; same as
`usableModalities()` on this path).

### What you use the vectors for

- **Semantic similarity** between two strings (REPL in `Example` / `EmbeddingsHelloWorld`).
- **Dense RAG** (**since 1.1.0**, chapter 17): embed every corpus chunk once, embed the question, rank by cosine —
  `DenseRagIndex` or hybrid BM25+dense via `RagFactory.withEmbeddings`.

There is **no** chat template, KV cache, or next-token sampler on this path. Lifecycle is still
`LlmModel` as `AutoCloseable`: close when finished; for hybrid RAG, keep the encoder open while the index is in use.

### Limits (honest)

- **BERT embeddings:** GGUF or ONNX BERT-encoder graphs (`bert` / `roberta` / `xlm-roberta` when names map) — **not** a full Hugging Face safetensors
  BERT directory load yet.
- GGUF embedding files still need a **supported** GGML dtype (ch. 7a table); exotic quants fail at load.
- Dense retrieval is an **in-process linear scan**, not an ANN index.
- Embedding quality depends on the checkpoint (gte-small is small and English-oriented; E5 is multilingual but wants
  the `query:` / `passage:` prefixes above).

### Summary

> **Since 1.1.0, load a BERT-family embedding GGUF with `LlmModelFactory.make`. Since 1.3.0, `LLM.builder(model)` then
> `LLM.embed` uses the shared CPU pool; `LlmModel.embed` remains a sequential shortcut (dense RAG indexing). Vectors are
> L2-normalized (bidirectional encoder + mean pool). E5-style checkpoints still need their `query:` /
> `passage:` text prefixes; the library does not insert them.**

**Further reading:** original BERT — [Devlin et al. (arXiv)](https://arxiv.org/abs/1810.04805); sentence embeddings /
GTE family on Hugging Face (search “thenlper/gte-small”); E5 prefixes —
[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small); this guide **chapter 8**
(bidirectional vs causal attention), **chapter 17** (dense / hybrid RAG).


---

## 7c. ONNX weight folders and Llama (**since 1.1.0**)

Chapter 7 described **safetensors** as the usual crate of named float tensors inside an HF folder. **Since 1.1.0**,
the same folder shape may instead ship **ONNX** weight files (`model.onnx`, `model_fp16.onnx`, … under the folder root
or an `onnx/` subfolder). The public API does **not** change: still `LlmModelFactory.make(folder)`.

### What this engine does with ONNX (Tier A)

This is **weight import**, not an ONNX Runtime session:

| Does | Does not |
|------|----------|
| Read **graph initializers** (the learned tensors) | Execute the ONNX compute graph / operators / QDQ |
| Map names onto the existing Java causal / BERT graphs | Depend on native ORT |
| Prefer **safetensors** if both formats are present | Treat every community `.onnx` as loadable (see filters below) |

**In the code:** `models.llmcontainer.OnnxTransport` (file pick + protobuf + dtype decode) → `ArchitectureProcessor.bind` → `fill` (name remap / MatMul transpose) →
`ModelLoader.assembleFromNamedTensors` → `ArchitectureProcessor.createCausal` / `createEmbedding`
(Piper voices use `createSynthesis` — chapter **7e**).

### Supported formats / restrictions (ONNX)

**Where files may live:** folder root or `onnx/` subfolder.

**Preferred names** (first match wins): `model.onnx`, `model_fp16.onnx`, `decoder_model_merged.onnx`,
`decoder_model.onnx`, `encoder_model.onnx`; otherwise the first sorted allowed `*.onnx`.

| Allowed | Rejected / skipped |
|---------|-------------------|
| Float initializers: **FLOAT**, **FLOAT16**, **BFLOAT16**, **DOUBLE** → float32 | Filename contains `_q4`, `_int8`, `_uint8`, `_bnb4`, `_quantized`, or `with_past` (community quant / KV-past exports) |
| `make(Path)` may follow **external_data** sidecars next to the `.onnx` | **`ModelFileSource` / classpath** loads: `external_data` → `ModelLoadException` (use `make(Path)`) |
| File size up to **8 GiB** on disk; in-memory buffer path capped at **~2 GiB** | Empty initializer list; float8 / nibble / unknown TensorProto weight types (fail loud) |
| Architectures that map to Qwen3 / Gemma3 / **Llama** / BERT embedding schemas | **LFM2-from-ONNX**; **Gemma 4** (safetensors QAT only, ch. 7); QDQ / int8 weight graphs; running the ONNX op graph. **Piper** voices are a separate synthesis path (ch. **7e**), not this chat/BERT remap. |

INT/UINT/BOOL/STRING/COMPLEX initializers and float **scalars** are **skipped** as graph constants (not weight
tensors). See the TensorProto table below.

### Which architectures

| Checkpoint family | Graph | Typical demo |
|-------------------|-------|--------------|
| Qwen3 / Gemma3 / **Llama** | Causal chat (`LLM.builder`) | [SmolLM2-135M-Instruct-ONNX](https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX) (ChatML), [Tiny-LLM-ONNX](https://huggingface.co/onnx-community/Tiny-LLM-ONNX) (base / completion toy) |
| BERT-style | Embeddings (`LLM.embed` / `LlmModel.embed`) | [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) ONNX (`models/download-multilingual-e5-small.sh`; prepend `query:` / `passage:` — ch. **7b**); [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) ONNX (`models/download-xlm-roberta-base.sh`) |
| **Piper** voice (**since 1.3.0**) | Text→audio (`LLM.synthesize`) | Official Piper `*.onnx` + `*.onnx.json` (chapter **7e**) — not a chat/BERT remap |

**Llama** here means the Llama-style stack assembled by `LlamaForCausalLM`: RMSNorm, RoPE, GQA, SiLU MLP, **without**
Qwen’s extra Q/K head norms. Tiny-LLM and SmolLM2 Instruct ONNX both declare `LlamaForCausalLM` in `config.json`.
**Gemma 4 text QAT** is **safetensors only** (packed int2/4/8; not ONNX / GGUF) — chapter 7.

**Chat vs base:** instruct checkpoints (SmolLM2 Instruct) ship a ChatML `chat_template` and belong in the Example
**chat** menu. Base / completion toys (Tiny-LLM, base SmolLM2-135M-ONNX) only continue text; wrapping them in
`user:` / `assistant:` chat prompts produces nonsense — the menu labels them **base**.

### TensorProto types (what loads, what skips, what fails)

`OnnxDataTypes` catalogs ONNX `TensorProto.DataType` codes:

| Kind | Examples | Behavior |
|------|----------|----------|
| Loadable float | FLOAT, FLOAT16, BFLOAT16, DOUBLE | Decode to float32 weights |
| Skip | INT/UINT*, BOOL, STRING, COMPLEX*, UNDEFINED; float **scalars** | Graph constants — ignore |
| Fail loud | FLOAT8*, INT4/UINT4, FLOAT4, unknown codes | `ModelLoadException` (no silent drop of weights) |

transformers.js / onnx-community exports often store MatMul `B` as `[in, out]` under anonymous `onnx::MatMul_*`
names; the loader remaps via MatMul **node paths** and **transposes** to PyTorch Linear `[out, in]`. Raw payloads may
live in protobuf field **9** (legacy) or **13**.

### Download scripts

- Chat ONNX demo: `models/download-smollm2-135m-instruct-onnx.sh` → `models/SmolLM2-135M-Instruct-ONNX/`
- Base ONNX smoke test: `models/download-tiny-llm-onnx.sh` → `models/Tiny-LLM-ONNX/`
- Multilingual embeddings: `models/download-multilingual-e5-small.sh` → `models/multilingual-e5-small/`
  (Unigram tokenizer; prepend `query:` / `passage:` — ch. **7b**)
- XLM-RoBERTa-base ONNX: `models/download-xlm-roberta-base.sh` → `models/xlm-roberta-base/`
  (`onnx/model.onnx`; mean-pooled encoder, no E5 prefixes)

```java
// Instruct / ChatML folder (same factory as safetensors)
try (LlmModel model = LlmModelFactory.make(Path.of("models/SmolLM2-135M-Instruct-ONNX"));
     LLM llm = LLM.builder(model).maxModelLen(2048).build()) {
  String answer = llm.chat(256).send("What is 2+2?").answer();
}

// Tiny-LLM-ONNX is a base / completion toy — use complete / generateTokenIds, not chat templates:
// try (LlmModel toy = LlmModelFactory.make(Path.of("models/Tiny-LLM-ONNX"));
//      LLM llm = LLM.builder(toy).build()) {
//   String cont = llm.complete("Once upon a time");
// }
// Linear demo: samples.NextTokenHelloWorld
```

### Summary

> **Since 1.1.0, an HF folder may use ONNX initializers instead of safetensors. The engine extracts weights only (no
> ORT), applies file-name and TensorProto filters, builds Qwen3 / Gemma3 / Llama / BERT graphs as usual, and rejects
> float8/nibble weight types, unsupported community quant exports, LFM2-from-ONNX, and Gemma 4 (safetensors QAT only)
> explicitly.**

**In the code:** `models.llmcontainer.OnnxTransport` reads graph initializers (no ORT); `models.llmcontainer.LoadProgress` reports decode
with the same percent/ETA bar as safetensors and GGUF; `ModelLoader.assembleFromNamedTensors` matches names into
`WeightBag`. Application entry is still `LlmModelFactory.make(folder)`.

**Further reading:** [ONNX](https://onnx.ai/); community exports such as
[onnx-community/SmolLM2-135M-Instruct-ONNX](https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX).


---

## 7d. Whisper speech-to-text (**since 1.3.0**)

Chapters 7–7c were about **text in, text (or a vector) out**. **Since 1.3.0**, the same factory also loads a third
kind of network: **Whisper**, which turns **audio** into **text**.

The demo crate is Hugging Face safetensors for [openai/whisper-base](https://huggingface.co/openai/whisper-base)
(`models/download-whisper-base.sh`) or the smaller `whisper-tiny`. CTranslate2 `model.bin` folders, Whisper GGUF, and
Whisper ONNX are refused.

### What “speech-to-text” means here

| | Causal chat LM | Whisper ASR |
|---|---|---|
| Question it answers | “What token comes **next**?” | “What words were **spoken** in this clip?” |
| Input | Token ids | 16 kHz mono PCM (WAV decoded, or already-resampled samples) |
| Typical API here | `LLM.builder` → `chat` / `generate` | `LLM.builder` → `transcribe` |
| Output | Token ids / text | Transcript string |
| `usableModalities()` | text→text | audio→text |

There is **no** chat template, KV cache, or next-token sampler on this path. Decode is **greedy** (highest logit),
multilingual, with optional language. Clips longer than 30 s are split into 30 s chunks.

### How this project loads and runs it

1. **Same factory:** `LlmModelFactory.make(pathToWhisperFolder)`.
2. **Detect:** `model.isSpeechModel()` is true; `architectureName()` is `whisper`.
3. **Open an engine:** `LLM.builder(model).build()` — same `cpuThreads` / matmul pool as chat (Linear, attention, stem Conv1d).
4. **Transcribe:** `llm.transcribe(wavBytes)` (in-memory RIFF/WAVE), `llm.transcribe(path)`, or `llm.transcribe(pcm, sampleRate)`.
5. **Language:** pass a `Locale` (`null` / `Locale.ROOT` = auto-detect; region is ignored — `Locale.US` is English).

```java
try (LlmModel model = LlmModelFactory.make(Path.of("models/whisper-base"));
     LLM llm = LLM.builder(model).build()) {
  String text = llm.transcribe(wavBytes);                    // uncompressed WAV in memory
  String en = llm.transcribe(wavBytes, Locale.ENGLISH);      // optional hint
}
```

`LlmModel.transcribe` remains a sequential shortcut (no engine pool). Prefer `LLM.transcribe` when you already opened
an engine.

**In the code:** `LlmModelFactory` → `SpeechArchitecture` / `WhisperProcessor` → `WhisperForAsr`; public surface
`LLM.transcribe` / `LlmModel.transcribe`; sample `TranscribeHelloWorld`; `Example` opens a WAV session (`/tone`, `/lang`).

### Inside one `transcribe` call (pipeline)

```text
WAV bytes or PCM
  → mix to mono, resample to 16 kHz
  → log-mel spectrogram (30 s windows)
  → encoder (bidirectional transformer)
  → greedy decoder (language token, then text tokens)
  → tokenizer decode → String
```

**Limits (honest)**

- Uncompressed WAV or raw PCM only — no MP3, no streaming, no word timestamps, no VAD, no beam search.
- Greedy decode only.
- Gemma 4 audio towers in a chat crate are still skipped; this path is a dedicated Whisper checkpoint.

### Summary

> **Since 1.3.0, load Hugging Face Whisper safetensors with `LlmModelFactory.make`, then `LLM.builder` and
> `LLM.transcribe` on WAV bytes, a WAV file, or 16 kHz PCM. Language is optional (`Locale`; `null`/`ROOT` = auto).
> CTranslate2, Whisper GGUF, and Whisper ONNX are out of scope.**

**Further reading:** [Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*](https://arxiv.org/abs/2212.04356);
Hub [openai/whisper-base](https://huggingface.co/openai/whisper-base).


---

## 7e. Piper text-to-speech (**since 1.3.0**)

The fourth graph kind is the inverse of Whisper: **text in, audio out**. **Piper** voices ship as an ONNX file plus a
sidecar `*.onnx.json` (phoneme ids, sample rate, speaker). This engine still reads **initializers only** — no ONNX
Runtime, no JNI, no libespeak-ng.

Demo crates: `models/download-piper-en-lessac-medium.sh` (US English) and
`models/download-piper-ru-irina-medium.sh` (Russian). Sample `SynthesizeHelloWorld` prefers Lessac when both folders
exist.

### What “text-to-speech” means here

| | Causal chat LM | Piper TTS |
|---|---|---|
| Question it answers | “What token comes **next**?” | “What **waveform** speaks this sentence?” |
| Typical API here | `LLM.builder` → `chat` | `LLM.builder` → `synthesize` |
| Output | Text | Uncompressed WAV bytes (PCM16 little-endian, mono) |
| `usableModalities()` | text→text | text→audio |

`synthesize` returns the file contents. Write the `byte[]` only if you need a `.wav` on disk.

### How this project loads and runs it

1. **Same factory:** `LlmModelFactory.make(voiceFolder)` (detects `*.onnx` + `*.onnx.json`).
2. **Optional espeak-ng-data:** `LlmModelFactory.open(path).optionalData(LlmOptionalData.ESPEAK_DATA, dataDir).make()`.
   Default directory is `{model}/espeak-ng-data`. A missing or incomplete folder is **ignored** — synthesize still runs.
3. **Open an engine:** `LLM.builder(model).build()` — same CPU pool; 1-D conv / conv-transpose split output channels.
   Non-chat engines skip KV paging (a large heap no longer overflows `int` when sizing chat KV for a voice with no
   transformer layers).
4. **Speak:** `byte[] wav = llm.synthesize("Hello world");`

```java
Path voice = Path.of("models/piper-en-lessac-medium");
try (LlmModel model = LlmModelFactory.open(voice)
         .optionalData(LlmOptionalData.ESPEAK_DATA, voice.resolve("espeak-ng-data"))
         .make();
     LLM llm = LLM.builder(model).build()) {
  byte[] wav = llm.synthesize("Hello world");
}
```

**In the code:** `SynthesisArchitecture` / `PiperProcessor` → `PiperForTts`; G2P in `models.internal` espeak readers;
sample `SynthesizeHelloWorld`; `Example` lists Lessac and Irina.

### Grapheme-to-phoneme (how letters become phones)

Order of preference:

1. **`dictsource`** (`*_list` / `*_rules`) when present for the voice language — including suffix/prefix `S`/`P` and
   number fragments.
2. **Compiled** `phontab` + `{lang}_dict` for listed words and `$N` stress-only entries (used together with source files
   when both exist).
3. **Letter-to-sound** fallback (English lexicon + digraphs; Russian palatalization, vowel reduction, destressing,
   voicing assimilation). Russian letter G2P emits espeak phones for ш/ж/ы (`ʃ`/`ʒ`/`y`), not academic retroflex IPA.

Download scripts copy compiled dicts from a system espeak-ng-data when present. Packaged `ru_dict` omits some
high-frequency lemmas (`это`, `мама`).

### Inside one `synthesize` call (pipeline)

```text
text
  → G2P (dictsource / compiled dict / letter tables) → phoneme ids
  → VITS reverse (duration, residual couplings with channel flips, WaveNet skips)
  → HiFi-GAN vocoder (Conv / ConvTranspose geometry from the ONNX graph)
  → PCM16 LE mono WAV bytes
```

Conv vs ConvTranspose weight axis order differs (`[out,in/g,k]` vs `[in,out/g,k]`). Spatial attributes (stride, pads,
dilation, groups, output padding) come from the consuming ONNX node (`ConvLayout`), not from a voice-name guess. Irina
uses ResBlock2 (`convs.0`/`convs.1`) with graph dilations.

**Limits (honest)**

- One utterance → one WAV; no streaming, no SSML, no multi-speaker picker beyond what the voice JSON already encodes.
- Compiled espeak rules bytecode is not a full MatchRule VM.
- Official Piper ONNX exports load; exotic renamed graphs may still need more aliases.

### Summary

> **Since 1.3.0, load a Piper voice folder (`*.onnx` + `*.onnx.json`) with `LlmModelFactory.make`, optionally point
> `optionalData(ESPEAK_DATA)` at espeak-ng-data, then `LLM.builder` and `LLM.synthesize` for WAV bytes. Missing
> espeak data is ignored. G2P prefers dictsource, then compiled phontab/dict, then letter-to-sound.**

**Further reading:** [Piper](https://github.com/OHF-Voice/piper1-gpl); [Kim et al., VITS (arXiv)](https://arxiv.org/abs/2106.06103);
[espeak-ng](https://github.com/espeak-ng/espeak-ng).


---

## 8. Attention: kinds of looking-back, and how they work

Attention is the part people mean when they say the model “pays attention” to something you wrote earlier. It is not a
spotlight of consciousness. It is a **rule for mixing other places in the text into this place**.

There is not only one kind. Models differ in *who may look at whom*, *how many separate glances run in parallel*, and
*how thriftily they store the KV cache*. This chapter defines those variants, then states which ones this
project actually uses.

### The shared recipe (every kind below starts here)

Suppose the text so far is:

> Mary gave Susan a book because she

At *she*, the model builds three notes from that position’s portrait:

| Note      | Everyday question                                      |
|-----------|--------------------------------------------------------|
| **Query** | What am I looking for *from here*?                     |
| **Key**   | How should *this* place advertise itself to searchers? |
| **Value** | If someone chooses me, what content should they take?  |

It compares **this Query** with the **Keys** of allowed other places. Strong matches contribute more of their
**Values**. The blend updates the portrait for *she*.

Library metaphor: Query = catalog search; Key = spine card; Value = the book you pull when the card matches.

So attention always means:

> For this place, **how much should each allowed other place influence me?** Then take that blend as part of my
> updated meaning.

The *kinds* of attention differ mainly in the word **allowed**.

**Further reading:** original multi-head attention —
[Vaswani et al.](https://arxiv.org/abs/1706.03762); gentle annotated code —
[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).


---

### Kind 1 — Self-attention (look within the same text)

**Self-attention** means Queries, Keys, and Values all come from **the same passage** you are processing. The sentence
attends to itself. That is what chat and completion models do for your prompt and reply.

**Cross-attention** (mentioned only for contrast) would let one text attend to *another* text — e.g. a translator’s
decoder looking at the source sentence. Classic encoder–decoder machines used that. **This project’s Qwen3 / Gemma3
paths are self-attention only** — one stream, looking at itself.

---

### Kind 2 — Causal (masked) attention — “no peeking ahead”

When the job is to **write the next word**, a position may look at itself and what came **before**, never at the future.
That restriction is **causal** (or *causal masked*) attention.

```text
  positions:  1    2    3    4    5
              The  cat  sat  on   the
                                ↑
                         position 5 may look at 1…5
                         not at words that do not exist yet
```

Without this mask, the model could “cheat” by reading the answer it has not written. Bidirectional models (like the
BERT-style readers in **chapter 7b**, **since 1.1.0**) allow looking left *and* right because their job was understanding
a finished sentence, not continuing it. **Language models that generate text use causal self-attention.** Chat and
completion in this project do too; embedding GGUFs use bidirectional attention instead.

---

### Kind 3 — Multi-head attention — several glances at once

Instead of one comparison, the model runs **several attentions in parallel**. Each is a **head**.

Think of several readers of the same paragraph: one watches grammar, one watches names, one watches tone. Nobody assigns
those jobs by hand; training pushes heads toward different habits. Afterwards the glances are **merged** into one
portrait.

```text
  same passage
      │
      ├── head 1  ──► glance A
      ├── head 2  ──► glance B
      ├── head 3  ──► glance C
      └── …
      │
      ▼
   merge → updated portrait
```

**Full multi-head attention (MHA)** gives every head its **own** Key and Value notebooks. Rich, but memory-heavy when
texts grow long.

---

### Kind 4 — Sharing notebooks: MQA and GQA (thrift)

Keys and Values dominate memory during long chats (the “notebooks” of chapter 12). Designers invented thriftier layouts:

| Name                    | Idea                                          | Everyday picture                       |
|-------------------------|-----------------------------------------------|----------------------------------------|
| **MHA** (multi-head)    | Each head has its own Keys and Values         | Every reader brings a private notebook |
| **MQA** (multi-query)   | All Query heads share **one** Key/Value pair  | Many searchers, one shared catalog     |
| **GQA** (grouped-query) | Several Query heads share one Key/Value group | Small reading groups share a notebook  |

```text
  MHA:   Q Q Q Q     each with its own K V
         K K K K
         V V V V

  GQA:   Q Q Q Q     two Queries share each K V
         K   K
         V   V

  MQA:   Q Q Q Q     all Queries share one K V
         K
         V
```

**This project uses GQA** whenever the model config says there are fewer key/value heads than query heads (common in
Qwen3 and Gemma3). Same looking-back idea; fewer duplicate notebooks; faster long answers on limited machines.

**Further reading:** [Ainslie et al., *GQA*](https://arxiv.org/abs/2305.13245); extreme sharing was earlier called
multi-query attention in
[Shazeer, *Fast Transformer Decoding*](https://arxiv.org/abs/1911.02150).


---

### Kind 5 — Global vs sliding-window (local) attention

Even with the causal rule, a position might still look back over a **very long** past — the whole prompt. That is
**global causal** attention (within the past).

**Sliding-window** (local) attention narrows the view: each position may look back only about *W* tokens (a window), not
the entire history.

```text
  Global causal (past only):

  … 1 2 3 4 5 6 7 8 9 …
                  ↑ position 9 may see 1…9

  Sliding window of width 4:

  … 1 2 3 4 5 6 7 8 9 …
                  ↑ position 9 may see only 6…9
```

Why bother? Long global attention is expensive (every new word glances at everything). Local windows are cheaper and
still capture nearby grammar and recent facts. Distant facts must be carried forward in the portraits from earlier
layers — or refreshed by occasional **global** layers.

**Gemma3 in this project** often **mixes** layer types: some reading rooms use a sliding window; others stay global. The
blueprint (`layer_types`, `sliding_window`) decides per room. **Qwen3** here is typically global causal GQA in every
room.

Local layers may also use a different RoPE base (`rope_local_base_freq`) than global ones — a Gemma detail so near and
far matching do not fight each other (chapter 5 / RoPE section below).


---

### Kind 6 — Prefill attention vs decode attention (same rule, different workload)

The *rule* (causal, heads, GQA, window) stays the same; the *shape of the work* changes:

| Phase       | What attention does                                                               | Everyday picture                                     |
|-------------|-----------------------------------------------------------------------------------|------------------------------------------------------|
| **Prefill** | Many positions at once; each looks back over the prompt (and fills notebooks)     | Read the whole letter carefully once                 |
| **Decode**  | Usually **one** new position; it looks over past Keys/Values already in notebooks | Write the next sentence using notes you already took |

So “types” of attention in engineering talk sometimes means this **phase**, not a different philosophy. This engine
implements both. **Chapter 12** is the home of the notebooks: they store each position’s **Key** and **Value** (after
RoPE on K), not Query, not chat text, and not weights; decode then attends those pages in place.

---

### Order still matters (RoPE) for all these kinds

Before Query and Key are compared, **RoPE** (*Rotary Position Embedding*) is applied.

**What is rotated?** Not the position number itself. For each token at index `p`, RoPE takes the Query (and Key) vector
of every head, splits each head vector into pairs of components, and **rotates those 2-D pairs** by an angle that grows
with `p` (angles come from frequencies derived from `rope_theta`). Value is left as produced by the linear projection in
this port.

**Why bother?** After rotation, the dot-product that scores “how well does this Query match that Key” depends on both
*what* the tokens are and *how far apart* they sit. “dog bites man” is no longer interchangeable with “man bites dog”
for the matcher. Relative distance is encoded in the geometry of Q and K, without adding a separate learned position
table to the token embedding.

```text
  Q_p, K_p  --RoPE(p)-->  Q'_p, K'_p
  score(i, j) uses Q'_i and K'_j   // sensitive to (i − j) as well as content
```

**Where:** built once in `Norms.RotaryEmbedding` (cos/sin cache); applied each layer in
`Qwen3Attention#forward` / `Gemma3Attention#forward` via `rotaryEmb.forward(positions, q, k)` before
`Attention#forward`. Gemma may use a different `rope_theta` / `rope_local_base_freq` for sliding vs global layers; the
idea is the same.

Config fields: chapter 5 (`rope_theta`, …). Formal RoPE and attention scoring: **chapter 10**. Contrast with **token
embedding**: also chapter 10.

**Further reading:** [Su et al., *RoFormer* / RoPE](https://arxiv.org/abs/2104.09864).

---

### Map: what this Java project actually runs

| Feature                                   | In this project?                                  |
|-------------------------------------------|---------------------------------------------------|
| Self-attention                            | Yes                                               |
| Causal (no future)                        | Yes                                               |
| Multi-head Queries                        | Yes                                               |
| GQA (shared KV groups)                    | Yes, when the model config says so                |
| MQA (extreme sharing)                     | Only if a model’s config collapses to one KV head |
| Sliding-window / local layers             | Yes, for Gemma layers marked as such              |
| Global layers                             | Yes (Qwen; some Gemma layers)                     |
| Cross-attention to a second text          | No                                                |
| Bidirectional BERT-style                  | Yes, for embedding GGUFs only (**since 1.1.0**; ch. 7b) |
| Fancy GPU kernels (flash-attention, etc.) | No — CPU math with SIMD / panel GEMV; no GPU path     |

---

**In the code (kinds this port runs):** causal self-attention + GQA/MQA geometry in `layers.Attention` for chat LMs;
bidirectional self-attention in `layers.BidirectionalAttention` for BERT embedding GGUFs (**since 1.1.0**, chapter 7b);
Gemma sliding window / global via model config in `Gemma3ForCausalLM`; RoPE in `Norms.RotaryEmbedding` (causal path);
prefill vs decode branches inside `Attention.forward` (chapter 16).

---

### Attention is not yet “the answer”

Attention updates **inner portraits**. It does not print words. Words come later, when the final portrait is scored
against the vocabulary and one token is drawn. Attention is **inward rereading**; speaking aloud is a later stage.

### After attention: the private rewrite

Each reading room also contains a large feed-forward block (MLP):

1. Attention — *consult allowed places in the current passage.*
2. MLP — *rewrite this position using trained habits beyond a simple glance.*

Both happen in **every** room, stacked many times.

**In the code:** `Attention.forward`, `storeKvCache`, `attendRange`; model path `Qwen3Attention.forward` /
`Gemma3Attention.forward` / `Lfm2Attention.forward` (plus LFM2 short-conv layers; chapter 16).

### One sentence to keep

> All generating models here use **causal self-attention**; they usually run it as **many heads with shared KV
> notebooks (GQA)**; Gemma may **alternate local windows and global views**; the same rule is heavy at **prefill** and
> lighter at **decode** thanks to notebooks.

---

## 9. The thinking process: how it is organized and how it works with the model

People say models “think.” That single English word hides several different mechanisms. This chapter separates them,
shows how they are **organized in time**, and explains how they **use** the loaded model (weights, attention,
notebooks) — including what this Java chat path does with visible “thinking” text.

### First distinction: three senses of “thinking”

| Sense | Name in this guide                   | What it really is                                                                              | Visible as text?        |
|-------|--------------------------------------|------------------------------------------------------------------------------------------------|-------------------------|
| **A** | Silent inner work                    | The full stack of attention + rewrites for every next-token step                               | No                      |
| **B** | Written reasoning (chain of thought) | Extra tokens that *narrate* steps, then become part of the past                                | Yes, in the reply       |
| **C** | Tagged scratchpad                    | Sense B wrapped in markers like `<think>…</think>` so software can split “notes” from “answer” | Yes, if the UI shows it |

Sense A **always** runs. Senses B and C are optional styles of *output*. They are not a second engine. They are more
language produced by the same engine — language that later attention can reread.

```text
  ALWAYS (Sense A)                    SOMETIMES (B / C)
  ────────────────                    ─────────────────
  numbers walk the layers             model also emits words
  → scores → pick one token           that look like “reasoning”
                                      those words join the past
                                      → help later Sense A steps
```

---

### Sense A in detail — silent inner work, organized as a pipeline

There is no separate “thinking module” beside the model. **Using the model once to propose the next token** *is* the
thinking step. Organization is a fixed pipeline:

#### Step A1 — Text is already numbers

The prompt (and any reply so far) exists as token ids. Each id has a portrait from the embedding shelf loaded in chapter
4.

#### Step A2 — Prefill or decode chooses the workload

- **Prefill:** process the whole current prompt; fill Key/Value notebooks; produce the first new token.
- **Decode:** process mostly the newest token; reread notebooks; produce the next token.

Every later unit of Sense A is another decode step unless a new long prompt arrives.

#### Step A3 — Walk every reading room (layer), in order

For the token positions being computed, portraits enter **layer 1**, then **layer 2**, … up to **layer N** (N comes from
the blueprint — often dozens of rooms).

Inside **each** room, in order:

1. **Normalize** the stream (RMSNorm — volume control).
2. **Build Query / Key / Value** from the loaded projection shelves.
3. **Twist by position** (RoPE) so order matters.
4. **Attend** with the rules from chapter 8 (causal; GQA; maybe sliding window).
   - Write new Keys/Values into notebooks for positions being computed.
   - Mix past Values into the present according to match strengths.
5. **Mix heads** back to one stream; **add** onto the residual draft.
6. **Normalize** again.
7. **MLP rewrite** using the large feed-forward shelves; **add** onto the residual again.

Then the portrait enters the next room and repeats.

```text
  token portraits
       │
       ▼
  ┌─ layer i ─────────────────────────────┐
  │  norm → Q,K,V → RoPE → attention      │
  │       → + residual                    │
  │  norm → MLP → + residual              │
  └───────────────┬───────────────────────┘
                  ▼
             layer i+1 …
                  │
                  ▼
             final norm
                  │
                  ▼
             score all vocabulary scraps (LM head)
                  │
                  ▼
             sample one next token
```

That whole walk for **one** next token is one unit of Sense A. A ten-token answer is roughly ten such units after
prefill (plus the prefill itself).

#### Step A4 — Verdict and choice

After the last room, the final portrait is compared to every vocabulary scrap (the LM-head shelf). Sampling (chapter 11)
draws one token. That token is appended. Organization loops to A2/A3 until stop.

#### What Sense A can and cannot do

- It can **recombine** patterns stored in the fixed weights with the current page (via attention).
- It cannot **fetch** fresh world facts from outside this process.
- It does not “know that it is thinking”; it only transforms numbers.
- Depth of “thought” in Sense A means **more layers and more tokens of context**, not a wiser little person inside.

---

### How Sense A works *with* the loaded model

The loaded shelves are the **only** long-term habits. Organization of use:

| Loaded piece                     | Role in one Sense A step                                              |
|----------------------------------|-----------------------------------------------------------------------|
| Embedding table                  | Start portraits for token ids                                         |
| Per-layer Q/K/V and output mixes | Build and merge attention                                             |
| Per-layer MLP weights            | Private rewrite after attention                                       |
| Norm scales                      | Keep signals usable                                                   |
| Final LM head (or tied embed)    | Turn last portrait into next-token scores                             |
| Tokenizer / template             | Not inside Sense A math — they only prepare ids and later decode text |

Runtime notebooks (KV cache) are **not** loaded knowledge; they are the short-term memory of *this* page so Sense A need
not rebuild every past Key/Value each time.

So: **weights = long-term habits; notebooks = short-term notes for this conversation; Sense A = the procedure that
combines them every step.**

**In the code (Sense A):** `LLM.generate` / `step` → `Scheduler.schedule` → `Transformer.step` → layer
`forward` stacks on `Qwen3ForCausalLM` / `Gemma3ForCausalLM` / `Gemma4ForCausalLM` / `LlamaForCausalLM` /
`Lfm2ForCausalLM` (chapter 16).

---

### Sense B — thinking as writing (chain of thought)

Sometimes the model is trained or prompted to emit tokens like:

> First, the user asks for a sum. Two plus two is four. So I should answer “4”.

That narration is **still Sense A**, token by token. The difference is *content*: the tokens describe reasoning.

#### Why writing can help later steps

Once those tokens exist on the page, later attention (still Sense A) may look back at them. The model has given itself a
**scratch manuscript**. Multi-step questions often benefit because intermediate results become ordinary text the causal
glance can reuse.

```text
  time →

  Sense A produces: “First,”
  Sense A produces: “ add”
  Sense A produces: “ 2”
  …
  (those words are now in the past)
  Sense A at the end can attend to “2” and “2” and “add”
  Sense A produces: “4”
```

#### How Sense B is organized (and how it is not)

- There is **no** internal checklist engine unless the *text itself* lists checks.
- Style comes from **training** and from **prompt / system directions**.
- Longer Sense B uses more tokens, more notebook pages, more time — and can wander.
- Fluency of reasoning text ≠ truth. The model may imitate the *genre* of explanation while inventing steps.

**Further reading:** prompting models to write steps —
[Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903).

**In the code (Sense B):** no separate class — written reasoning is ordinary tokens from the same `Sampler` /
`generate` loop; later steps reread them via the KV notebooks like any other past text (chapter 16).

---

### Sense C — tagged scratchpad (how this project organizes visible thinking)

Sense C is Sense B with **stage directions** so software can separate “notes for the assistant” from “lines for the
human.”

#### The format this chat path expects (Qwen-style tagged scratchpad)

For **Qwen-style** chat (`Tokenizer.invitesThinking()`), the default system stage directions ask roughly:

1. Open with `<think>` … `</think>` for short private notes (intent, useful history, plan).
2. **Close** the tag before the user-visible answer.
3. Do not hide the only real answer inside the think block.

Example shape of one assistant turn:

```text
<think>
User wants a sum.
History has no conflicting numbers.
Plan: answer with 4.
</think>
4
```

**Other ChatML checkpoints** (e.g. many `lfm2` GGUFs) wrap turns as **ChatML** (`<|im_start|>role` …
`<|im_end|>`). Library default system text is empty; demos may set a short
`SampleChatPrompts.PLAIN_ASSISTANT_SYSTEM` without `<think>` rules. Think invitation follows vocab markers. See §7a.

Custom open/close markers are not Qwen- or GGUF-only: pass `ThinkTags` at load with
`LlmModelFactory.open(path).thinkTags(tags).make()` (**since 1.1.0**). That pair is frozen on the
`LlmModel` and inherited by every `LLM`. `ChatSession.thinkTags` / `RagSession.thinkTags` override one conversation.
`ChatReply.parse(raw)` without tags still assumes `<think>` / `</think>`.
Chat markup stripped from the visible answer is `ChatSpecials` (`open(path).chatSpecials(…)` /
`OPTION_CHAT_SPECIALS`); omitted options receive `ThinkTags.DEFAULT` and `ChatSpecials.DEFAULT` on the model.

#### How the program organizes Sense C around the model

```text
  1. Build chat history + system directions
  2. Apply chat template → one big prompt string
       (turn-based ChatFormat: no ChatML think invitation)
       (ChatML + think vocab: enableThinking / invitesThinking gate empty `<think>` seed + sample system text)
  3. Prefill + decode (pure Sense A) until the turn ends
  4. Decode tokens → raw assistant text
  5. AssistantParts.parse splits raw text into:
        thinking  = inside <think>…</think>
        answer    = after the closed tag
        thinkOpen = tag never closed (incomplete)
  6. ChatSession may:
        • stream TEXT_RAW (unparsed decode, think tags and chat specials kept)
        • stream TEXT_THINKING / TEXT_ASSISTANT (parsed channels; CLI streamTo prints these only)
        • salvage an answer from thinking if the visible part is empty
        • finish the turn for history / UI
```

Important: **the model does not call a `Think()` function.** It emits the characters `<`, `t`, `h`, … as ordinary tokens
if sampling chose them. Parsing happens **after** generation (and incrementally while streaming).

#### ChatFormat vs architecture (organization in this project)

|                                         | ChatML + think vocab | Turn-based (`<start_of_turn>`) | ChatML without think tags / hybrid GGUF |
|-----------------------------------------|----------------------|--------------------------------|-----------------------------------------|
| System directions about `<think>`       | App/sample owned (library default empty) | App/sample owned (usually empty) | App/sample owned (demos may use plain cue) |
| `invitesThinking` / default enable      | **true** when both `<think>` and `</think>` exist in vocab | **false** (no ChatML think seed) | Vocab-gated; ChatML via `<\|im_start\|>` even if GGUF omits `chat_template` |
| Reliable tagged scratchpad              | Encouraged by **sample** system text when used | Not relied on | Only if tags appear in vocab / output |
| Sense A (layers)                        | Architecture backend (`qwen3`, `llama`, …) | Architecture backend (`gemma3`, …) | e.g. `lfm2` hybrid short-conv + GQA |

So “thinking UI” is a **chat convention** (`ChatFormat` + vocab markers) on top of architecture backends.
`enableThinking` / `invitesThinking` are not a second brain switch; they only change how the **prompt string** is wrapped
before Sense A runs.

#### Streaming organization

While tokens arrive, the printer updates:

- text still inside an open `<think>` → thinking channel;
- after `</think>` → answer channel;
- if the tag never closes, the session may recover a visible answer from the notes (`salvageFromThinking`).

That is bookkeeping for humans. The model only ever produced one token stream.

---

### One turn, all senses together (organization in time)

Imagine the user asks: “What is 2+2?”

1. **Load** already happened earlier (fixed shelves).
2. **Template** wraps roles; system may invite Sense C.
3. **Prefill (Sense A)** reads the prompt through all layers; notebooks fill.
4. **Decode loop (Sense A):**
   - maybe emits `<think>` and note tokens (Sense C text appearing);
   - notebooks now include those note tokens; later steps can attend to them (B/C helping A);
   - emits `</think>` then `4`.
5. **Parse** splits notes vs answer for the UI.
6. Next user message rebuilds a templated prompt; Sense A runs again on that new page.

```text
  [system + history + user]  --prefill-->  notebooks filled
           │
           ▼
     decode token …  (Sense A)  → maybe “<think>…” notes
     decode token …  (Sense A)  → “</think>”
     decode token …  (Sense A)  → “4”
           │
           ▼
     parse → ChatReply(thinking, answer, thinkOpen)  + stats after generate
```

---

### How “thinking” is organized across multi-turn chat

- **History** in this project’s `ChatSession` stores the **visible answer** for each assistant turn — not the
  `<think>…</think>` scratchpad. The next prompt therefore normally **cannot** attend to prior Sense C notes unless you
  put them into history yourself.
- Within a **single** turn, note tokens still help: once emitted, later decode steps can attend to them before the turn
  ends.
- Each new turn **rebuilds** a templated prompt and runs Sense A again (KV notebooks for that generate fill for that
  request).
- Prefix caching may reuse identical early pages when openings match — an optimization, not a persistent mind object.
- There is **no** separate long-term thought log inside the weight shelves.

---

### Thinking and attention — how they cooperate

| Mechanism        | Relationship to thinking                                                    |
|------------------|-----------------------------------------------------------------------------|
| Causal attention | Lets each new token (including note tokens) look only at the past           |
| GQA / windows    | Same thinking pipeline; different cost and reach of the glance              |
| Notebooks (KV)   | Remember Keys/Values of prompt **and** of written notes already emitted     |
| MLP              | Rewrites portraits with trained habits during every Sense A step            |
| Sampling         | Chooses whether the next visible scrap is a note word, a tag, or the answer |

Written thinking helps **because** attention can reread it — not because a second thinker appears.

---

### What thinking is *not* (keep these sharp)

- Not a search of the internet (this project has no tools).
- Not guaranteed truth or self-knowledge.
- Not a human pause for reflection — only repeated next-token procedures.
- Not a diary written into the model files at load time.
- Not something you can open inside a layer to read English propositions; layers hold numbers.
- Sense C tags are not magic: if the model omits them, there is nothing to parse.

---

**In the code:** `ChatSession.send` → `applyChatTemplate` → `LLM.generate` → `AssistantParts.parse` → history keeps
`answer` only (chapter 16).

### A fair humanities summary of this chapter

> **Silent thinking (A)** is the organized walk through every loaded layer — attention plus rewrite — once per next
> token, using fixed weights and short-term notebooks.
> **Written thinking (B/C)** is more language produced by that same walk; once written, it becomes part of the page
> that later attention can use.
> This project’s chat layer **invites and parses** tagged scratchpads for Qwen-style dialogs; Gemma relies on silent
> work without that ceremony.
> Nowhere is there a ghost that thinks *beside* the model — only the model, used in a loop.

---

## 10. Tensors, embeddings, and the arithmetic of inference

Earlier chapters describe *what* the engine does. This chapter is the **math catalog** for this port: the data objects,
then each algebraic brick used in a decoder layer, with formulas that match the Java implementation (`tensor.Ops`,
`VectorMath`, `FloatKernels`, `Attention`, `Sampler`, `Norms`).

You can skim the notation and still follow later chapters; the point is that “linear,” “attention,” and “norm” stop
being slogans and become named equations with a home in the code.

### Map of a decoder layer (equations first)

One transformer block (after token embeddings) is approximately:

$$
\begin{aligned} \mathbf{h}' &= \mathrm{Attention} (\mathrm{RMSNorm} (\mathbf{h})) + \mathbf{h}, \\ \mathbf{h}'' &= \mathrm{MLP} (\mathrm{RMSNorm} (\mathbf{h}')) + \mathbf{h}'. \end{aligned}
$$

In this codebase the residual add is often **fused** with the next RMSNorm (`Ops.addRmsNorm` returns both the normalized
tensor and the updated residual stream). Prefill/decode and KV paging change *which* Keys/Values are visible, not these
formulas.

| Brick            | Role                                    | Primary code                          |
|------------------|-----------------------------------------|---------------------------------------|
| Embedding lookup | id → $\mathbf{x} \in \mathbb{R}^{H}$    | `EmbeddingKernel` via `VocabParallelEmbedding` |
| Linear / GEMM    | $W\mathbf{x}(+b)$                       | `LinearKernel` → `MatmulRuntime` / `FloatKernels.dot` |
| RMSNorm          | scale by inverse RMS                    | `Ops.rmsNorm` / `addRmsNorm`          |
| RoPE             | rotate Q/K by position                  | `Norms.RotaryEmbedding`               |
| Attention        | softmax-weighted Values                 | `layers.Attention`                    |
| Gated MLP        | SiLU or GELU-tanh gate × up             | `Ops.siluAndMul` / `geluPytorchTanh…` |
| Softmax          | scores → probabilities                  | `Ops.softmaxLastDim`; also in Sampler |
| Sampling         | temperature, top-k/p, Gumbel-style draw | `layers.Sampler`                      |

### Tensors — definition and notation

A **tensor** $T$ of **rank** $r$ (also called *order*) is an array whose elements are indexed by $r$ integers. Its
**shape** is $(n_1, n_2, \ldots, n_r)$. The number of scalar elements is

$$
\mathrm{numel} (T) = n_1 \cdot n_2 \cdots n_r.
$$

Common special cases:

| Rank | Name   | Notation (typical)              | Role in this model                            |
|------|--------|---------------------------------|-----------------------------------------------|
| 1    | vector | $\mathbf{h} \in \mathbb{R}^{H}$ | Hidden state of one position                  |
| 2    | matrix | $W \in \mathbb{R}^{m \times n}$ | Linear layer; embedding table                 |
| ≥3   | —      | e.g. batched activations        | Prefill batches, caches (exact layout varies) |

**Indexing convention in this guide.** Matrices are written with shape `[rows, cols]`. An embedding table
$E \in \mathbb{R}^{V \times H}$ has **one row per vocabulary id**. Row $t$ is written $E_{t,:}$ or $E[t]$.

**Contiguous storage.** Implementations store elements in a flat buffer (here: `float[]`) in **row-major** order: the
last axis changes fastest. For a matrix of shape `[V, H]`, the element at row $t$, column $j$ sits at flat index
`t * H + j`.

**Reshape** changes the shape tuple without changing $\mathrm{numel}$ or (usually) the underlying buffer. It does not
create new learned parameters.

**Weights vs activations.**

| Kind       | Lifetime                          | Examples                                            |
|------------|-----------------------------------|-----------------------------------------------------|
| Weight     | Loaded once; reused every request | `embed_tokens`, `q_proj`, RMSNorm scales, `lm_head` |
| Activation | Ephemeral per forward step        | Token hidden states, attention scores, logits       |

Both are `Tensor` instances; only weights come from `.safetensors` (chapter 7).

#### How `tensor.Tensor` represents this in Java

| Property         | Meaning                                               |
|------------------|-------------------------------------------------------|
| `data`           | Underlying `float[]` (float32 after load)             |
| `shape`          | `int[]` of axis lengths                               |
| `offset`         | Start index into `data` (views)                       |
| `numel` / `size` | Number of logical elements                            |
| `reshape`        | New shape, same buffer and offset, if `numel` matches |

Kernels in `Ops` write a **result tensor** (they do not mutate weight shelves). Elementwise work (add, mul, scale,
SiLU, GELU, RMSNorm, `tanhSoftcap`) goes through SIMD `FloatKernels` with a scalar tail when the Vector API is present.

```text
  flat index of E[t, j]  =  t * H + j     for shape [V, H]
```

**In the code:** `tensor.Tensor`, `Ops`, `VectorMath`, `FloatKernels` (chapter 16).

### Linear maps (matrix–vector / batched GEMM)

A **linear** (affine) map takes $\mathbf{x} \in \mathbb{R}^{n}$ and $W \in \mathbb{R}^{m \times n}$, optionally
$\mathbf{b} \in \mathbb{R}^{m}$:

$$
\mathbf{y} = W\mathbf{x} + \mathbf{b}
$$

(bias omitted when absent).

**Layout in this port.** Weights are stored as `[out, in]` = $[m, n]$: row $o$ is the weight vector for output channel
$o$. For one input row, output channel $o$ is the **dot product** of $\mathbf{x}$ with that row (plus bias). Batched
over `rows`, `MatmulRuntime` tiles over output channels (`TILE_N`) and input features (`TILE_K`), calling
`FloatKernels.dot` for each partial product — same math, cache-friendlier loops. Dense decode (`rows == 1`) uses
**panel GEMV**: the activation vector is loaded once and dotted across several output rows. Packed GGUF / QAT weights
dequantize one weight row, then dot (a fused dequant-dot without float row scratch is still open).

This pattern builds **Q / K / V**, the attention **output projection**, MLP **up / gate / down**, and the **LM head**.

**In the code:** `layers.Linear` → `LinearKernel` → `MatmulRuntime` / `FloatKernels.dot` (`Ops.linear` remains a
helper for ad-hoc calls).

### Embeddings — from discrete ids to vectors

#### Definition

The tokenizer only hands the model integer ids. A **token embedding** is the learned lookup table that turns each id
into a dense vector of length `hidden_size` — the starting description of that place in the text. Without it, attention
and the MLP would have nothing continuous to work on. What you obtain after lookup is not a verbal definition of the
token, only a fixed numeric row that later layers will rewrite; nearby rows may correlate with similar usage in
training, but the engine never needs that story to run.

Formally, a **token embedding** is a learned map

$$
\mathrm{emb} : \{0,1,\ldots,V-1\} \rightarrow \mathbb{R}^{H},
$$

implemented as $E \in \mathbb{R}^{V \times H}$. For token id $t$,

$$
\mathbf{x} = E_{t,:} \in \mathbb{R}^{H}.
$$

Equivalently $\mathbf{x} = E^{\mathsf{T}}\mathbf{e}_t$ for a one-hot $\mathbf{e}_t$, but the engine **copies row $t$**
(`EmbeddingKernel.gather` via `VocabParallelEmbedding`) — a gather, not a matmul.

$H$ = `hidden_size` (e.g. 1024 for Qwen3-0.6B, 640 for Gemma3-270M). $V$ = `vocab_size`.

#### Gemma embedding scale

On Gemma, right after the table lookup, every embedding vector is multiplied by a constant — the square root of
`hidden_size`. That is not another learned table; it is a fixed training convention so starting activations sit at the
scale the rest of the stack expects. Qwen’s path in this port skips that step.

After lookup, Gemma multiplies by $\sqrt{H}$ (`scaleEmbed`). Fixed from config; not a learned tensor.

#### Tied LM head

At the end of the stack the model must score every vocabulary id as a candidate next token. Those scores (logits) can
come from a **separate** LM-head matrix, or from **reusing the same embedding table** that started the forward pass
(**tied** embeddings). Tying saves a large block of parameters and is common on the small models this project documents;
untied heads keep input lookup and output scoring as two distinct matrices.

Logits $\boldsymbol{\ell} \in \mathbb{R}^{V}$ from last hidden $\mathbf{h}$:

| Arrangement | Idea                                                                                |
|-------------|-------------------------------------------------------------------------------------|
| Tied        | Reuse $E$: conceptually $\boldsymbol{\ell} = E\mathbf{h}$ (via linear / LM-head layout) |
| Untied      | Separate $W_{\mathrm{lm}} \in \mathbb{R}^{V \times H}$                              |

#### RoPE vs token embedding

People also call **RoPE** an “embedding,” but it is a different mechanism: it does not look up a vocab row. Instead it
rotates pairs of features inside Query and Key by an angle that depends on the token’s position, so relative order
affects attention scores. This port does **not** add older GPT-2-style learned vectors per absolute position on top of
the token table. Formal RoPE math is in the section below.

#### Where embedding runs

In this library the forward pass begins with the vocab table lookup (and Gemma’s optional scale), then the transformer
layers; RoPE runs inside attention on Q and K. When it is time to pick the next token, `computeLogits` scores the
vocabulary with the LM head — the same matrix as the embedding table when weights are tied. The diagram is only a map
onto those Java calls.

```text
CausalLM#forward
  → VocabParallelEmbedding#forward → EmbeddingKernel#gather
  → (Gemma) × √H
  → layers… (RoPE inside attention)
CausalLM#computeLogits
  → ParallelLMHead#forward → LinearKernel#apply   // tied or separate weights
```

**In the code:** `VocabParallelEmbedding` / `EmbeddingKernel`, `ParallelLMHead` / `LinearKernel`,
`Norms.RotaryEmbedding` (chapter 16).

### RMSNorm

For a vector $\mathbf{x} \in \mathbb{R}^{H}$ and learned scale $\mathbf{g} \in \mathbb{R}^{H}$:

$$
\mathrm{RMS} (\mathbf{x}) = \sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2 + \varepsilon}, \qquad \mathrm{RMSNorm} (\mathbf{x}) = \frac{\mathbf{x}}{\mathrm{RMS} (\mathbf{x})} \odot \mathbf{g}'.
$$

Here $\varepsilon$ is `rms_norm_eps`. The energy $\sum x_i^2$ is `VectorMath.sumSquares`.

**Gemma style:** stored weight $g_i$ is applied as $g'_i = 1 + g_i$ (HF convention). Otherwise $g' = g$.

**Fused residual:** `Ops.addRmsNorm` computes $\mathbf{s} = \mathbf{x} + \mathbf{r}$, then
$\mathrm{RMSNorm} (\mathbf{s})$, and returns `{normed, s}` so the residual stream stays correct.

**In the code:** `Ops.rmsNorm`, `Ops.addRmsNorm`, `Norms.RMSNorm`.

### Softmax and temperature

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

**Numerics:** compute $e^{z_i - \max_j z_j}$ then renormalize (`Ops.softmaxLastDim`, and inside `Sampler`).

**Temperature** $\tau > 0$ for sampling: use $z_i / \tau$ before softmax. Small $\tau$ → peaked; large $\tau$ → flatter.
This port rejects $\tau \rightarrow 0$ (pure greedy) in `SamplingParams`. Repeatable argmax is
`LLM.Builder.deterministic()` / `SamplingParams.deterministic()` (`topK = 1`), not temperature zero.

### Causal self-attention (the scoring math)

For one head, with Queries, Keys, Values $\mathbf{q}_t, \mathbf{k}_j, \mathbf{v}_j \in \mathbb{R}^{d}$
($d$ = `head_dim`):

$$
\alpha_{tj} = \frac{\exp\big (\langle \mathbf{q}_t, \mathbf{k}_j \rangle / s\big)} {\sum_{j' \in \mathcal{A} (t)} \exp\big (\langle \mathbf{q}_t, \mathbf{k}_{j'} \rangle / s\big)}, \qquad \mathbf{o}_t = \sum_{j \in \mathcal{A} (t)} \alpha_{tj}\, \mathbf{v}_j.
$$

- $\langle\cdot,\cdot\rangle$ is a **dot product** (`VectorMath.dot`).
- Scale $s$: typically $\sqrt{d}$; Gemma may use $\sqrt{q}$ where $q$ is `query_pre_attn_scalar` from config.
- Allowed set $\mathcal{A} (t)$: **causal** $j \le t$, optionally intersected with a **sliding window** of width $W$
  (Gemma local layers).
- **Multi-head:** several heads in parallel; outputs concatenated and mixed by $W_O$ (`o_proj`).
- **GQA / MQA:** several Query heads share one Key/Value group (`num_key_value_heads` ≤ `num_attention_heads`); this
  port sets `repeats = numHeads / numKvHeads`.

Q, K, V themselves come from linear maps of the (normalized) hidden state. Chapter 8 describes *kinds* of attention;
this section is the shared algebra.

**In the code:** `layers.Attention` (scores via `VectorMath.dot` × `scale`, then softmax-like normalization in the
attend loop); projections in the model’s attention module.

### RoPE — rotary positions on Q and K

For even head dimension $d$, split channels into $d/2$ pairs. At position $p$, pair $i$ is rotated by angle
$\theta_{p,i} = p \cdot \omega_i$ with

$$
\omega_i = \mathrm{base}^{-2i/d}.
$$

Here $\mathrm{base}$ is the config field `rope_theta` (or a local RoPE base for sliding layers).

If $(x_1, x_2)$ is one pair:

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}.
$$

Applied to **Q and K only** (not V, not the token embedding). Cos/sin are cached in
`Norms.RotaryEmbedding` as `[max_position, head_dim]`. Relative distance then influences attention scores without an
additive position table.

**In the code:** `Norms.RotaryEmbedding#forward(positions, q, k)` before `Attention#forward`.

### Gated MLP (SwiGLU / Gemma GELU-tanh)

After attention + residual, the feed-forward block expands features. This port uses a **gated** form:

1. Linear map to width related to `intermediate_size`, often producing a packed vector
   $[\mathbf{g}; \mathbf{u}] \in \mathbb{R}^{2h}$ (gate and up halves).
2. Activate the gate; multiply by up elementwise; project back to $H$.

**Qwen (SiLU / SwiGLU-style):**

$$
\mathrm{SiLU} (z) = \frac{z}{1 + e^{-z}}, \qquad \mathrm{MLP} (\mathbf{x}) = W_{\mathrm{down}}\big (\mathrm{SiLU} (\mathbf{g}) \odot \mathbf{u}\big).
$$

**Gemma (tanh GELU approx):**

$$
\mathrm{GELU}_{\mathrm{tanh}} (z) = \tfrac12 z \big (1 + \tanh\big (\sqrt{2/\pi}\, (z + 0.044715\, z^3)\big)\big),
$$

then $\mathrm{GELU}_{\mathrm{tanh}} (\mathbf{g}) \odot \mathbf{u}$, then down-project.

`Ops.siluAndMul` / `Ops.geluPytorchTanhAndMul` implement the activate×multiply on the packed last dimension (must be
even).

**In the code:** model MLP modules → those `Ops` methods → `down_proj` via `Linear` / `LinearKernel`.

### Sampling math (after logits)

Given logits $\boldsymbol{\ell} \in \mathbb{R}^{V}$:

1. **Temperature:** $z_i = \ell_i / \tau$.
2. **Softmax** → probabilities $p_i$.
3. **Top-k** (optional): zero all but the $k$ largest $p_i$; renormalize.
4. **Top-p** (nucleus, optional): keep the smallest prefix of sorted mass whose cumulative probability ≥ $p$; zero the
   rest; renormalize.
5. **Draw:** this port uses a **Gumbel-max–style** score $p_i / (-\log U_i)$ with $U_i \sim \mathrm{Uniform} (0,1)$
   and picks the argmax (equivalent in spirit to sampling from categorical $p$; see `Sampler`).

Stop token ids / `maxTokens` / `maxModelLen` / degenerate token-loops end the sequence in the scheduler, not in the
sampler. **Since 1.1.0**, `Sequence.hasDegenerateRepetition` stops soft loops (same-token streaks, exact repeated
blocks, overused n-grams) so tiny models cannot fill the whole budget with the same paragraph. Engine
`maxModelLen` is also capped by `config.json` `max_position_embeddings` so RoPE never walks past its cache.

**In the code:** `layers.Sampler`; defaults in `SamplingDefaults`. Narrative: chapter 11.

### Dot product and SIMD

Everywhere “score” or “linear partial sum” appears, the primitive is

$$
\mathrm{dot} (\mathbf{a},\mathbf{b}) = \sum_{i=0}^{n-1} a_i b_i
$$

on float slices (`FloatKernels.dot` / `sumSquares`), with a Vector API (SIMD) main loop that uses several independent
accumulators, plus a scalar tail. That is the numeric heart under attention scores and dense / packed linear dots via
`MatmulRuntime`. Paged KV attention **reads cache slots in place** (no per-step copy of K/V pages into a dense tensor).

### What this math is *not*

None of these maps *is* human understanding. After training they implement a deterministic (up to sampling noise)
procedure that often **imitates** fluent continuation. Keeping that distinction sharp is part of reading this project
honestly.

**In the code (full stack):** `tensor.Tensor` → `Ops` / `VectorMath` / `FloatKernels` → `Attention` / `Norms` /
`Sampler` (chapter 16).

**Further reading:** [Vaswani et al.](https://arxiv.org/abs/1706.03762); [RMSNorm](https://arxiv.org/abs/1910.07467);
[SwiGLU](https://arxiv.org/abs/2002.05202); [RoPE / RoFormer](https://arxiv.org/abs/2104.09864);
[nucleus sampling](https://arxiv.org/abs/1904.09751);
[softmax](https://en.wikipedia.org/wiki/Softmax_function).


---

## 11. Choosing a word: not always the most obvious one

After scoring the vocabulary, the program must **pick** one token.

Imagine a very large hat of slips of paper. Softmax writes how many copies of each slip go into the hat. Then:

- **Top-k** — keep only the *k* most popular slips; throw the rest away; refill proportions (`0` means “off”).
- **Top-p** — keep the smallest set of popular slips that together cover a large share of the hat (often about 0.9–0.95
  here); discard the long tail of oddities.

Then one slip is drawn. This project’s sampler uses a **Gumbel-max–style** draw over the remaining probabilities (not a
naive left-to-right walk of a cumulative table). Pure greedy decoding (temperature ≈ 0) is **rejected** by
`SamplingParams`. For the same tokens every run, call `LLM.Builder.deterministic()` (or
`SamplingParams.deterministic()` / `ChatSession.deterministic()` / `RagSession.deterministic()`):
that keeps only the highest-logit token (`topK = 1`) and does not draw from the RNG.

Default helpers (`SamplingDefaults.neutral()` / `forTokenizer`, and `SamplingParams.builder().build()`)
use temperature `0.6`, top-p `0.95`, 256 new tokens, and **top-k off** (`0`) for every tokenizer.
Product/family knobs (e.g. turn-based top-k 64) belong in the app or samples (`SampleChatPrompts`).

So the model is not forced to say the single most likely word every time. Controlled chance is why two answers to the
same question can differ — and why “creativity” settings exist in chat products.

You can ask for short or long answers by limiting **how many** tokens may be drawn before stopping. Special **end** /
stop token ids (from the tokenizer and optional `generation_config.json`) mean “the assistant considers this reply
finished.”

**Further reading:** nucleus (top-p) sampling —
[Holtzman et al., *The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751).

**In the code:** `SamplingParams` / `SamplingDefaults` / `LLM.Builder.deterministic()`; draw in `Sampler.forward`
(Gumbel-max, or argmax when `topK = 1`); stop ids, `maxTokens`, `maxModelLen`, and degenerate-loop checks
enforced in `Scheduler.postprocess` (chapter 16).


---

## 12. Why the program keeps a notebook of the past

Attention needs the Key and Value notes for **everything already seen**. Recomputing them from scratch for the whole
prompt on every single new word would be like rereading an entire novel each time you write the next sentence.

So the program keeps a **notebook** (the **KV cache**): once a position has been processed, its Key and Value are stored
and reused. Building an `LLM` reserves the blank pages; answering **writes** into them. This chapter says **what** is
stored, **what is not**, **how** a write and a read look, and **when** the notebook is thrown away.

### What “KV” means here

At each token position, every attention layer builds three views of that place’s portrait (chapter 8):

| View        | Everyday job                                      | Cached? |
|-------------|---------------------------------------------------|---------|
| **Query**   | “What am I looking for *from here*?”              | **No** — recomputed for the *current* token only |
| **Key**     | “How should this place advertise itself?”         | **Yes** — the address card |
| **Value**   | “If chosen, what content should the reader take?” | **Yes** — the page you pull |

**RoPE** (chapter 8 / 10) is applied to Query and Key **before** the Key is written. The cache therefore holds
**position-twisted Keys** and **untwisted Values**, ready for later dots. You do not store the raw hidden state or the
token id in these tensors — only those two attention views.

**Where the numbers live.** For each attention **layer** that owns a cache, this engine keeps two tensors:

```text
  kCache[layer], vCache[layer]   shape  [numBlocks, blockSize, numKvHeads, headDim]
```

- **`numKvHeads`** — how many Key/Value groups the model has (often fewer than Query heads: **GQA**, chapter 8).
- **`headDim`** — width of one head.
- **`blockSize`** — tokens per page (default **256**).
- **`numBlocks`** — how many such pages the engine reserved (see “How large the shelf is” below).

One **slot** is one token’s K (or V) for that layer: page number × page size + offset inside the page. Attention
indexes those slots **in place**; it does not copy whole pages into a dense “past K/V” tensor on every decode step.

**GQA.** Several Query heads share one Key/Value group, so the cache is sized by `num_key_value_heads`, not by
`num_attention_heads`. That is the main reason long chats stay feasible on a laptop heap.

**Gemma 4 shared-KV layers (since 1.1.0).** Some layers do **not** allocate their own pages. They **read** an earlier
same-type layer’s K/V (`writeKv = false`). The notebook is still Keys and Values — just one physical copy serving two
rooms.

### What the cache is *not*

| Not stored in KV | Where that lives instead |
|------------------|--------------------------|
| Learned **weights** | Immutable `LlmModel` (the library shelves) |
| Chat **strings** / roles | `ChatSession` history, re-templated each `send` |
| **Query** vectors of past tokens | Thrown away after that step; only the new token’s Q is built |
| **Logits** / sampled token ids | `Sequence` token list for this `generate` |
| BERT **sentence vectors** | Embedding models have **no** KV cache (chapter 7b) |

The notebook is **short-term arithmetic memory for one `LLM.generate`**, not a diary of the conversation and not extra
world knowledge.

### Why decode would be unbearable without it

Causal attention at token *t* needs Keys and Values for positions `0 … t`. Without a cache, every new word would force
every layer to rebuild K and V for the **entire** prefix — work that grows with the length of the reply.

With a cache:

```text
  prefill  (prompt tokens 0 … n−1, all at once)
      compute Q, K, V for the prompt
      write K, V into pages
      attend (causal) using those pages
      sample the first new token

  decode   (token n, then n+1, …  one at a time)
      compute Q, K, V for the newest token only
      write that token’s K, V into the next slot
      this Q  vs  all cached Keys  →  mix cached Values
      sample the next token
```

That is why the **first pause** can feel longer than each following word: prefill walks the whole prompt; each decode
step is “one new line in the notebook, then a glance over the pages you already have.”

The *rule* of attention (causal, GQA, optional sliding window) does not change between phases. Only the **shape of the
work** changes (chapter 8, kind 6).

### Pages, not one endless scroll

Instead of one giant contiguous notebook per conversation, this project (following the vLLM **PagedAttention** idea)
uses **fixed-size pages** (blocks). A running sequence holds a **block table**: a list of page numbers — like a library
call slip pointing to several short notebooks on a shared shelf.

```text
  token positions:   0 … 255     256 … 511     512 …
  block table:       page 7      page 3        page 12     ← Sequence.blockTable()
  physical slot:     7×256+off   3×256+off     12×256+off
```

**Why pages?**

- Many conversations can share one pool of pages (`BlockManager`); unused pages go back on the free list.
- If two prompts start with the **same token prefix** (same first full pages), the engine can **reuse** those already
  filled pages (`hashBlocks` / prefix hash) instead of writing them twice.
- Memory is reserved as a **fixed arena** at `LLM.builder().build()` (`KvCacheArena`), not grown as a Java `ArrayList`
  of tokens.

Default page length is **`kvcacheBlockSize = 256`**. When a sequence crosses a page boundary, `mayAppend` takes one more
free page.

### How large the shelf is

Pages are allocated **per `LLM`**, not inside `LlmModelFactory.make`. Two engines sharing one model do **not** share
notebooks.

How many pages:

- You may set `.numKvcacheBlocks(N)` explicitly.
- Otherwise the engine estimates from `maxModelLen` × `maxNumSeqs`, then **caps** by a fraction of the JVM max heap
  (`.kvHeapFraction(0.25f)` by default) using bytes per page ≈ `2 × blockSize × numKvHeads × headDim × 4` **per
  non-shared layer** (the `2` is K and V).

If the pool runs out (`no free KV blocks`), a sequence may wait or be preempted (chapter 13). That is a **memory**
limit, distinct from the **token** limit `maxModelLen`.

### Two phases of work (same generate)

1. **Prefill** — read the whole current prompt once, fill notebooks, produce the first new token.
2. **Decode** — for each later token, read mostly from notebooks, write one new slot, pick one new token.

One `ChatSession.send` (or `rag.send`) is typically **one** `generate`: one prefill of the templated prompt, then decode
until stop. Advisor roles, if configured, are a **separate** batched generate **before** that, with their own short-lived
pages.

### Context window, chat history, and not losing the thread

Three different “pasts” are easy to confuse. Only together do they explain how a chat stays coherent — and when it
**loses the thread**.

| Layer              | What it stores                                                                          | Lifetime                                   | Role                                           |
|--------------------|-----------------------------------------------------------------------------------------|--------------------------------------------|------------------------------------------------|
| **Context window** | Token ids that fit in one forward (capped by `maxModelLen` ≤ `max_position_embeddings`) | One `generate` call                        | Hard length budget for attention + RoPE tables |
| **Chat history**   | `List<ChatMessage>` in `ChatSession` (roles + text)                                     | Across turns until `clear()`               | Application memory of the dialogue             |
| **KV cache**       | Key/Value tensors for positions already processed                                       | Inside one `generate` (then freed/rebuilt) | Speeds decode; not a permanent diary           |

**Within one reply** the model “remembers” earlier tokens because attention reads the **KV cache** (and the growing
token list in the `Sequence`). It does **not** magically recall turns you never put back into the next prompt.

**Across replies** this library keeps track like this:

```text
  ChatSession history  (system + prior user/assistant text)
        │
        ▼
  ChatMessages#truncateHistory   // drop oldest turns if over budget
        │
        ▼
  Tokenizer#applyChatTemplate → encode → LLM#generate
        │                         │
        │                         └── prefill fills KV for THIS prompt only
        ▼
  finishTurn → history.add(assistant(answer only))
```

Budget for the templated prompt (before generating new tokens):

> `budget ≈ max(64, maxModelLen − maxTokens − 16)`  
> (`ChatMessages#truncateHistory`; margin leaves room for the reply.)

If the encoded prompt is still too long, the oldest non-system turns are removed (and a following assistant turn may be
removed with its user turn) until it fits — or only the minimum kept messages remain.

#### How the model keeps history — and how it can lose track

| Mechanism                     | Keeps the thread by…                                         | You can lose track when…                                                                          |
|-------------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Chat history + template       | Re-feeding prior turns as text each `send`                   | `truncateHistory` drops early facts you still need                                                |
| Answer-only history           | Storing the visible answer, not `<think>` notes              | Important details lived only inside thinking tags                                                 |
| KV cache                      | Attending to all prompt+reply tokens **inside** one generate | A **new** turn rebuilds the prompt; old KV from the previous generate is not the long-term store  |
| Sliding-window layers (Gemma) | Local layers see only the last *W* tokens                    | Distant tokens in the same prompt are invisible to those layers (global layers still see farther) |
| Short `maxModelLen`           | Fitting on your machine                                      | Long dialogues must forget the beginning                                                          |

There is **no** separate long-term memory module beyond weights + whatever you keep in `ChatSession` history (and
whatever fits in the context window of the next prefill).

**In the code:** arena `engine.KvCacheArena` (per-`LLM` K/V pages); paging `engine.BlockManager` (`allocate`,
`hashBlocks`, prefix reuse, `mayAppend`); slot lists on `Sequence.blockTable()`; prefill/decode scheduling in
`Scheduler`; write `Attention.storeKvCache`, read `attendRange` on paged slots; dialogue memory in `ChatSession` /
`ChatMessages#truncateHistory`. Size knobs: `LLM.Builder.kvHeapFraction` / `numKvcacheBlocks` / `kvcacheBlockSize`
(chapter 16).

**Further reading:** paged KV cache and high-throughput serving —
[Kwon et al., *Efficient Memory Management for Large Language Model Serving with
PagedAttention*](https://arxiv.org/abs/2309.06180)
(vLLM).


---

## 13. Serving several conversations without chaos

A naïve program would finish Alice’s entire answer before Bob gets a turn. This engine instead keeps a **waiting room**
and a **work floor**:

- new prompts wait until there is room;
- several conversations can advance a little on the same “tick”;
- if memory pages run short, one conversation may be politely paused and restarted later.

You can think of a restaurant kitchen preparing several dishes in interleaved steps, not cooking one meal completely
before lighting the next stove. That idea is called **continuous batching**. For a single chat on your laptop you may
barely notice it; it matters when many requests share one model.

**Further reading:** same vLLM paper above discusses iteration-level scheduling with paging; also see
the [vLLM project](https://github.com/vllm-project/vllm).

**In the code:** waiting room and work floor are `Scheduler` (`schedule`, `postprocess`, `clear` on cancel); each
in-flight reply is a `Sequence` (chapter 16).


---

## 14. Chat versus finishing a sentence

Two different manners of use:

| Manner         | What you are doing                      | Everyday analogy                          |
|----------------|-----------------------------------------|-------------------------------------------|
| **Completion** | Continue raw text as-is                 | Finish this paragraph…                    |
| **Chat**       | Messages with roles, history, templates | A scripted dialogue with stage directions |

Chat keeps a **history** of turns, wraps them in the model’s expected markers, and may trim old turns if the context
window is full (chapter 12 — `ChatMessages#truncateHistory`). That history is how multi-turn chat **stays on topic**:
each `send` rebuilds a prompt from remaining messages, then prefill+decode run again with a fresh KV cache for that
prompt.

**System prompts** (“answer briefly and factually”) are stage directions to the assistant. Some models (notably Gemma in
this project’s default) do **not** have a proper “system” role; stuffing a long lecture into the first user line can
make a small model freeze into polite filler (“Okay, I’m ready”) instead of answering. Short, concrete user text works
better there.

This program’s chat helper can also separate “thinking out loud” tags from the visible answer when the model uses them —
Sense C from the thinking chapter: marked scratchpad versus fair copy.

**Further reading:** [chat templating](https://huggingface.co/docs/transformers/chat_templating) in Transformers.

**In the code:** chat is `ChatSession` (`send`, `listen` / `streamTo`, history as `ChatMessage`);
`listen` can take `TEXT_RAW` (unparsed decode) plus parsed `TEXT_THINKING` / `TEXT_ASSISTANT`;
`streamTo` prints the parsed channels only. Completion is `LLM.complete` / raw `generate`; templates via
`Tokenizer.applyChatTemplate`; defaults in `ChatPrompts.systemFor` (chapter 16).

```java
try (LlmModel model = LlmModelFactory.make(modelDir);
     LLM llm = LLM.builder(model).build()) {
  // Chat: history + chat template + optional <think> parse
  String chat = llm.chat(256).send("What is 2+2?").answer();
  String once = llm.chatOnce("What is 2+2?");           // one turn, no kept session

  // Completion: raw continuation (no chat template)
  String raw = llm.complete("The capital of France is");
}
```

---

## 15. A full walk-through: “What is 2+2?”

This chapter retells one question as a story **and** as a call chain through this library. For each stage you get:

1. what happens in plain language;
2. **where** it lives as `Class#method` (and how those methods call each other).

Notation: `ChatSession#send` means method `send` on class `ChatSession`. Package root is
`com.igormaznitsa.nanollvm`. See chapter 16 for samples and the full file tree.

**Further reading (optional depth):** Transformer + attention ([Vaswani et al.](https://arxiv.org/abs/1706.03762)),
chain-of-thought prompting ([Wei et al.](https://arxiv.org/abs/2201.11903)), paged KV ideas
([Kwon et al.](https://arxiv.org/abs/2309.06180)).

### Master call chain (keep this picture)

When you write something like `llm.chat(256).send("What is 2+2?")`, the library walks this path once for the **open**,
then this path for the **turn**:

```text
OPEN (once)
  LlmModelFactory#make(Path | ModelFileSource | fromClasspath*)
      → (HF safetensors) SafetensorsTransport → ArchitectureProcessor bind/fill/create
         + Tokenizer#fromPretrained   // tokenizer.json, else tokenizer.model (1.2.0), else config.json
      → (HF ONNX, since 1.1.0) OnnxTransport → ArchitectureProcessor bind/fill/create → same CausalLM / EmbeddingEncoder graphs
      → (GGUF causal) GgufTransport → ArchitectureProcessor bind/fill
         → Qwen3ForCausalLM or Lfm2ForCausalLM + Tokenizer#fromGguf
      → (GGUF bert) ArchitectureProcessor.createEmbedding → BertForEmbedding
      → (Whisper safetensors, since 1.3.0) SpeechArchitecture → WhisperForAsr
      → (Piper ONNX + json, since 1.3.0) SynthesisArchitecture → PiperForTts
  LLM#builder(LlmModel) → LLM.Builder#build → LLM.<init>         // every kind
      → (chat) Transformer.<init>          (binds shared model; allocates KvCacheArena)
      → (embed / speech / synthesis) skip KV; same MatmulRuntime
      → Scheduler.<init>                   (owns BlockManager; unused on non-chat)
      → optional LLM.Builder#warmup()      (off by default; chat)

ONE TURN
  LLM#chat(maxTokens) → ChatSession#open → ChatSession#send("What is 2+2?")
      → ChatMessage#user / history.add
      → ChatMessages#truncateHistory
      → LLM#runAdvisors (optional; when Builder#advisors configured)
          → one batched LLM#generate for all LlmAdvisor roles
          → returns AdvisorEnrichment (contains List<AdvisorResponse>)
          → LlmAdvisorMixer#mixPrompt(…, ChatHistory, prompt) → enriched model user text
      → ChatSession#generateTurn
          → Tokenizer#applyChatTemplate
          → LLM#generate
              → encode → Scheduler#add (new Sequence)
              → loop while !Scheduler#isFinished:
                    private stepUnlocked
                      → Scheduler#schedule          (BlockManager#allocate / #mayAppend)
                      → Transformer#step
                          → preparePrefill | prepareDecode
                          → CausalLM#forward        (Qwen3 / Gemma3 / Llama / Lfm2)
                          → CausalLM#computeLogits
                          → Sampler#forward
                      → Scheduler#postprocess       (append token / finish on stop)
                    onToken → decode(ids, skipSpecials=false) as TEXT_RAW
                              + AssistantParts#parse → TEXT_THINKING / TEXT_ASSISTANT
          → AssistantParts#parse(Tokenizer#decode(final ids))
      → ChatSession#finishTurn
          → maybe AssistantParts#salvageFromThinking
          → ChatMessage#assistant(answer only) into history
```

That is the whole “2+2” turn in library terms. The subsections below zoom each box.

### You ask

A few lines of Java (or the example app) open the model folder and say, in effect: *chat with me; keep answers short.*

```java
try (LlmModel model = LlmModelFactory.make(modelDir);
     LLM llm = LLM.builder(model).maxModelLen(2048).build()) {
  ChatReply reply = llm.chat(256).send("What is 2+2?");
  String visible = reply.answer();   // user-facing text (same as reply.text())
  String notes   = reply.thinking(); // optional scratchpad; not stored in history
  boolean open   = reply.thinkOpen(); // false after send; true only on streaming snapshots
  double tokPerSec = reply.stats().completionTokensPerSecond();
} // close LLM first (try-with-resources order), then LlmModel
```

| Step            | Call                                   | Role                                               |
|-----------------|----------------------------------------|----------------------------------------------------|
| Load once       | `LlmModelFactory#make`                    | Immutable shared `LlmModel` (weights + tokenizer)     |
| Start builder   | `LLM#builder(LlmModel)` only           | Fluent open; **no** path overload — load with the factory |
| Finish open     | `LLM.Builder#build` → `LLM.<init>`     | Wire engine + per-LLM KV arena                     |
| Start session   | `LLM#chat(int)` → `ChatSession#open`   | History + `LLM#defaultSampling(maxTokens)`          |
| Ask             | `ChatSession#send`                     | One user turn through template → generate → parse  |
| CLI alternative | `samples.Example#main`                 | Same ideas with `streamTo` on stderr/stdout        |

### Loading (once)

- Blueprint read → empty rooms built to the right sizes (`hidden_size`, layers, heads, …).
- Each weight name in the crates poured onto the matching shelf (Query/Key/Value packs merged where needed).
- Dictionary opened at the door.
- Blank Key/Value notebooks laid out for this session’s text.
- Learned shelves will not change; notebooks will.

**Attention’s role at load time:** none yet — only empty notebooks waiting.  
**Thinking’s role at load time:** none — no Sense A until a prompt runs.

| Step            | Call                                                               | Role                                                      |
|-----------------|--------------------------------------------------------------------|-----------------------------------------------------------|
| Blueprint       | `Config.HfConfig#load` (via `LlmModelFactory`)                        | Read `config.json` (HF) or GGUF metadata              |
| Empty graph     | `ArchitectureProcessor#createCausal` / `#createEmbedding` / `#createSpeech` / `#createSynthesis`            | `Qwen3ForCausalLM`, `Gemma3ForCausalLM`, `Gemma4ForCausalLM`, `LlamaForCausalLM`, `Lfm2ForCausalLM`, `BertForEmbedding`, `WhisperForAsr`, or `PiperForTts` (all under `models.internal`) |
| Pour weights    | `ContainerTransport` + `ArchitectureProcessor#fill` → `WeightBag` | Merge shards / dequant path; construct immutable graph |
| Seal            | (graph is immutable at construction)                               | No post-load weight mutation                              |
| Dictionary      | `Tokenizer#fromPretrained` / `#fromGguf` / `#fromSentencePiece`     | HF JSON, SentencePiece `tokenizer.model` (**since 1.2.0**), or GGUF-embedded vocab + chat template / stop ids |
| Blank notebooks | `Transformer` → `KvCacheArena`                                     | Per-`LLM` KV pages; bound into `Context` for `Attention`  |
| Waiting room    | `Scheduler.<init>` → `BlockManager.<init>`                         | Page pool for later allocate                              |
| Optional        | `LLM.Builder#warmup()`                                             | Tiny generate after open so first real answer is not also cold-start (**off by default**) |

### Your sentence becomes numbers

The chat template wraps your question with “user” / “assistant” markers (and, on Qwen-style chat, system directions that
may *invite* a `<think>` scratchpad). The tokenizer turns that into a list of token numbers.

Illustrative shape (markers depend on the model):

```text
[system … maybe “use <think> for notes” …]
[user] What is 2+2?
[assistant]          ← generation starts here
```

Those ids are just a line of dictionary numbers. No attention has run yet.

| Step             | Call                                                                    | Role                                             |
|------------------|-------------------------------------------------------------------------|--------------------------------------------------|
| Record user      | `ChatSession#send` → `ChatMessage#user`                                 | Append to history                                |
| Fit desk         | `ChatMessages#truncateHistory`                                          | Drop old turns if context is tight               |
| Stage directions | `Tokenizer#applyChatTemplate(…, enableThinking)`                        | Role markers via `ChatFormat`; think seed when ChatML + `invitesThinking` |
| To ids           | inside `LLM#generate` → encode → `Scheduler#add` (new `Sequence`) | Prompt string → token ids → `Sequence`           |

`ChatSession#generateTurn` is the private brick that calls `applyChatTemplate` then `LLM#generate`.

### Prefill — Sense A on the whole prompt (attention’s first big job)

Portraits for every prompt token walk through **every** reading room.

**Where attention works in prefill**

At each layer, for each position in the prompt (for example the token `2`, the token `+`, the later `2`, the `?`):

1. Build Query / Key / Value from that position’s hidden state.
2. Compare this Query with Keys of **earlier** positions only (causal — no peeking at the future).
3. Mix Values from the strong matches into an updated hidden vector.
4. **Write** this position’s Key and Value into the KV cache for later decode (Query is not stored; chapter 12).

So when the model is still “reading” `What is 2+2?`, attention is already linking pieces: the second `2` can glance at
the first `2` and at `+`; the end of the user line can glance at the whole question. Multi-head glances (and GQA
sharing) run in parallel; Gemma may restrict some layers to a sliding window of recent tokens.

**Where thinking works in prefill**

This is pure **Sense A** (silent inner work): stacked attention + MLP through all layers. You see no English “reasoning”
yet. Prefill ends by scoring the vocabulary at the last prompt position and **sampling the first assistant token** —
which might be `<`, or `4`, or `The`, depending on style and chance.

```text
  prompt tokens ──► layer 1 (attend + rewrite) ──► … ──► layer N
       │                      │
       │                      └── fill KV notebooks
       ▼
  first next-token scores → sample token₁
```

**How it is called (prefill tick)**

```text
LLM#generate
  └─ private stepUnlocked                 // first tick is usually prefill
       ├─ Scheduler#schedule          // BlockManager#canAllocate / #allocate; prefill=true
       ├─ Transformer#step(seqs, true)
       │    ├─ Transformer#preparePrefill   // ids, positions, Context slot maps / block tables
       │    ├─ CausalLM#forward             // e.g. Qwen3ForCausalLM#forward
       │    │    └─ per layer: … → Qwen3Attention#forward → Attention#forward
       │    │         ├─ Attention#storeKvCache
       │    │         └─ Attention#prefillWithCache | #prefillDense → #attendRange
       │    │    └─ per layer MLP: Qwen3MLP#forward → Ops#siluAndMul   (Gemma: gelu…)
       │    ├─ CausalLM#computeLogits
       │    └─ Sampler#forward
       └─ Scheduler#postprocess       // append first new token to Sequence
```

| Brick        | Call                                           | Role for “2+2?”                         |
|--------------|------------------------------------------------|-----------------------------------------|
| Pick work    | `Scheduler#schedule`                           | Prefill batch for this `Sequence`       |
| Pages        | `BlockManager#allocate`                        | Give the prompt notebook pages          |
| Pack tensors | `Transformer#preparePrefill`                   | Build input ids / positions / `Context` |
| Walk rooms   | `Qwen3ForCausalLM#forward` (or Gemma)          | Embedding → N layers → norm             |
| One glance   | `Qwen3Attention#forward` → `Attention#forward` | QKV, RoPE, store KV, causal attend      |
| Rewrite      | `Qwen3MLP#forward`                             | Gated feed-forward after attention      |
| Score & draw | `CausalLM#computeLogits` → `Sampler#forward`   | First assistant token                   |

Inside one attention brick (Qwen-shaped; Gemma is analogous in `Gemma3Attention#forward`):

```text
Qwen3Attention#forward
  → Linear.Qkv#forward          // build packed Q/K/V
  → Ops#splitLast
  → optional RMSNorm#forward on Q/K heads
  → RotaryEmbedding#forward     // rotate Q,K by position angle (RoPE)
  → Attention#forward
       → Attention#storeKvCache
       → Attention#prefill… → #attendRange
  → Linear.Row#forward          // o_proj
```

### Decode — one new scrap at a time (attention + optional written thinking)

Each further step:

1. take only the **newest** token through the rooms;
2. **attention** looks back over past Keys/Values in the notebooks (and writes a new notebook line);
3. MLP rewrites;
4. score the vocabulary;
5. draw one more token;
6. stop on an end marker or a length limit.

**Attention’s role in decode (for this sum)**

The new token’s Query asks: “given everything so far, what should influence me?” Keys/Values already stored for
`What`, `is`, `2`, `+`, `2`, `?`, and any assistant tokens already emitted, are the library it searches. It does **not**
re-read the prompt from scratch as raw text; it reuses notebook notes. That is why decode feels lighter than prefill.

Concrete (illustrative) glances:

| While producing… | Attention may lean on…                                    | Why it matters                             |
|------------------|-----------------------------------------------------------|--------------------------------------------|
| Early note words | User tokens `2`, `+`, `2`                                 | Bind the question’s parts                  |
| Later note words | Earlier note words already emitted                        | Written scratchpad becomes rereadable past |
| The visible `4`  | Question tokens and/or note tokens that mentioned the sum | Final answer conditioned on that past      |

Attention never “knows arithmetic” as a separate calculator. It **routes** trained patterns toward the current place
using similarity of Keys and Queries. If training left useful habits for digit sums, those habits fire when the right
places attend to each other.

**How it is called (each decode tick)** — same outer loop as prefill; only the middle changes:

```text
private stepUnlocked
  ├─ Scheduler#schedule              // prefill=false; BlockManager#mayAppend
  ├─ Transformer#step(seqs, false)
  │    ├─ Transformer#prepareDecode  // mostly the newest token + block tables
  │    ├─ CausalLM#forward
  │    │    └─ Attention#forward → #storeKvCache → #decode → #attendRange
  │    ├─ CausalLM#computeLogits
  │    └─ Sampler#forward
  └─ Scheduler#postprocess           // stop if eos / stop ids / maxTokens / maxModelLen / degenerate loop
```

`LLM#generate` keeps calling private `stepUnlocked` until `Scheduler#isFinished()` is true. There is **no** public
`LLM#step` or `LLM#isFinished`. Each appended token may fire the `onToken` callback that `ChatSession#generateTurn`
registered.

**Thinking’s three roles in this same decode loop**

| Sense                     | What happens on “What is 2+2?”                                                                                                   | Library home                                                                 |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **A — silent**            | Every decode step is a full layer-walk (attention + MLP + sample). Always on, never shown as text.                               | `Transformer#step` → model `#forward` / `#computeLogits` → `Sampler#forward` |
| **B — written reasoning** | The model may emit ordinary words like “add” or “four” as intermediate text. Those words join the past.                          | Same loop; tokens land in `Sequence` via `Scheduler#postprocess`             |
| **C — tagged scratchpad** | On Qwen-style chat it may emit `<think> … </think>` then the answer. Same Sense A underneath; markers let the UI split channels. | `AssistantParts#parse` on decode/`finishTurn`                                |

Example timeline (one possible Qwen-style path — not guaranteed wording):

```text
  decode → "<think>"           Sense A; attention sees only the prompt so far
  decode → "User" "asks" …     Sense A; attention can also see prior think tokens
  decode → "2+2" "→" "4"       notes cite the question; attention binds them
  decode → "</think>"
  decode → "4"                 Sense A; attention may weigh notes + original "2" "+" "2"
  decode → end-of-turn
```

If there is **no** scratchpad (turn-based `ChatFormat`, or ChatML without think tags), Sense A still runs the same way;
you only see the final answer tokens. There is no missing “thinker” — only missing **visible** Sense B/C text.
`ChatSession` defaults thinking from `Tokenizer#invitesThinking(open, close)` for the session’s `ThinkTags` (default `<think>` / `</think>`) unless the caller sets `enableThinking(…)`.

**How written thinking helps attention (and how it can fail)**

Once note tokens exist, later attention can look at them (causal self-attention over the growing reply). That is the
whole mechanism: **thinking-as-text becomes more past for attention to mix.**

It can help a multi-step question by parking intermediate results on the page. It can also produce fluent wrong notes;
attention will happily attend to those too. Tags do not create a second brain — `AssistantParts#parse` only **splits**
the finished stream for display (`thinking` vs `answer`).

### After the turn — parse and show

- Tokenizer decodes ids → raw assistant string.
- Chat helper splits `<think>…</think>` from the visible answer when markers exist.
- You may see notes on a “thinking” channel and `4` on the answer channel.
- `ChatSession` appends only the **visible answer** to history for the next turn (not the scratchpad). The next prefill
  attends across that history subject to length limits.

| Step               | Call                                                                             | Role                                 |
|--------------------|----------------------------------------------------------------------------------|--------------------------------------|
| Live UI (optional) | `onToken` → `Tokenizer#decode` → `AssistantParts#parse` → `StreamPrinter#update` | Split channels while tokens arrive   |
| Final split        | `AssistantParts#parse` → `ChatReply#from`                                        | `thinking` + `answer` + `thinkOpen`; session then `withStats` |
| Recover            | `AssistantParts#salvageFromThinking`                                             | If answer blank but notes exist      |
| Commit history     | `ChatSession#finishTurn` → `ChatMessage#assistant(answer)`                       | **Answer only** stored for next turn |
| Close stream       | `StreamPrinter#closeTurn`                                                        | End of CLI printing for this reply   |

### One picture of the whole turn (story + calls)

```text
  LlmModelFactory#make → LlmModel              LOAD immutable weights + tokenizer
  LLM.<init> / Transformer.<init>        BIND LlmModel + empty KvCacheArena
           │
           ▼
  Tokenizer#applyChatTemplate
  + Tokenizer#encode                  TEMPLATE + TOKENIZE  "What is 2+2?"
           │
           ▼
  LLM#generate → stepUnlocked (prefill)   PREFILL (Sense A)
  Scheduler#schedule
  Transformer#step → CausalLM#forward
  Attention storeKv / prefill…           attention: every prompt place looks back; notebooks fill
  Sampler#forward                        thinking: silent only
           │
           ▼
  stepUnlocked (decode) × N           DECODE LOOP (Sense A each step)
  Attention#decode / #attendRange        attention: new token queries notebooks
  Sampler#forward                        thinking: optional <think> tokens (B/C)
  Scheduler#postprocess
           │
           ▼
  stop ids / maxTokens / maxModelLen / degenerate loop    SAMPLE "4" … STOP
           │
           ▼
  AssistantParts#parse
  ChatSession#finishTurn              PARSE / DISPLAY  thinking + answer "4"
```

### What this walk-through should leave you with

You should be able to point at **both** the story and the library:

| Story piece            | Primary `Class#method` homes                                                                |
|------------------------|---------------------------------------------------------------------------------------------|
| Open model             | `LlmModelFactory#make` → `LlmModel`; `LLM#builder(LlmModel)` → `Transformer.<init>` (`KvCacheArena`) |
| Chat ask               | `ChatSession#send` → `#generateTurn` → `#finishTurn`                                        |
| Template / ids         | `Tokenizer#applyChatTemplate` / `#encode` / `#decode`                                       |
| Engine loop            | `LLM#generate` → private `stepUnlocked` → `Scheduler#schedule` / `#postprocess`             |
| One forward+sample     | `Transformer#step` → `CausalLM#forward` / `#computeLogits` → `Sampler#forward`              |
| Attention + notebooks  | `Context#bindKvCache` → `Attention#forward` (by `layerIndex`) / `#attendRange`              |
| Visible thinking split | `AssistantParts#parse` (history keeps answer via `finishTurn`)                              |

Chapter 16 repeats these as tables and copy-shaped samples.

- **Attention** is the glance that mixes allowed past into the present — heavy at prefill, notebook-backed at decode.
- **Thinking** is either that silent stack (always) or extra generated text (sometimes) that attention can later reuse.
- **“2+2→4”** is not a separate arithmetic module; it is trained continuation steered by those glances and draws.

Nothing mystical happened — unpacking a crate, then a long chain of rereads, rewrites, and draws among likely scraps of
text. The *impression* of understanding is an effect of that chain, shaped by training you do not see.

---

## 16. Where it lives in the code (classes, methods, samples)

The earlier chapters tell the story. This one is the **map to the source**: which Java types and methods implement each
idea, plus short samples you can recognize in the tree under
`nano-vllm-java/src/main/java/com/igormaznitsa/nanollvm/` (demos live in `nano-vllm-java-samples`).

Every earlier chapter’s **In the code** note points here. Use this chapter when you want methods, samples, and paths in
one place.

You do not need to read every file. Use the tables to jump, then skim the named methods.

### Package layout (folders ↔ ideas)

**JPMS:** module `com.igormaznitsa.nanollvm` **exports** `models`, `llm`, `chat`, `rag`, `exceptions`, `tokenizer`,
`utils`. Everything else below (`models.internal`, `models.llmcontainer`, `models.llmarch`, `engine`, `layers`,
`tensor`, `internal`, `prompts`) is
**not exported** — fine to read in this guide; application code under the module system should stay on the exported
packages. Demos live in the separate Maven module `nano-vllm-java-samples`.

| Folder / type                                                                          | Role in the story                                    | Exported? |
|----------------------------------------------------------------------------------------|------------------------------------------------------|-----------|
| `llm/` — `LLM`, `LLM.Builder`, `Config`, `SamplingParams`, `GenerationStats`, `LlmAdvisor`, `LlmAdvisorMixer`, `AdvisorResponse`, `AdvisorEnrichment` | Front door; named advisors + mixer; stats; `embed` / `transcribe` / `synthesize` | yes |
| `models/` — `LlmModel`, `LlmModelFactory`, `ModelSupport`, `LlmOptionalData`, `LlmModality` / `LlmModalities`, `ModelFileId`, `ModelFileSource`, `ModelFileSources` | Shared immutable loaded model + architecture catalog + load extras + stream/classpath sources | yes |
| `models.internal/` — `WeightBag`, `CausalLM*`, `BertForEmbedding`, `EmbeddingEncoder`, `WhisperForAsr`, `PiperForTts`, … | Graphs, weight bags, BERT encode, Whisper ASR, Piper TTS | **no** |
| `models.llmcontainer/` — `ContainerTransport`, `GgufTransport`, `SafetensorsTransport`, `OnnxTransport`, `LoadProgress` | Weight-file I/O and catalog | **no** |
| `models.llmarch/` — `ArchitectureProcessor`, family processors, `ModelBinding` / `ModelFill` | Bind / fill / create per architecture (causal, embedding, speech, synthesis) | **no** |
| `chat/` — `ChatSession`, `ChatHistory`, `ChatMessage`, `ChatMessages`, `ThinkTags`, `ChatSpecials`, `LlmListener`, `LlmTextKind`, `ChatReply`, `StreamPrinter` | Dialog + unified text/status events | yes |
| `tokenizer/Tokenizer`, `GgufTokenizerSource`                                           | HF / GGUF / SentencePiece vocab → encode / decode / chat template (**`fromSentencePiece` since 1.2.0**) | yes |
| `utils/NanoLlvmProps`, `ResourceLimits`                                                | Property/env knobs; process-wide parser/corpus caps (**since 1.0.0**) | yes |
| `exceptions/`                                                                          | Typed library failures                               | yes |
| `rag/` — `RagFactory`, `PreparedRag`, `DenseRagIndex`, `HybridRagIndex`, `RagSession`, `RagIndex`, `RagHit`, `RagLoadOptions`, `RagTuner`, `RagResource`, … | Text RAG: BM25 and (since **1.1.0**) dense/hybrid + classpath docs; load-time tuners since **1.2.0** | yes |
| `internal/` — `Json`, `Context`, `GgufDequant`, `ModelFileBundle` | JSON helper, per-step context, GGUF dequant, in-memory file bundle | **no** |
| `engine/` — `Scheduler`, `Sequence`, `BlockManager`, `Transformer`, `KvCacheArena`     | Prefill/decode loop, pages, one forward+sample       | **no** |
| `layers/` — `Attention`, `BidirectionalAttention`, `Sampler`, `Linear`, `Norms`, …     | Attention, sampling, projections, norms/RoPE         | **no** |
| `tensor/` — `Tensor`, `Ops`, `MatmulRuntime`, `LinearKernel`, `EmbeddingKernel`, …     | Arrays, float ops, parallel GEMM                     | **no** |
| `prompts/` — `ChatPrompts`, `RagPrompts`, `AdvisorPrompts`                             | Default system / RAG / advisor wording               | **no** |
| *(Maven module `nano-vllm-java-samples`)* `Example`, `HelloWorld`, `NextTokenHelloWorld`, `Bench`, `EmbeddingsHelloWorld`, `TranscribeHelloWorld`, `SynthesizeHelloWorld`, `LogTriageHelloWorld`, `AdvisorRagHelloWorld`, `RagTunerHelloWorld`, `utils/Bundled*` | Runnable demos (not in the library JAR) | n/a |

`Config.HfConfig` (in `llm/Config`) holds the blueprint plus per-LLM engine knobs (`maxModelLen`, `kvHeapFraction`, …).

### Concept → class → methods

| Story idea                       | Primary type                                                         | Methods / entry points to open                                                                                |
|----------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Open a model                     | `LlmModelFactory`, `LlmModel`, `LLM.Builder`                               | `open(Path).make()` / `make(Path)` / `make(…, Map)` / `fromClasspath*` / `openClasspath*`; HF **safetensors or ONNX** (**since 1.1.0**); Gemma 4 **text QAT** packed safetensors; GGUF **`qwen3`** / **`lfm2`** chat + **`bert`** embed; Whisper safetensors / Piper ONNX **since 1.3.0**; `thinkTags(ThinkTags)` / `OPTION_THINK_TAGS`; `chatSpecials(ChatSpecials)` / `OPTION_CHAT_SPECIALS`; `optionalData` / `OPTION_OPTIONAL_DATA`; `LLM.builder(model)` for every kind; `toString()` summarizes load |
| Custom scratchpad / answer specials | `ThinkTags`, `ChatSpecials`, `LlmModel`                                 | `open(path).thinkTags(tags).chatSpecials(specials).make()`; `ChatSession.thinkTags` / `RagSession.thinkTags` (**since 1.1.0**); omitted options get library defaults |
| Sentence embedding (BERT GGUF / ONNX) | `LLM` / `LlmModel` (internal `BertForEmbedding`)                                   | `make(gteGguf)` → `isEmbeddingModel()` → `LLM.builder` then `embed(text)` (**since 1.3.0**); `LlmModel.embed` sequential shortcut (chapter **7b**, **since 1.1.0**)                        |
| Speech to text (Whisper)         | `LLM` / `LlmModel` (internal `WhisperForAsr`)                                      | `make(whisperDir)` → `isSpeechModel()` → `transcribe(wavBytes \| Path \| pcm)` (**since 1.3.0**, chapter **7d**) |
| Text to speech (Piper)           | `LLM` / `LlmModel` (internal `PiperForTts`)                                        | `open(voice).optionalData(ESPEAK_DATA, dir).make()` → `synthesize(text)` → WAV bytes (**since 1.3.0**, chapter **7e**) |
| Named advisors                   | `LlmAdvisor`, `LlmAdvisorMixer`, `AdvisorEnrichment`, `ChatHistory`        | `Builder.advisors(mixer, …)` — unique non-blank names; `LLM#runAdvisors` → `AdvisorEnrichment`; one batched `generate` |
| Chat turn                        | `ChatSession`                                                        | `llm.chat(maxTokens)`, `.listen(…)`, `.streamTo(…)` (`TEXT_THINKING` / `TEXT_ASSISTANT`; ignores `TEXT_RAW`), `.send(user)`, `.clear()`; `emitDebugPrompts(true)` opts in `TEXT_DEBUG` (off by default); `recoverUnusableAnswers` / `unusableAnswer` opt-in |
| Stream unparsed decode           | `LlmTextKind.TEXT_RAW`, `LlmListener`                                  | `ChatSession.listen`; deltas of tokenizer decode with think tags / chat specials kept (**since 1.1.0**) |
| One-shot / raw text              | `LLM`                                                                | `chatOnce(…)`, `complete(…)`, `generate(…)`, `generateTokenIds(…)` → `GenerationOutput` (`tokenIds`, `text`, `stats`) |
| Token / timing stats             | `GenerationStats` / `ChatReply`                                      | `promptTokens`, `completionTokens`, `elapsedNanos`, `completionTokensPerSecond()`                             |
| Cancel / timeout                 | `LLM`                                                                | `cancel()`; `generate(…, timeout, onToken)`                                                                   |
| Tokenize                         | `Tokenizer` (on `LlmModel`)                                             | `LlmModel.tokenizer()`; `encode`, `decode`, `applyChatTemplate(…, enableThinking)`; load: `fromPretrained` (`tokenizer.json` else `tokenizer.model`) / `fromGguf` / `fromSentencePiece` (**since 1.2.0**) |
| Token embedding / RoPE / LM head | (internal) `VocabParallelEmbedding`, kernels, `RotaryEmbedding`, … | Via `CausalLM#forward` / LM head — not public app API                                                         |
| Blueprint                        | `Config.HfConfig`                                                    | `HfConfig.load(config.json)`; internal `ArchitectureProcessor` bind/create                                        |
| Pour weights                     | `LlmModelFactory` + internal loaders / `WeightBag`                      | Sealed inside `make`; apps do not call `ModelLoader` directly                                                 |
| One engine tick                  | `LLM` (private `stepUnlocked`), `Scheduler`, `Transformer`           | Inside `generate`: `schedule` → `Transformer.step` → `postprocess`                                            |
| Forward + sample                 | `Transformer`, internal `CausalLM`, `Sampler`                        | `network.forward` → `computeLogits` → `sampler.forward`                                                       |
| Attention + KV write             | `Attention`, `KvCacheArena`, `internal.Context`                      | `Context.bindKvCache`; `Attention.forward` by `layerIndex`; prefill/decode helpers                            |
| Pages / prefix reuse             | `BlockManager`                                                       | `canAllocate`, `allocate`, `hashBlocks`, `mayAppend`                                                          |
| Split thinking UI                | `ChatReply`                                                          | `ChatReply.parse` / `parse(raw, llm)` / `salvageFromThinking`                                                 |
| Text RAG (prepare + retrieve)    | `RagFactory`, `PreparedRag`, `DenseRagIndex`, `HybridRagIndex`, `RagSession`, `RagTuner` | `RagFactory.make` / `withEmbeddings` → `llm.rag(index).send(…)`; `Builder.addProcessor` tuners **since 1.2.0** (chapter 17); session knobs match `ChatSession` |
| Resource caps                    | `ResourceLimits`                                                     | Process-wide defaults + builder for file/JSON/GGUF/corpus/history budgets (**since 1.0.0**)               |
| Math bricks                      | `Ops`, `LinearKernel`, `EmbeddingKernel`, `MatmulRuntime`            | norms / MLP gates / softmax; linear & embed via kernels (internal); `.dedicatedMatmulPool()` **since 1.2.0** |

### Sample A — library use (what most apps call)

```java
import com.igormaznitsa.nanollvm.chat.LlmTextKind;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.LlmAdvisor;
import com.igormaznitsa.nanollvm.llm.LlmAdvisorMixer;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.rag.RagFactory;

import java.nio.file.Path;

try (LlmModel model = LlmModelFactory.open(Path.of("models/Qwen3-0.6B")).make();
     LLM llm = LLM.builder(model)
         .maxModelLen(2048)
         .kvHeapFraction(0.25f)                 // default; sizes KV arena vs heap
         .sampling(SamplingParams.builder().maxTokens(256).build())
         .systemPrompt("Answer briefly and factually.")  // optional; demos often set policy themselves
         .advisors(LlmAdvisorMixer.defaults(),
             LlmAdvisor.builder().name("Facts").prompt("List only hard facts useful for the user.").build())
         // optional: .advisorNoteFilter(note -> !note.contains("setup boilerplate"))
         // optional: .warmup()                 // off by default
         // optional: .allowUnpackParameters()  // late float32 for packed GGUF
         .build()) {

  // Multi-turn with streaming (history + template + optional <think> parse)
  var session = llm.chat().streamTo(System.err, System.out, false);
  // streamTo prints TEXT_THINKING / TEXT_ASSISTANT only; TEXT_RAW needs .listen(…)
  // TEXT_DEBUG (prepared model-user after advisors) is off unless session.emitDebugPrompts(true)
  String reply = session.send("What is 2+2?").answer();

  // One-shot chat (no kept session)
  String once = llm.chatOnce("What is 2+2?", 64);

  // Raw continuation (no chat template)
  String raw = llm.complete("The capital of France is");

  // Text RAG — prepare documents once (like LlmModel), share freely
  var rag = RagFactory.builder()
      .addFolders(Path.of("docs/kb"), Path.of("docs/policies"))
      .addResource(MyApp.class, "/help/faq.md")
      .add("support-hours", "Live chat is available 9–17 UTC.")
      .build();
  // one folder: RagFactory.make(Path.of("docs"))
  // since 1.1.0: one classpath file — RagFactory.makeResource("docs/a.md")
  //             hybrid — RagFactory.withEmbeddings(rag, embedModel)  (chapter 17)
  // since 1.2.0: builder().addProcessor(RagTuner…) filter / extract / preprocess (chapter 17)
  //             builder().addFolders(dir, dir) for several disk trees
  String grounded = llm.rag(rag).topK(2).send("What is the capital of France?").answer();
} // close LLM before LlmModel (try-with-resources closes in reverse declaration order)

// Unparsed streaming decode (think tags + chat specials kept):
// llm.chat(256).listen((src, ev) -> { if (ev.kind() == LlmTextKind.TEXT_RAW) System.err.print(ev.text()); })

// Embedding GGUF (since 1.1.0): LLM.builder then llm.embed; model.embed is a sequential shortcut (chapter 7b)
// Whisper (since 1.3.0): llm.transcribe(wavBytes) — chapter 7d
// Piper (since 1.3.0): llm.synthesize(text) → WAV bytes — chapter 7e
// Classpath / streams: see Sample A1 below
```

Interactive CLI wiring lives in `nano-vllm-java-samples` → `samples.Example.main`: load status and chat share
`samples.utils.OrderedConsole` (millis-stamped queue → stdout vs stderr), then **model menu**
(downloaded first, Qwen3-0.6B preferred for chat quality; Enter = first; empty disk → download instructions),
**RAG-mode menu** (none / BM25 / dense / hybrid — dense needs a BERT-encoder checkpoint under `models/`; **since 1.1.0**; Enter =
none), **advisor-count menu** (`0`–`3`; Enter = none), then `llm.chat(…).streamTo(…)` or
`llm.rag(index).streamTo(…)`. Prepared-prompt
`TEXT_DEBUG` lines are off by default (`ChatSession.emitDebugPrompts(true)` to enable; `Example`
opts in with `--debug`, chapter 17). Embedding
checkpoints skip RAG/advisors, then an encoder-session menu: embed REPL, or few-shot classify
(`label | text` / `/demo` / `/load`, centered prototypes on `LlmModel.embed` — not a Hub
classification head). Speech checkpoints skip RAG and open a WAV transcribe session. Synthesis
checkpoints skip RAG and open a text→WAV session. Each mode is a named method in `Example`.

### Sample A1 — classpath / stream sources (**since 1.1.0**)

```java
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.ModelFileSources;

// HF folder on the classpath (config + tokenizer + safetensors or ONNX)
try (var model = LlmModelFactory.fromClasspath(MyApp.class.getClassLoader(), "models/Qwen3-0.6B");
     var llm = LLM.builder(model).build()) {
  llm.chatOnce("Hello");
}

// Same idea via ModelFileSource (custom streams also implement ModelFileSource)
var source = ModelFileSources.classpath(MyApp.class.getClassLoader(), "models/Qwen3-0.6B");
try (var model = LlmModelFactory.make(source)) { /* … */ }

// GGUF resource: LlmModelFactory.fromClasspathGguf(loader, "models/gte-small.Q2_K.gguf");
// Note: ONNX external_data sidecars need make(Path), not classpath streams.
```

### Sample A2 — cancel and timeout

```java
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

try (LlmModel model = LlmModelFactory.make(Path.of("models/Qwen3-0.6B"));
     LLM llm = LLM.builder(model).build()) {

  // Wall-clock limit on a raw generate batch (null / ZERO = unbounded)
  llm.generate(
      List.of("Write a short poem about rivers."),
      llm.defaultSampling(128),
      Duration.ofSeconds(30));

  // Abort an in-flight generate from another thread
  Thread.ofVirtual().start(llm::cancel);
}
```

### Sample A3 — CPU threads and ResourceLimits

```java
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.utils.ResourceLimits;

import java.nio.file.Path;

// Process-wide parser / corpus / history caps (optional; defaults are already set)
ResourceLimits.setCurrent(
    ResourceLimits.builder()
        .maxCorpusFiles(2_000)
        .maxHistoryMessages(100)
        .build());

try (LlmModel model = LlmModelFactory.make(Path.of("models/Qwen3-0.6B"));
     LLM llm = LLM.builder(model)
         .cpuThreads(4)          // or .allCpuThreads() / .disableMultiCpu()
         .dedicatedMatmulPool()  // since 1.2.0: engine-owned pool; close() shuts it down
         .build()) {
  llm.chatOnce("Hello");
}
// Builder cpuThreads wins over -Dnanollvm.cpu.threads
// .dedicatedMatmulPool() cannot combine with .matmulExecutor(appPool)
```

### Sample A4 — next tokens / raw continuation (**since 1.1.0**)

```java
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.SamplingParams;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;

import java.nio.file.Path;
import java.util.List;

try (LlmModel model = LlmModelFactory.make(Path.of("models/Tiny-LLM-ONNX"));
     LLM llm = LLM.builder(model).disableMultiCpu().maxModelLen(256).build()) {
  Tokenizer tokenizer = llm.tokenizer();
  List<Integer> promptIds = tokenizer.encode("The capital of France is");
  SamplingParams sampling = SamplingParams.builder().temperature(0.2f).maxTokens(8).build();

  LLM.GenerationOutput out = llm.generateTokenIds(
      List.of(promptIds),
      sampling,
      (LLM.TokenEvent event) -> System.out.println(
          event.tokenId() + "\t" + tokenizer.decode(List.of(event.tokenId()))))
    .getFirst();

  System.out.println(out.text());           // newly sampled tokens only
  // String same = llm.complete("The capital of France is", 8);
}
```

Demo: `samples.NextTokenHelloWorld`. `complete` is the string shortcut; `generateTokenIds` is the token-id path
(optional per-token callback). Neither applies a chat template.

### Sample B — one generate tick (Sense A loop)

`LLM.generate` repeatedly calls **private** `stepUnlocked()` until `scheduler.isFinished()`. Each tick is:

```text
Scheduler.schedule()          // pick prefill or decode batch
    → Transformer.step(seqs, prefill)
        preparePrefill / prepareDecode  (+ Context.set slot maps, block tables)
        CausalLM.forward(inputIds, positions)
        CausalLM.computeLogits(hidden)
        Sampler.forward(logits, temperature, topK, topP)
    → Scheduler.postprocess(…)   // append token, finish on stop / maxTokens
```

Compressed from private `stepUnlocked` / `Transformer.step` (conceptual — `stepUnlocked` is not public):

```java
ScheduleResult scheduled = scheduler.schedule();
List<Integer> tokenIds = transformer.step(scheduled.sequences(), scheduled.prefill());
scheduler.postprocess(scheduled.sequences(), tokenIds, scheduled.prefill(), appendedOut);
```

Inside `Transformer.step`:

```java
Tensor hidden = network.forward(inputIds, positions);
Tensor logits = network.computeLogits(hidden);
int[] tokenIds = sampler.forward(logits, temperatures, topKs, topPs);
```

### Sample C — attention stores and reads the notebook

`layers.Attention.forward(q, k, v)`:

1. If slot mapping is set, `storeKvCache` writes K/V into paged `kCache` / `vCache`.
2. Prefill → `prefillWithCache` or `prefillDense`.
3. Decode → `decode` (map `blockTables` to slots, then `attendRange` **on those cache slots in place** — no dense K/V copy).

That is the code behind “write notebooks at prefill; reread them at decode.” What those tensors hold, how pages map to
slots, and how that differs from chat history: **chapter 12**.

### Sample D — loading shelves

```text
LlmModelFactory.make(dir|gguf|ModelFileSource):
  HF safetensors: HfConfig + ArchitectureProcessor → WeightBag → causal graph
                  (Qwen3 / Gemma3 / Gemma 4 text / Llama)
                  tokenizer.json, else tokenizer.model (1.2.0), else config.json fallback
  HF ONNX (1.1.0): OnnxTransport → ArchitectureProcessor bind/fill/create → same causal / BERT graphs
  Whisper (1.3.0): SpeechArchitecture → WhisperForAsr; LLM.transcribe
  Piper (1.3.0): SynthesisArchitecture → PiperForTts; optionalData(ESPEAK_DATA); LLM.synthesize
  GGUF: GgufTransport → ArchitectureProcessor bind/fill/create
        → Qwen3ForCausalLM or Lfm2ForCausalLM or BertForEmbedding; Tokenizer.fromGguf
  Classpath / streams (1.1.0): ModelFileSources → heap bytes (no disk cache)
  Load status (all three weight formats): models.llmcontainer.LoadProgress — one in-place percent/ETA bar
  Optional Map (1.1.0): LlmModel.OPTION_THINK_TAGS → ThinkTags; OPTION_CHAT_SPECIALS → ChatSpecials
                        (frozen on the model; omitted keys get library defaults)
  Optional (1.3.0): LlmModelFactory.Builder.optionalData / OPTION_OPTIONAL_DATA
LLM.builder(model).build():   // every kind — chat / embed / transcribe / synthesize
  chat: Transformer allocates KvCacheArena (per LLM); optional warmup / allowUnpackParameters
  non-chat: numKvcacheBlocks 0; same cpuThreads / matmul pool
```

### Sample E — chat thinking path (Sense C)

```text
ChatSession.send(user)
  history.add(user message)
  ChatMessages.truncateHistory(…)
  LLM.runAdvisors(modelUser, priorUsers, sampling)   // optional; Builder.advisors(mixer, …)
      → batched generate → AdvisorEnrichment (List<AdvisorResponse> inside)
      → mixer.mixPrompt(llm, responses, ChatHistory, prompt)
  Tokenizer.applyChatTemplate(history with mixed user, …)
  LLM.generate(prompt, sampling, onToken →
      TEXT_RAW (decode, specials kept)
      + AssistantParts.parse(partial decode) → TEXT_THINKING / TEXT_ASSISTANT)
  AssistantParts.parse(full decode) → ChatReply(thinking, answer, thinkOpen)
  finishTurn: maybe salvageFromThinking; history.add(assistant(answer only))
```

Key types: `ChatSession.send` / `generateTurn` / `finishTurn`, `LlmAdvisor` / `LlmAdvisorMixer` /
`AdvisorEnrichment` / `AdvisorResponse` / `ChatHistory`, `AssistantParts.parse`, `LlmTextKind.TEXT_RAW`
(unparsed decode) vs `TEXT_THINKING` / `TEXT_ASSISTANT`, `ChatPrompts.systemFor` (always empty),
`SamplingDefaults.forTokenizer` (neutral), optional `LLM.Builder#advisorNoteFilter`. `TEXT_DEBUG` (prepared
model-user after mix) is off unless `ChatSession.emitDebugPrompts(true)`. Advisors need **unique non-blank
names**; they share one batched `generate` and must not interleave with another `generate` on the same `LLM`.

### Sample F — where “2+2” meets attention in the model graph

For Qwen, one layer’s attention path is roughly
`Qwen3ForCausalLM.Qwen3Attention.forward`:

```text
hidden → qkvProj → split Q,K,V → optional qNorm/kNorm
      → RotaryEmbedding.forward(positions, q, k)
      → Attention.forward(q, k, v)
      → oProj
```

MLP: `gateUpProj` → `Ops.siluAndMul` → `downProj` (Gemma uses `geluPytorchTanhAndMul`).
GGUF Qwen3 uses the same rooms with **unfused** packed linears (`attn_q` / `attn_k` / `attn_v`, `ffn_gate` / `ffn_up`)
instead of fused `qkv_proj` / `gate_up_proj` (chapter 7a).

### Sample G0 — BERT embedding GGUF (**since 1.1.0**)

```text
LlmModelFactory.make(gte-small.Q2_K.gguf)
    → GgufModelLoader (arch bert) → BertForEmbedding
    → LlmModel (isEmbeddingModel == true)

LLM.builder(model).build().embed(text)   // since 1.3.0; shared CPU pool
model.embed(text)                        // sequential shortcut (dense RAG indexing)
    // E5: pass "query: …" or "passage: …" yourself; the library does not add it
    → tokenize + [CLS]/[SEP] or <s>/</s>
    → token+pos+type emb → LayerNorm → bidirectional blocks
    → mean pool → L2 normalize → float[]

// Dense RAG: DenseRagIndex.of(prepared, model) / RagFactory.withEmbeddings (ch. 17)
//   still calls LlmModel.embed (sequential unless you pass an Executor)
```

Narrative: **chapter 7b**. Demo: `samples.EmbeddingsHelloWorld` (default multilingual-e5-small ONNX), `Example` menu
embeddings items.

### Sample G0b — Whisper transcribe (**since 1.3.0**)

```text
LlmModelFactory.make(models/whisper-base)
    → SpeechArchitecture / WhisperProcessor → WhisperForAsr
    → LlmModel (isSpeechModel == true)

LLM.builder(model).build().transcribe(wavBytes | Path | pcm, optional Locale)
    → mix/resample 16 kHz → log-mel → encoder → greedy decode → String
```

Narrative: **chapter 7d**. Demo: `samples.TranscribeHelloWorld`.

### Sample G0c — Piper synthesize (**since 1.3.0**)

```text
LlmModelFactory.open(voiceFolder).optionalData(ESPEAK_DATA, espeakDir).make()
    → SynthesisArchitecture / PiperProcessor → PiperForTts
    → LlmModel (isSynthesisModel == true)

LLM.builder(model).build().synthesize(text)
    → G2P → VITS reverse → HiFi-GAN → PCM16 LE mono WAV bytes
```

Narrative: **chapter 7e**. Demo: `samples.SynthesizeHelloWorld`.

### Sample G — text RAG (prepare once, ask many times)

```text
RagFactory.make(docs|file) / .of(…) / .builder()… / makeResource(…)   // classpath since 1.1.0
    → CorpusLoader
         optional RagTuner (since 1.2.0; Builder.addProcessor):
           filter AND → first extract Optional → preprocess pipeline
         then Markdown cleanup, sentence packing
    → PreparedRag.fromChunks          // passage prep + inverted BM25 + IDF
    → PreparedRag                     // shareable like LlmModel

optional since 1.1.0:
  LlmModel embed = LlmModelFactory.make(gteGguf)   // embedding encoder; LLM.builder optional for a pool
  DenseRagIndex.of(prepared, embed)                // cosine over L2 vectors (sequential LlmModel.embed)
  DenseRagIndex.of(prepared, embed, executor)      // same, caller Executor
  HybridRagIndex / RagFactory.withEmbeddings(…)    // RRF over RagIndex sources (BM25+dense factory)

llm.rag(index).topK(k).send(user)   // any RagIndex: BM25, dense, or hybrid
    → RagIndex.retrieve
    → RagSession.formatUserMessage(…) / UserMessage.format
    → ChatSession.sendPrepared(historyUser, modelUser)
```

Narrative and design notes: **chapter 17**.

### How to read the tree on disk

```text
nano-vllm-java/src/main/java/com/igormaznitsa/nanollvm/
  llm/LLM.java                 ← start here for API
  llm/Config.java / SamplingParams.java
  chat/LlmListener.java / LlmListeners.java / LlmTextKind.java
  llm/AdvisorRunner.java / AdvisorPrompt.java
  llm/LlmAdvisor.java / LlmAdvisorMixer.java / AdvisorResponse.java / AdvisorEnrichment.java
  chat/ChatSession.java / ChatHistory.java / ChatMessage.java / ChatMessages.java
  rag/RagFactory.java          ← prepare documents once (Path, text, classpath)
  rag/RagTuner.java / RagResource.java  ← since 1.2.0: load-time filter / extract / preprocess
  rag/PreparedRag.java         ← shareable corpus + BM25 index (+ Passage record)
  rag/DenseRagIndex.java       ← since 1.1.0: embedding cosine index
  rag/HybridRagIndex.java      ← since 1.1.0: RRF over two or more RagIndex (any list since 1.2.0)
  rag/RagSession.java          ← retrieve → prompt → chat (UserMessage, QueryRewrite)
  rag/CorpusLoader.java        ← package-private load/chunk pipeline
  engine/Transformer.java      ← forward + sample (not exported)
  engine/Scheduler.java        ← prefill/decode batches
  layers/Attention.java        ← QKV cache + attendRange (paged slots in place)
  models/LlmModel.java / LlmModelFactory.java / LlmOptionalData.java
  models/ModelFileId.java / ModelFileSource.java / ModelFileSources.java   ← since 1.1.0
  models/internal/CausalLMFactory.java / WeightBag.java
  models/internal/BertForEmbedding.java / EmbeddingEncoder.java   ← since 1.1.0
  models/internal/WhisperForAsr.java / PiperForTts.java / TextToSpeech.java / SpeechToText.java   ← since 1.3.0
  models/internal/Qwen3ForCausalLM.java / Gemma3ForCausalLM.java / Gemma4ForCausalLM.java / LlamaForCausalLM.java / Lfm2ForCausalLM.java
  layers/BidirectionalAttention.java   ← BERT embed path (since 1.1.0)
  tokenizer/Tokenizer.java / GgufTokenizerSource.java   ← fromSentencePiece since 1.2.0; codecs package-private
  prompts/ChatPrompts.java / RagPrompts.java / AdvisorPrompts.java
  models/llmcontainer/SafetensorsTransport.java / SafetensorsReader.java / OnnxTransport.java / ContainerTransport.java
  models/llmcontainer/GgufTransport.java / GgufReader.java
  models/llmarch/ArchitectureProcessor.java / ArchitectureProcessors.java / ModelBinding.java
  models/llmarch/ModelLoader.java / GgufModelLoader.java / Gemma4QatLoader.java / GgufConfigs.java
  internal/Json.java / GgufDequant.java
  internal/Context.java        ← per-step KV / conv slot maps
  utils/NanoLlvmProps.java / ResourceLimits.java   ← ResourceLimits since 1.0.0
  tensor/LinearKernel.java / EmbeddingKernel.java / MatmulRuntime.java
  tensor/kernels/DenseF32LinearKernel.java / PackedLinearKernel.java / …
  exceptions/…

nano-vllm-java-samples/src/main/java/com/igormaznitsa/nanollvm/samples/
  Example.java / HelloWorld.java / NextTokenHelloWorld.java / Bench.java / EmbeddingsHelloWorld.java / TranscribeHelloWorld.java / SynthesizeHelloWorld.java / LogTriageHelloWorld.java / AdvisorRagHelloWorld.java / RagTunerHelloWorld.java
  utils/BundledModels.java / BundledRag.java / OrderedConsole.java / EpubText.java
```

### Threading reminder (API contract)

One `LLM` must not run concurrent `generate` / chat / `runAdvisors` calls. `LLM.cancel()` is safe from another thread
and clears in-flight work via the scheduler. One `LlmModel` / `PreparedRag` may be shared across many `LLM`s
(chapter 17). Dense / hybrid indexes (**since 1.1.0**) additionally need a live embedding `LlmModel` for query-time
`embed`. Close engines before the shared model.

---

## 17. Text RAG: documents beside the model

The earlier chapters describe how the **model** continues text from weights it already carries. Those weights are a
fixed library. They do not magically contain your project notes, your engine README, or yesterday’s fact sheet.

**Retrieval-augmented generation (RAG)** is a small organization around that fact: before the model guesses the next
token, the program **looks up relevant passages** from a document collection you prepared, and **places those passages
into the user turn** the chat template will see. The model still only does next-token prediction. The new work is
**which words appear in the prompt**.

This project’s RAG is **CPU-local** and works over **text passages** you load once. The default index is **BM25**
(classic lexical ranking) — the same spirit as loading a shared immutable `LlmModel` once and attaching many `LLM`
engines to it. **Since 1.1.0**, you may also rank by **dense embeddings** from a separate BERT-family GGUF (e.g.
gte-small) via `LlmModel.embed` — see **chapter 7b** for what BERT is and how encoding works — alone or **hybrid** with
BM25 (reciprocal rank fusion). There is still **no ANN vector database**: dense search is a linear scan over
precomputed passage vectors in process memory.

### Why bother (the humanities picture)

Imagine a student who may answer from memory, but who is also allowed to open a **folder of short cards** before
speaking. RAG is the act of:

1. **Preparing** those cards carefully (cut, label, index) — once.
2. **Selecting** a few cards that match the current question — every turn.
3. **Reading the cards aloud into the prompt**, then letting the model continue.

If the cards are wrong or noisy, a small model will still say confident nonsense. If the cards are tight and the prompt
is short, the same model has a much better chance of quoting or paraphrasing the right names.

### How it is organized (two timescales)

| When                      | What happens                                                                  | Analogy                           |
|---------------------------|-------------------------------------------------------------------------------|-----------------------------------|
| **Load time** (once)      | Read files → optional tuners → clean → chunk → BM25 index → `PreparedRag` | Binding the card box              |
| **Optional (1.1.0)**      | Embed each passage with an embedding `LlmModel` → `DenseRagIndex` / hybrid    | Photographing each card once      |
| **Query time** (each ask) | Score candidate cards (BM25 and/or cosine) → format prompt → chat generate    | Pulling a few cards, then writing |

`PreparedRag` is **immutable and shareable**. Many `LLM` instances (Qwen, Gemma, several chats) may point at the
**same** prepared index without rebuilding it — parallel to sharing one `LlmModel` across engines. A dense or hybrid
index keeps a reference to the embedding model for query-time `embed`; close that encoder only after the index is
unused.

**In the code (organization):** package `com.igormaznitsa.nanollvm.rag`; entry `RagFactory` → `PreparedRag` (and
optionally `withEmbeddings`); session `LLM.rag(index)` → `RagSession`; demo corpus folder `rag/` via
`samples.utils.BundledRag` in `samples.Example` (model → RAG-mode → advisor-count menus **since 1.1.0**)
and the linear `samples.AdvisorRagHelloWorld` (one custom advisor + BM25 over `rag/`) plus
`samples.RagTunerHelloWorld` (bundled *R.U.R.* EPUB through `RagTuner` extract + BM25).

### Load path — preparing documents

```text
folders / files / classpath resources / inline strings
        │  (one RagFactory.builder(); addFolders since 1.2.0)
        ▼
  CorpusLoader (package-private)
        │  optional RagTuner (since 1.2.0): filter files, custom extract, preprocess text
        │  optional Markdown cleanup, section titles, sentence packing
        │  RagLoadOptions: maxChunkChars, atomicSentences, dedupe, …
        ▼
  PreparedRag.fromChunks
        │  NFC normalize; model vs search text; source-stem tokens on searchText
        │  termFreqs per passage; inverted postings + IDF (inside PreparedRag)
        ▼
  PreparedRag   (passages + BM25 index + options; share freely)
        │
        │  optional since 1.1.0
        ▼
  DenseRagIndex / HybridRagIndex   (embed passages; keep encoder open for queries)
```

#### Load-time tuners (**since 1.2.0**)

The default loader reads **UTF-8**, then chunks. There is no built-in PDF (or EPUB) parser — those
formats need a **`RagTuner` extractor**, same pattern as `samples.RagTunerHelloWorld`. **Tuners** are optional
hooks on that path — skip a file, parse a format the library does not know, or rewrite text before sentence packing.
They are **load-time only**; they do not change BM25, dense rank, or `RagSession`.

Register them on the fluent builder (not on `RagFactory.make(path)`):

```java
PreparedRag prepared = RagFactory.builder()
    .addProcessor(
        RagTuner.allowing(resource -> !resource.fileName().startsWith("_")),
        RagTuner.extracting(resource -> resource.fileName().endsWith(".html")
            ? Optional.of(stripTags(resource))
            : Optional.empty()),
        RagTuner.preprocessing(String::strip))
    .folderExtensions(Set.of(".txt", ".md", ".html"))  // folder walks still use this set
    .addFolder(Path.of("docs"))
    .build();
```

Several `addProcessor` calls **append** in order. The list is not one nested “call the next” object. Each file or
classpath document is walked **three times**, with a different combine rule each time:

| Pass | Method | How the list is combined |
|------|--------|--------------------------|
| **Filter** | `isRagResourceAllowed(RagResource)` | **AND** — every tuner must return `true`, or the document is skipped (not read). Inline `add(text)` never hits this pass. |
| **Extract** | `extractRagText(RagResource)` | **First present `Optional`** — that string becomes the body; later extractors are not called. `Optional.empty()` means “not this format.” If every tuner is empty, UTF-8 runs. |
| **Preprocess** | `preprocessRagText(String)` | **Pipeline** — `c(b(a(text)))` in registration order, then the usual `RagLoadOptions.preprocess()` packing. Defaults are identity, so a filter-only tuner does not rewrite text. |

`RagResource` is the document handle: disk `Path` or `classpath:…` label, file name, and (at extract time) loaded
bytes. Folder walks still honor `folderExtensions`; add extra suffixes for custom extractors (`.pdf`, `.epub`, `.html`, …).
Override only the methods you need, or use `RagTuner.allowing` / `extracting` / `preprocessing`.
Demo: `samples.RagTunerHelloWorld` extracts a classpath EPUB, then indexes chunks with BM25.

**In the code (tuners):** `RagFactory.Builder.addProcessor` → package-private `RagTunerChain` inside `CorpusLoader`
(filter before read, extract in `readBody`, preprocess in `appendChunks`).

#### Chunking and cleanup

Load-time chunking is **structural**, not linguistic policy: strip code fences and Markdown links, keep heading text as
a section label on following sentences (`Capitals — Paris is…`), split on sentence boundaries (including common CJK
punctuation). It does **not** maintain dictionaries of user replies like “yes” / “no”.

With `atomicSentences` (see `RagLoadOptions.forTinyModels()`), each sentence stays its own chunk — better for tiny
generators that drown in long context. Optional **dedupe** drops identical normalized passages.

The character ceiling is `RagLoadOptions.maxChunkChars` (defaults **500**, tiny preset **220**). Change it with
`RagLoadOptions.defaults().withMaxChunkChars(n)` (or `withChunkOverlap`) and pass the options into
`RagFactory.make(path, options)` / `RagFactory.Builder.options`. Units are Java `char`s, not tokens; packing may
emit shorter chunks. Prompt concatenation is a separate cap (`RagSession.maxContextChars`, default 3500).

#### Preparsing and BM25 (inside `PreparedRag`)

At load, each passage gets:

- **Unicode NFC** and whitespace cleanup for stable tokens.
- **Split roles:** model-facing text stays readable for the prompt; **search** text may include **file-name stems** so a
  query about “capitals” can find `facts-capitals.md` even if the word appears only in the path.
- **Term frequencies** counted once per passage (`PreparedRag.Passage`).
- **Inverted BM25:** posting lists per term; queries score only candidate docs (Okapi BM25; weak hits dropped).

**In the code (load):** `RagFactory.make` / `of` / `builder` → `CorpusLoader` (UTF-8 text/markup;
**since 1.1.0** also `makeResource` / `Builder.addResource` for classpath paths,
source label `classpath:…`; **since 1.2.0** `Builder.addFolders` for several disk trees and
`Builder.addProcessor(RagTuner…)` for filter / extract / preprocess) → `PreparedRag.fromChunks`.
Options live in `RagLoadOptions`.

```java
// BM25 only — prepare once, share across many LLM engines
PreparedRag prepared = RagFactory.builder()
    .addFolders(Path.of("docs/kb"), Path.of("docs/policies"))
    .addResource(MyApp.class, "/help/faq.md")
    .add("support-hours", "Live chat is available 9–17 UTC.")
    .build();
// one folder: RagFactory.make(Path.of("rag"), RagLoadOptions.forTinyModels());
// one classpath file: RagFactory.makeResource("docs/facts.md");
// HybridRagIndex.of(index, index) fuses rankings (BM25+dense), not document lists.

assert prepared.isOutsideCorpus("what do you think about BMW?");
assert prepared.retrieve("what do you think about BMW?", 3).isEmpty();

try (LLM llm = LLM.builder(chatModel).build()) {
  String grounded = llm.rag(prepared).topK(2)
      .send("Who are the Brothers Grimm?").answer();
}
```

#### Dense and hybrid indexes (**since 1.1.0**)

Load an embedding GGUF with the same factory as chat models, then `embed` — `LLM.builder` on the encoder is allowed
for the shared CPU pool (**since 1.3.0**); dense RAG indexing still calls `LlmModel.embed`:

```java
PreparedRag lexical = RagFactory.make(Path.of("docs"));
LlmModel embed = LlmModelFactory.make(Path.of("models/gte-small.Q2_K.gguf"));
RagIndex index = RagFactory.withEmbeddings(lexical, embed); // HybridRagIndex, sequential embed
// or: DenseRagIndex.of(lexical, embed);
// or: DenseRagIndex.of(lexical, embed, executor);          // parallel embed; caller owns executor

try (LLM llm = LLM.builder(chatModel).build()) {
  llm.rag(index).topK(3).send("What city is France's capital?");
}
// Keep `embed` open while `index` is in use; close the embedding model only after the index is unused.
```

- **`DenseRagIndex`:** at build time embeds every chunk (calling thread, or a caller `Executor` since 1.2.0); at query time embeds the question and ranks by cosine (dot
  product on L2-normalized vectors). Linear scan — fine for small corpora.
- **`HybridRagIndex`:** fuses two or more `RagIndex` rankings with **RRF**
  (`HybridRagIndex.of(bm25, dense)` or `of(index, index, …)`). Nested hybrids flatten so each
  source ranks once. Off-topic gating requires **every** source to agree (`isOutsideCorpus`), so
  paraphrases can still retrieve when one side has no lexical overlap.
  `RagFactory.withEmbeddings` is the BM25+dense convenience factory.
- Chat generation stays on the causal `LLM`; the embedding model is only for retrieval.

### Query path — one RAG turn

```text
user text
   │
   ├─ (optional) if the turn has fewer than 6 tokens and a longer prior user turn exists,
   │             retrieval query = prior turn + current   // structural only (SHORT_FOLLOW_UP_MAX_TOKENS)
   │
   ▼
RagIndex.retrieve(query, topK)  →  List<RagHit>
   │  PreparedRag: BM25 + coverage/length re-rank
   │  DenseRagIndex: cosine over passage vectors          (since 1.1.0)
   │  HybridRagIndex: RRF over two or more RagIndex sources  (since 1.1.0; any list since 1.2.0)
   │  off-topic: index may refuse (hybrid needs every source to agree)
   ▼
RagSession.formatUserMessage(hits, user text, maxContextChars)
   │  (internal: UserMessage.format; wording in prompts.RagPrompts)
   │  hits: grounding line (“Answer using only the passages…”) + bullets + blank + question
   │  no hits: question only
   ▼
ChatSession.sendPrepared(historyUser = original text,
                         modelUser   = RAG prompt,
                         isolateGeneration?)
   │  isolateGeneration default false; when true, template = system seed + this turn only
   │  so prior wrong answers cannot latch the next generate
   ▼
 ordinary chat generate (chapters 14–15)
```

History keeps what the **human typed**. The **model** sees the retrieved passages on that turn. That split matters:
the conversation log stays readable; the generator gets the cards.

When several BM25 passages match, `PreparedRag.retrieve` re-ranks by query term coverage and passage length — compact
grounding without corpus-specific filename rules. Grounded turns also clamp sampling temperature.

Very short follow-ups (token count under **6**, not a word list) expand the **retrieval** string with the previous
longer user turn so a one-word reply does not become a random lexical hunt through the corpus. The model still receives
the current user text in history; chat context does the conversational work. Query path may also use nested
`QueryRewrite` (isolated LLM keywords) and prefer a prior source when the index supports it.

**In the code (query):** `RagSession.send` → `RagIndex.retrieve` → `RagSession.formatUserMessage` (or
`UserMessage.format`) → `ChatSession.sendPrepared`. Short follow-ups may use nested `QueryRewrite` (isolated LLM
keywords). `LLM.rag(RagIndex)` / `rag(index, maxTokens)` open the session.

### How this works *with* the model

RAG does not change attention math, the KV cache, or sampling math itself. It only changes the **token ids of the last
user message** (and thus the prefill), plus a temperature clamp on grounded turns. Everything in chapters 8–12 still
applies: longer RAG context means a heavier prefill; tiny models can still ignore instructions, so the stack prefers
short passages, a one-line “answer using only the passages / do not invent books” prefix (`RagPrompts.GROUNDING`),
optional `isolateGeneration` (library default **false**; demos may enable it), and low temperature
(`RagLoadOptions.forTinyModels()`, small `topK`, compact formatting in `samples.Example`).

Think of the layers again:

| Layer                         | Role                                              |
|-------------------------------|---------------------------------------------------|
| `PreparedRag`                 | Your documents, cut and BM25-indexed              |
| `RagTuner` / `RagResource`    | Optional load-time filter / extract / preprocess (**since 1.2.0**) |
| `DenseRagIndex` / hybrid      | Optional embedding rank (**since 1.1.0**)         |
| `RagSession`                  | Retrieve + format for this turn                   |
| `LLM` / `ChatSession`         | Same inference engine as plain chat               |

### Project demo corpus

The repository folder `rag/` holds sample Markdown (fairy-tale / Grimm demos). `samples.Example` loads it through
`samples.utils.BundledRag` when present (`-Dnanollvm.rag.dir` / `NANOLLVM_RAG_DIR` override the path). **Since 1.1.0**
the sample asks for a **RAG mode** after you pick a chat model: none (plain chat; **Enter**), BM25, dense, or hybrid
(dense and hybrid need a BERT-encoder GGUF or ONNX folder under `models/`), then **how many advisors** (`0`–`3`; **Enter** = none). Choosing
an embedding checkpoint still opens an encoder-session menu: embed REPL, or few-shot classify on
`LlmModel.embed` vectors (`samples.EmbeddingsHelloWorld` defaults to multilingual-e5-small
ONNX and adds `query: ` for that family).
`samples.AdvisorRagHelloWorld` is the non-interactive BM25 + custom-advisor path (Gemma3-270M, advisor Alex,
Grimm names and father). `samples.RagTunerHelloWorld` extracts a bundled EPUB of Čapek's *R.U.R.* with
`RagTuner` filter / extract / preprocess (plain text via JDK zip + StAX), indexes with BM25,
stamps the short OPF title on later chunks via a Markdown heading, isolates each generate from prior answers,
and asks questions from the play (`LLM.Builder.deterministic()` so repeats keep the same tokens).

### What this RAG is *not*

- Not an ANN / external vector database (dense search is in-process, linear).
- Not a query-time plugin — `RagTuner` runs only while the index is built.
- Not a guarantee of factual truth — only a way to **offer** text; the generator may still mis-copy it.
- Not a classifier of user intents or languages: preparation is about **documents**, not about scripting replies.
- Dense retrieval is **not** a second chat model — keep the chat `LLM` and the embedding `LlmModel` separate
  (`LLM.builder` on the encoder is only for `embed` / a CPU pool, not for `chat`).

### A fair one-sentence summary

> **Prepare documents once into a shareable BM25 index (optional tuners at load since 1.2.0; optional dense or hybrid
> embeddings since 1.1.0); each question pulls a few passages into the chat prompt; the model then continues as usual.**

**In the code (full map):** chapter 16 Sample G; types under `rag/`.

---

## 18. Word list

Short glossary. For the Java home of each idea, prefer the **In the code** notes in earlier chapters and the map in
**chapter 16**.

| Term you may meet     | Meaning                                                                                          |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `LlmModel`            | Pretrained parameters plus tokenizer and blueprint used for inference                            |
| `ModelFileSource`     | Stream/classpath (or custom) source of model bytes (**since 1.1.0**; no disk cache)              |
| `ResourceLimits`      | Process-wide caps for parsers, corpus, JSON/GGUF sizes, history (**since 1.0.0**)            |
| Loading               | Reading blueprint + dictionary + weight tensors into memory and wiring them; weight pour uses one percent/ETA bar for safetensors, GGUF, and ONNX |
| `config.json`         | Architectural hyperparameters (sizes, norms, RoPE) — not the learned weights                     |
| `tokenizer.json`      | Vocab, merges / Unigram scores, and text pipeline (string ↔ token ids)                            |
| `tokenizer.model`     | SentencePiece protobuf sidecar when `tokenizer.json` is absent (**since 1.2.0**)                  |
| Tensor                | Multidimensional numeric array with a shape; weights and activations are both tensors            |
| Shape / `numel`       | Axis lengths of a tensor; product = number of scalar elements                                    |
| `.safetensors`        | On-disk container: JSON catalog of named tensors + raw numeric payload                           |
| `.onnx` / ONNX        | Optional HF-folder weights (**since 1.1.0**, ch. **7c**): initializers only → float32; no ORT   |
| `.gguf` / GGUF        | Single-file container: binary metadata + GGML tensor table + aligned (often quantized) payload   |
| GGML type             | On-disk element/block layout in GGUF (`F32`, `Q4_K`, …); this port dequants to float32           |
| BPE                   | Byte-Pair Encoding: merge frequent pieces using an ordered merge list                            |
| `data_offsets`        | Byte range of one tensor inside a safetensors payload                                            |
| ChatML                | Turn markers `<\|im_start\|>` / `<\|im_end\|>` (LFM2 GGUF chat packaging here)                   |
| `hidden_size`         | Width $H$ of the residual stream; also the embedding dimension                                   |
| `intermediate_size`   | Temporary wider width inside each layer’s MLP expand→shrink step                                 |
| `num_hidden_layers`   | Number of stacked attention+MLP blocks                                                           |
| GQA heads fields      | `num_attention_heads` vs `num_key_value_heads` (sharing of KV cache groups)                      |
| Inference             | Running the pretrained model to produce text (not training)                                      |
| Token                 | A vocabulary unit with an integer id                                                             |
| Embedding (token)     | Matrix $E \in \mathbb{R}^{V \times H}$; row lookup starts the forward pass                       |
| BERT / embedding GGUF | Bidirectional encoder → mean-pool → L2 vector via `LLM.embed` / `LlmModel.embed` (**since 1.1.0**; `LLM.builder` **since 1.3.0**; ch. 7b)     |
| Whisper               | Audio→text from HF safetensors via `LLM.transcribe` (**since 1.3.0**; ch. 7d) |
| Piper                 | Text→WAV from `*.onnx` + `*.onnx.json` via `LLM.synthesize` (**since 1.3.0**; ch. 7e) |
| `LlmOptionalData`     | Typed load extras such as `ESPEAK_DATA` (**since 1.3.0**; `open(path).optionalData`) |
| Sentence embedding    | One fixed-length vector summarizing a string (cosine ≈ dot product after L2)                    |
| RoPE                  | Rotary Position Embedding: rotate pairs inside Q/K by angle(position); encodes relative distance |
| Tied embeddings       | Same matrix for input lookup and LM-head scoring (`tie_word_embeddings`)                         |
| Vocabulary            | Set of token ids the model may emit ($V =$ `vocab_size`)                                         |
| Logits                | Raw scores over the vocabulary before softmax / sampling                                         |
| Softmax               | Map raw scores to a probability distribution (nonnegative, sums to 1)                            |
| Attention             | Weighted combination of past Values using Query–Key similarities                                 |
| Self-attention        | Attention within the same sequence (not a second document)                                       |
| Causal                | May only attend to past and present positions, not the future                                    |
| Multi-head (MHA)      | Several parallel attention heads; each may have its own KV                                       |
| GQA                   | Grouped-query: several Query heads share one Key/Value group                                     |
| MQA                   | Multi-query: all Query heads share a single Key/Value pair                                       |
| Sliding window        | Attend only within a fixed recent span, not the full past                                        |
| Global attention      | Attend over the whole allowed past                                                               |
| Query / Key / Value   | Linear projections used by attention (search / address / content). Only **K** and **V** are cached (ch. 12) |
| Inner work (Sense A)  | Invisible stack of attention + MLP for each next token                                           |
| Chain of thought (B)  | Reasoning written as ordinary tokens in the reply                                                |
| Tagged scratchpad (C) | Written reasoning inside open/close markers (default `<think>…</think>`; override with `ThinkTags`) |
| ChatReply             | Parsed assistant turn: `thinking` (scratchpad), `answer` / `text()` (visible), `thinkOpen` (unclosed scratchpad while streaming), `stats` (`GenerationStats`; `NONE` until generate finishes). `parse(raw)` uses default think tags and `ChatSpecials`; `parse(raw, ThinkTags)` / `parse(raw, ThinkTags, ChatSpecials)` / `parse(raw, llm)` / `open(path).thinkTags` / `.chatSpecials` at load / `ChatSession.thinkTags` for custom markers. `ChatSession.send` salvages + attaches stats. History stores **answer only**. |
| `TEXT_RAW`            | `LlmTextKind` for the unparsed tokenizer decode on a `ChatSession` listener (think tags and chat specials kept). `TEXT_THINKING` / `TEXT_ASSISTANT` stay parsed. CLI `streamTo` ignores `TEXT_RAW`. |
| KV cache              | Per-layer **Key** and **Value** tensors for already-seen token positions (RoPE already applied to K). Query, weights, and chat strings are **not** stored. Paged blocks; lives inside one `generate` (ch. 12) |
| Prefill               | First pass over the prompt: compute Q/K/V for all prompt positions, write K/V into pages, sample the first new token |
| Decode                | Later tokens: Q/K/V for the newest token only, append one K/V slot, attend cached Keys/Values    |
| Sampling              | Drawing the next token from (filtered) probabilities                                             |
| Temperature           | Softmax temperature $\tau$; lower → more peaked                                                  |
| Context length        | How much past text fits in one forward pass                                                      |
| Context window        | Hard token budget for one forward (`maxModelLen` / `max_position_embeddings`)                    |
| Chat history          | `ChatSession` message list re-fed each turn; may be truncated                                    |
| Weights               | Learned parameters from `.safetensors` (float32) or `.gguf` (packed by default; dequant on use)  |
| Dedicated matmul pool | `LLM.Builder.dedicatedMatmulPool()` — engine-owned CPU workers shut down on `close()` (**since 1.2.0**) |
| Activations           | Ephemeral tensors produced during a forward pass                                                 |
| RAG                   | Retrieval-augmented generation: look up documents, put them in the prompt, then generate         |
| `PreparedRag`         | Immutable shareable corpus + BM25 index (`Passage` record; load once, like `LlmModel`)                |
| `PreparedRag.Passage` | Load-time model text, search text, and term frequencies for one chunk                               |
| BM25                  | Lexical ranking over passages (inverted index; default RAG path)                                     |
| `DenseRagIndex`       | Embedding cosine index over chunks (**since 1.1.0**; needs embedding `LlmModel`)                     |
| `HybridRagIndex`      | RRF over two or more `RagIndex` sources (**since 1.1.0**; any list **since 1.2.0**)                 |
| `RagTuner`            | Load-time filter / extract / preprocess (**since 1.2.0**; `RagFactory.Builder.addProcessor`)         |
| `RagResource`         | File or classpath document seen by tuners during load (**since 1.2.0**)                             |
| `RagSession`          | Retrieve → format prompt → `ChatSession.sendPrepared`                                            |

---

## 19. Honest limits

This project is a **teaching instrument**, not a production cloud service.

- It runs on the ordinary processor. Dense float checkpoints keep numbers in a memory-hungry float32 form; GGUF and
  Gemma 4 text QAT stay packed and dequantize on use.
- It is slower than GPU systems you meet in products.
- Small models hallucinate, waffle, and latch onto polite filler — especially if prompts are vague. RAG can offer the
  right passage (BM25 and, since **1.1.0**, dense/hybrid embeddings) and the generator may still garble it.
- **ONNX demos** (**since 1.1.0**, ch. **7c**): SmolLM2 Instruct is a tiny ChatML chat model; Tiny-LLM-ONNX is a
  **base** completion toy — do not expect solid Q&A from either. Decode may stop early on degenerate loops or at a
  demo `maxTokens` cap so answers can look cut off. `samples.NextTokenHelloWorld` is the linear next-token demo
  for Tiny-LLM.
- **Gemma 4:** text-only QAT mobile loads (**since 1.1.0**). Vision/audio generation, Gemma 4 GGUF, and MoE
  (`enable_moe_block`) are out of scope.
- **Whisper (**since 1.3.0**, ch. **7d**):** Hugging Face safetensors only; greedy multilingual decode; uncompressed
  WAV or PCM; no MP3, timestamps, VAD, beam search, CTranslate2 `model.bin`, Whisper GGUF, or Whisper ONNX.
- **Piper (**since 1.3.0**, ch. **7e**):** official `*.onnx` + `*.onnx.json`; Java VITS + optional espeak-ng-data;
  no streaming / SSML / JNI. Missing dictionaries fall back to letter-to-sound.
- **Weight formats are curated, not universal:** safetensors float dtypes plus packed Gemma 4 QAT (ch. 7); GGUF = `qwen3|lfm2|bert` + listed
  GGML quants (ch. 7a); ONNX = Tier A initializers, no ORT / no community `*_q4*`/`*_int8*`/`with_past` names, no
  float8 weights, no LFM2-from-ONNX (ch. **7c**); Piper is a separate ONNX synthesis path (ch. **7e**).
- **RAG binaries:** folder walks are UTF-8 text. There is no built-in PDF extractor; index PDF/EPUB with
  `RagTuner.extracting` (**since 1.2.0**, ch. 17).
- “Understanding,” “knowing,” “thinking,” and “meaning” here are **metaphors** for statistical continuation and inner
  arithmetic. A humanities reader is right to keep that distinction sharp.

If this guide did its job, you can now explain to another non-specialist:

> Loading unpacks a fixed library of trained numbers. Attention is how each moment rereads allowed parts of the current
> page — causally, often with shared notebooks (GQA), sometimes through a sliding window. Thinking is organized as a
> loop: silent layer-walks for every next token, and optionally written notes (even tagged ones) that later attention
> can reuse. Text RAG prepares a separate box of document cards once, pulls a few into the prompt, then the same loop
> continues. Whisper transcribes audio; Piper speaks text as WAV. Then the program repeatedly draws the next scrap of
> text until the reply ends — or returns a vector, a transcript, or a waveform.

That is enough to understand what this Java program is doing — and what it is not.

**In the code (limits mirrored by design):** CPU kernels in `Ops` / `VectorMath`; no GPU path; one `LLM` is not safe for
concurrent `generate` — use `cancel()` from another thread only to abort (chapter 16). A server that wants parallel
matmul without the process-wide `nanollvm-matmul-*` pool uses `LLM.Builder.dedicatedMatmulPool()` (**since 1.2.0**)
or supplies `.matmulExecutor(appPool)`. RAG defaults to lexical BM25;
**since 1.1.0** dense / hybrid embedding retrieval is also available (`DenseRagIndex` / `HybridRagIndex`, chapter 17).
Close `LLM` engines before a shared `LlmModel`.

---

## 20. External reading index

A single list of the links woven into earlier chapters. Prefer the in-chapter notes for context; use this as a bookmark
page.

Implementation homes stay in the **In the code** notes, **chapter 16**, **chapter 7b** (BERT embeddings **since
1.1.0**), **chapter 7c** (ONNX / Llama **since 1.1.0**), **chapter 7d** (Whisper **since 1.3.0**), **chapter 7e**
(Piper **since 1.3.0**), and **chapter 17** (RAG); this index is papers and format docs
only.

| Topic                      | Link                                                                                                                               |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Transformer / attention    | [Vaswani et al. (arXiv)](https://arxiv.org/abs/1706.03762)                                                                         |
| Annotated Transformer      | [Harvard NLP notebook](https://nlp.seas.harvard.edu/annotated-transformer/)                                                        |
| BPE subwords               | [Sennrich et al. (arXiv)](https://arxiv.org/abs/1508.07909)                                                                        |
| Byte-level BPE / GPT-2     | [OpenAI GPT-2 report (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| HF Tokenizers              | [Documentation](https://huggingface.co/docs/tokenizers/index)                                                                      |
| Chat templates             | [Transformers guide](https://huggingface.co/docs/transformers/chat_templating)                                                     |
| Model `config` class          | [PretrainedConfig](https://huggingface.co/docs/transformers/main/en/main_classes/configuration)                                    |
| Safetensors                | [HF docs](https://huggingface.co/docs/safetensors) · [GitHub format notes](https://github.com/safetensors/safetensors)             |
| ONNX weight folders        | [ONNX](https://onnx.ai/) · (this guide §7c — Tier A initializers, Llama, SmolLM2 / Tiny demos **since 1.1.0**)                    |
| GGUF                       | [ggml GGUF docs](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) · (this guide §7a — layout, dtypes, Qwen3 / LFM2)              |
| BERT (original paper)      | [Devlin et al. (arXiv)](https://arxiv.org/abs/1810.04805) · (this guide §7b — embedding GGUFs **since 1.1.0**)                    |
| Whisper ASR                | [Radford et al. (arXiv)](https://arxiv.org/abs/2212.04356) · (this guide §7d **since 1.3.0**) |
| VITS / Piper TTS           | [Kim et al. (arXiv)](https://arxiv.org/abs/2106.06103) · [Piper](https://github.com/OHF-Voice/piper1-gpl) · (this guide §7e **since 1.3.0**) |
| LFM2                       | [Liquid LFM2 blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) · [LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) |
| RoPE                       | [Su et al. / RoFormer (arXiv)](https://arxiv.org/abs/2104.09864)                                                                   |
| Token / positional embeds  | (this guide ch. 10; RoPE paper above)                                                                                              |
| Math catalog (formulas)    | (this guide ch. 10 — layer map, linear, RMSNorm, attention, RoPE, gated MLP, sampling, SIMD dots)                                  |
| Tensors / shapes           | (this guide ch. 7–10; `tensor.Tensor` in source)                                                                                   |
| GQA                        | [Ainslie et al. (arXiv)](https://arxiv.org/abs/2305.13245)                                                                         |
| Multi-query attention      | [Shazeer (arXiv)](https://arxiv.org/abs/1911.02150)                                                                                |
| PagedAttention / vLLM      | [Kwon et al. (arXiv)](https://arxiv.org/abs/2309.06180) · [vLLM GitHub](https://github.com/vllm-project/vllm)                      |
| nano-vllm (upstream idea)  | [GitHub](https://github.com/GeeeekExplorer/nano-vllm)                                                                              |
| Chain-of-thought prompting | [Wei et al. (arXiv)](https://arxiv.org/abs/2201.11903)                                                                             |
| RMSNorm                    | [Zhang & Sennrich (arXiv)](https://arxiv.org/abs/1910.07467)                                                                       |
| SwiGLU                     | [Shazeer (arXiv)](https://arxiv.org/abs/2002.05202)                                                                                |
| Nucleus (top-p) sampling   | [Holtzman et al. (arXiv)](https://arxiv.org/abs/1904.09751)                                                                        |
| Softmax                    | [Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)                                                                        |
| BM25 / Okapi ranking       | [Robertson & Zaragoza survey (PDF)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)                      |
| Text RAG in this project   | (this guide ch. 17 — BM25; dense/hybrid + classpath **since 1.1.0**; load-time tuners **since 1.2.0**) |
| BERT embed + dense RAG     | (this guide §7b + ch. 17 — `LLM.embed` / `LlmModel.embed`, E5 `query:` prefixes, `DenseRagIndex`, `withEmbeddings`) |
| Whisper speech             | (this guide §7d — `LLM.transcribe`, HF safetensors) |
| Piper synthesis            | (this guide §7e — `LLM.synthesize`, `optionalData(ESPEAK_DATA)`) |

For a **curated learning order** (what to read first, and why it helps this project), see **chapter 21**.

**Link check:** every URL in chapters 20–21 was HTTP-checked against live hosts (2026-08-21); Whisper / Piper entries
added with **1.3.0** (2026-08-25). Prefer the arXiv abs page, the PDF host named here, or the project GitHub/docs URL — do not invent alternate slugs.

---

## 21. Suggested literature (learning path)

Chapter 20 is a **bookmark table** of links already cited in this guide. This chapter is a **reading list**: a small set
of papers, format docs, and tutorials ordered so a careful reader can build intuition for *this* codebase — inference,
not training; CPU; Qwen3 / Gemma3 / Gemma 4 text / Llama / LFM2 / BERT-GGUF; ONNX weight folders (**since 1.1.0**);
Whisper and Piper (**since 1.3.0**); BM25 and dense RAG.

**Provenance:** every link below names a real paper, format doc, blog, Hub page, or video that was resolved live
(HTTP 200 / valid PDF / arXiv title match / YouTube oEmbed) on **2026-08-21**. Titles match the cited work (e.g. arXiv
IDs below are the canonical abs pages). If a host moves, prefer the **stable identifier** (arXiv id, DOI where given,
or the project’s own docs) over a guessed new URL.

You do not need every item. Prefer the path that matches what felt foggy while reading chapters 1–19.

### How to use this list

1. Skim the **Why here** column before opening a link.
2. Pair each reading with the matching **guide chapter** (and, when ready, chapter 16’s code map).
3. Treat blogs and annotated notebooks as scaffolding; treat the arXiv papers as the durable definitions.

### A. Foundations (what a Transformer *is*)

| Read | Why here | Guide |
|------|----------|-------|
| [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) | Defines multi-head attention, encoder/decoder stacks, and the vocabulary this whole field shares. | ch. 2, 8, 10 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (Harvard NLP) | Line-by-line companion to Vaswani — formulas next to code-shaped exposition. | ch. 8, 10 |
| [Jay Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) | Visual walkthrough of Q/K/V and multi-head attention for readers who want pictures before papers. | ch. 8 |
| [Andrej Karpathy, *Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (video) | From-scratch next-token intuition (training demo); clarifies what **inference** reuses after training is done. | ch. 2, 11, 15 |

### B. Tokens, chat wrapping, and configs

| Read | Why here | Guide |
|------|----------|-------|
| [Sennrich et al., subword BPE](https://arxiv.org/abs/1508.07909) | Why text becomes mergeable scraps instead of whole words. | ch. 3, 6 |
| [OpenAI GPT-2 report (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Byte-level BPE as used by many chat tokenizers (Qwen-style path here). | ch. 6 |
| [Hugging Face Tokenizers docs](https://huggingface.co/docs/tokenizers/index) | Shape of `tokenizer.json` (normalizer / pre-tokenizer / model / decoder). | ch. 6 |
| [Chat templating guide](https://huggingface.co/docs/transformers/chat_templating) | Role markers and `chat_template` — what `applyChatTemplate` implements. | ch. 3, 14, 15 |
| [PretrainedConfig](https://huggingface.co/docs/transformers/main/en/main_classes/configuration) | Mental model for `config.json` fields this loader reads. | ch. 5 |

### C. Weights on disk (safetensors, GGUF & ONNX)

| Read | Why here | Guide |
|------|----------|-------|
| [Safetensors documentation](https://huggingface.co/docs/safetensors) · [format notes](https://github.com/safetensors/safetensors) | Named tensors + dtype payload without pickle — HF folder path in this port. | ch. 7 |
| [ONNX](https://onnx.ai/) · [SmolLM2-135M-Instruct-ONNX](https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX) | Tier A initializer import (no ORT) — Llama / ChatML demos **since 1.1.0**. | ch. **7c** |
| [GGUF format notes (ggml)](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | Single-file metadata + GGML quants — Qwen3 / LFM2 chat and BERT embedding loads. | ch. 7a, 7b |
| [Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | Example Qwen3 GGUF crate (`general.architecture=qwen3`); not downloaded by this repo’s scripts. | ch. 7a |
| [Liquid LFM2 overview](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) · [LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) | Hybrid short-conv + GQA architecture this GGUF chat path also targets. | ch. 7a |

### D. Attention variants and positional math used here

| Read | Why here | Guide |
|------|----------|-------|
| [Su et al., RoFormer / RoPE](https://arxiv.org/abs/2104.09864) | Rotary positions on Q/K — relative distance without absolute embedding tables. | ch. 5, 8, 10 |
| [Shazeer, Multi-Query Attention](https://arxiv.org/abs/1911.02150) | Extreme KV sharing; ancestor of GQA thrift. | ch. 8 |
| [Ainslie et al., GQA](https://arxiv.org/abs/2305.13245) | Grouped-query attention — the usual KV-notebook sharing in modern chat models. | ch. 8, 12 |
| [Zhang & Sennrich, RMSNorm](https://arxiv.org/abs/1910.07467) | Normalization used on many decoder stacks here. | ch. 10 |
| [Shazeer, SwiGLU](https://arxiv.org/abs/2002.05202) | Gated MLP form behind much of Qwen-style feed-forward bulk. | ch. 10 |

### E. Inference serving ideas (KV cache, paging)

| Read | Why here | Guide |
|------|----------|-------|
| [Kwon et al., vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) · [vLLM](https://github.com/vllm-project/vllm) | Why KV lives in **pages** and why prefill ≠ decode cost — inspiration for this engine’s block manager. | ch. 12, 13, 15 |
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | Compact Python reference this Java port is conceptually close to. | ch. 4, 16 |

### F. Sampling and “thinking” as text

| Read | Why here | Guide |
|------|----------|-------|
| [Holtzman et al., nucleus (top-p) sampling](https://arxiv.org/abs/1904.09751) | Why greedy argmax is not the only (or best) draw from logits. | ch. 11 |
| [Wei et al., chain-of-thought prompting](https://arxiv.org/abs/2201.11903) | Written intermediate steps as ordinary tokens — Sense B in this guide. | ch. 9 |
| Softmax ([Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)) | Scores → probabilities before temperature / top-k / top-p. | ch. 10, 11 |

### G. Embeddings, BERT, and RAG

| Read | Why here | Guide |
|------|----------|-------|
| [Devlin et al., BERT](https://arxiv.org/abs/1810.04805) | Bidirectional encoder — ancestor of the GGUF embedding path (`LLM.embed` / `LlmModel.embed`). | ch. 7b, 8 |
| [Reimers & Gurevych, Sentence-BERT](https://arxiv.org/abs/1908.10084) | Why mean-pooled encoders became practical **sentence vectors** for similarity search. | ch. 7b, 17 |
| [Lewis et al., RAG](https://arxiv.org/abs/2005.11401) | Original retrieval-augmented generation framing (retrieve → condition generation). | ch. 17 |
| [Robertson & Zaragoza, BM25 foundations (PDF)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) | Lexical ranking behind `PreparedRag`. | ch. 17 |
| [Cormack et al., Reciprocal Rank Fusion (PDF)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) | Fusing ranked lists — spirit of hybrid BM25 + dense RRF here. | ch. 17 |

*(Dense demo weights: **gte-small** GGUF via `models/download-gte-small-gguf.sh`; `EmbeddingsHelloWorld` defaults to
[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) ONNX.)*

### H. Model families this port actually opens

These are **optional depth** once the mechanics above are clear. Prefer the Hub model card + `config.json` beside any
technical report.

| Family | Starting points | Guide |
|--------|-----------------|-------|
| **Qwen3** | Model cards on [Hugging Face (Qwen)](https://huggingface.co/Qwen); this repo’s `models/download-*.sh` | ch. 5, 15 |
| **Gemma3** | [Gemma docs](https://ai.google.dev/gemma/docs) · example Hub card [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it) | ch. 5, 8 (sliding window) |
| **Gemma 4 text** | [Gemma docs](https://ai.google.dev/gemma/docs) — this port loads **text QAT mobile** safetensors only (not vision/audio, not GGUF/ONNX) | ch. 2, 7 |
| **Llama** | HF safetensors or ONNX (Tiny-LLM / SmolLM2 Instruct demos) | ch. **7c** |
| **LFM2** | Liquid blog + GGUF card linked in §C | ch. 7a |
| **BERT / GTE / E5 embeddings** | Devlin + Sentence-BERT above; Hub **gte-small** GGUF; [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) ONNX | ch. 7b |
| **Whisper** | [Radford et al.](https://arxiv.org/abs/2212.04356) · [openai/whisper-base](https://huggingface.co/openai/whisper-base) | ch. **7d** |
| **Piper** | [VITS](https://arxiv.org/abs/2106.06103) · [Piper](https://github.com/OHF-Voice/piper1-gpl) | ch. **7e** |

### I. Java / platform notes (only as needed)

This guide is not a Java textbook. When the **code map** (chapter 16) is the bottleneck:

| Topic | Where |
|-------|--------|
| Module exports vs internals | `src/main/java/module-info.java` (exported: `models`, `llm`, `chat`, `rag`, `exceptions`, `tokenizer`, `utils`) |
| Optional Vector API | JDK incubating `jdk.incubator.vector` — used when present; scalar fallback otherwise |
| Build / run | project `README.md`, `pom.xml`, `.mvn/jvm.config` (heap / module flags) |

### A short recommended order

If you want one linear path through the literature:

1. Illustrated Transformer **or** Annotated Transformer → Vaswani abstract/sections on attention.  
2. GPT-2 BPE section + HF chat templating.  
3. Safetensors **or** GGUF notes (whichever crate you open first).  
4. RoPE + GQA papers (short).  
5. PagedAttention / vLLM paper (KV paging intuition) + glance at nano-vllm.  
6. Holtzman (sampling) + Wei (chain-of-thought) as needed.  
7. BM25 survey + Lewis RAG; Sentence-BERT if you use dense / hybrid (**since 1.1.0**); RagTuner extractors if you
   index PDF/EPUB (**since 1.2.0**, ch. 17).  
8. Return to **chapter 16** and read the named classes with the papers as vocabulary.

### One-sentence close

> **Papers define the vocabulary; this guide maps that vocabulary onto a small CPU Java inference engine — load fixed
> weights, attend with a paged KV notebook, optionally retrieve documents, then sample the next token.**

---
