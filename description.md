# How a Language Model Works

### A plain-language guide to this Java project (Nano-vLLM)

This little book is written for a curious reader who is comfortable with ideas, stories, and careful argument — **not**
for someone who already thinks in code or equations.

You do not need to know Java, matrices, or “AI engineering.”  
If a term is unavoidable, it is explained the first time in everyday language.

The project this book describes is a small program that can **load a ready-made language model** and use it to continue
text or hold a short conversation — on an ordinary computer, without special graphics hardware. Java package / JPMS
module: `com.igormaznitsa.nanollvm`.

Where a topic has a standard paper or format guide, you will find a short **Further reading** note with links. Those
links are optional depth — the story in this book stands alone.

Where a topic has a home in this library, you will also find a short **In the code** note naming the Java types and
methods that implement it (package `com.igormaznitsa.nanollvm`). Those notes are signposts — **chapter 16** collects the
full map, samples, and file paths.

---

## Table of contents

1. [What this is really about](#1-what-this-is-really-about)
2. [The one trick: guessing the next piece of text](#2-the-one-trick-guessing-the-next-piece-of-text)
3. [Cutting language into pieces the machine can count](#3-cutting-language-into-pieces-the-machine-can-count)
4. [Loading a model: opening the library box](#4-loading-a-model-opening-the-library-box)
5. [`config.json` — the blueprint field by field](#5-configjson--the-blueprint-field-by-field)
6. [`tokenizer.json` — the dictionary file field by field](#6-tokenizerjson--the-dictionary-file-field-by-field)
7. [`*.safetensors` — the weight crates: format and contents](#7-safetensors--the-weight-crates-format-and-contents)
8. [Attention: kinds of looking-back, and how they work](#8-attention-kinds-of-looking-back-and-how-they-work)
9. [The thinking process: how it is organized and how it works with the model](#9-the-thinking-process-how-it-is-organized-and-how-it-works-with-the-model)
10. [The math, said gently](#10-the-math-said-gently)
11. [Choosing a word: not always the most obvious one](#11-choosing-a-word-not-always-the-most-obvious-one)
12. [Why the program keeps a notebook of the past](#12-why-the-program-keeps-a-notebook-of-the-past)
13. [Serving several conversations without chaos](#13-serving-several-conversations-without-chaos)
14. [Chat versus finishing a sentence](#14-chat-versus-finishing-a-sentence)
15. [A full walk-through: “What is 2+2?”](#15-a-full-walk-through-what-is-22)
16. [Where it lives in the code (classes, methods, samples)](#16-where-it-lives-in-the-code-classes-methods-samples)
17. [Word list](#17-word-list)
18. [Honest limits](#18-honest-limits)
19. [External reading index](#19-external-reading-index)

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

The rest of this book unpacks that sentence without assuming a technical background.

**In the code:** front door is `LLM` / `LLM.Builder`; interactive demo is `Example` (chapter 16).

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

Two families of models are supported here (different “editions” of the book, same kind of reading process): **Qwen3**
and **Gemma3**. You usually need not care which; the program detects which files you pointed it at.

**In the code:** architecture pick is `CausalLMFactory.detect` / `create` → `Qwen3ForCausalLM` or
`Gemma3ForCausalLM`; one next-token step is `ModelRunner.run` → `CausalLM.forward` / `computeLogits` → `Sampler.forward`
(chapter 16).

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
human language and that arithmetic.

A separate file, the **tokenizer**, holds that dictionary and the rules for chopping text. Chat models also store a
**template**: stage directions such as “this line is the user,” “this line is the assistant,” so the model is not
confused about who is speaking. Without those markers, a dialogue looks like an undifferentiated blob of prose.
**Chapter 6** opens `tokenizer.json` field by field.

**In the code:** `tokenizer.Tokenizer` — `fromPretrained`, `encode`, `decode`, `applyChatTemplate` (chapter 16).

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

You (or a download script in this project’s `models/` folder) fetch a **model directory** from a public model hub. That
directory is not a single magical file. It is a small collection of roles:

```text
models/Qwen3-0.6B/          (example)
  config.json               ← blueprint of the building
  tokenizer.json            ← dictionary + chat stage directions
  model-….safetensors       ← the heavy crates of learned numbers
  (sometimes more shards, license notes, extras…)
```

Until these files are **read into memory and wired into empty structure**, the program cannot answer anything. Loading
is that wiring.

### Three kinds of cargo (do not mix them up)

**1. The blueprint (`config.json`)**  
A list of measurements: how many stacked “reading rooms” (layers), how wide each stream of numbers is, how many
attention heads, how long a passage may be, which recipe (Qwen vs Gemma), and similar.

This is like an architect’s plan. It does **not** contain opinions about France or arithmetic. It only tells the program
how large the furniture must be. **Chapter 5** explains every field this project reads, with real Qwen and Gemma
examples.

**2. The dictionary (`tokenizer.json` and friends)**  
How human text becomes token numbers and back, plus special markers for “user” and “assistant.”  
This is the spelling system and the stage-direction language — still not the “knowledge.” **Chapter 6** details
`tokenizer.json` (and friends such as `tokenizer_config.json`).

**3. The learned numbers (`*.safetensors`)**  
The big cargo. Millions or billions of numbers shaped by training. These are what make one model sound different from
another. **Chapter 7** explains the binary layout, dtypes, and tensor names.

On disk they are often stored in a compact form (half-precision). This Java program **converts them into ordinary
decimal floating numbers in RAM** so the calculations stay simple to follow. Easy to teach; hungry for memory.

### What loading does, step by step

Think of a librarian preparing a reading desk:

1. **Read the blueprint**  
   The program opens `config.json` and learns the measurements.

2. **Build empty furniture**  
   It constructs the stack of reading rooms, empty weight shelves, empty embedding card-index, empty final scoring
   table — all the right *shapes*, still filled with zeros or placeholders.

3. **Open the heavy crates**  
   Each `.safetensors` file begins with a catalog (tensor name → where the bytes live). The loader walks that catalog.

4. **Match names to shelves**  
   A name like “layer 7’s output mix” must land in layer 7’s output-mix shelf. If the crate uses three separate packs
   for Query, Key, and Value, this project **merges them into one wider shelf** while loading (same contents, fewer
   drawers). Some models **share** the card-index with the final scoring table (one physical shelf, two jobs).

5. **Skip what does not belong**  
   Extra files or unrecognized names are ignored. Only registered shelves get filled.

6. **Open the dictionary**  
   Tokenizer files are loaded separately. They never become part of the neural arithmetic; they stand at the door,
   translating.

7. **Lay out blank notebooks**  
   Memory pages for Keys and Values during conversation are **created empty** here. They are not downloaded knowledge;
   they are scratch paper for the current text.

8. **Optional warm-up**  
   A tiny pretend question is run once so the first real answer is not also paying start-up costs.

After this, the **learned shelves stay fixed**. Chat does not rewrite the model files. Only the notebooks and temporary
worksheets change while answering.

**In the code:** `LLM` / `ModelRunner` construction runs `ModelLoader.loadModel`, `Tokenizer.fromPretrained`, and
`allocateKvCache` (chapter 16).

**Further reading:** Hub layout and `from_pretrained`-style folders are covered in Hugging Face
[Transformers docs](https://huggingface.co/docs/transformers); this Java port is inspired
by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and the serving ideas in
[vLLM / PagedAttention](https://arxiv.org/abs/2309.06180).

```text
  disk crate                    memory after load
  ─────────                    ─────────────────
  blueprint          ──►       sizes & recipe chosen
  dictionary         ──►       encode / decode ready
  weight crates      ──►       full shelves (fixed)
  (nothing yet)      ──►       empty KV notebooks
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
crashes. A successful load means: **every expected shelf has its numbers**, dictionary ready, notebooks allocated.

**In the code (shelves):** `CausalLM.getParameter` / `WeightSlot.load`; notebooks via
`ModelRunner.allocateKvCache` on each `Attention` (chapter 16).

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
| `model_type`    | Short family name (`qwen3`, `gemma3_text`, …)                                         | Auto-detect Qwen3 vs Gemma3 graph (`CausalLMFactory`) | Fall through to `architectures`, else assume Qwen3 |
| `architectures` | List of class-style names from Hugging Face (`Qwen3ForCausalLM`, `Gemma3ForCausalLM`) | Same detection if `model_type` is unclear             | Optional                                           |

You can override detection with `-Dnanovllm.arch=qwen3` or `gemma3` without editing the file.

**Examples from real folders**

- Qwen3-0.6B: `"model_type": "qwen3"`, `"architectures": ["Qwen3ForCausalLM"]`
- Gemma3-270M: `"model_type": "gemma3_text"`, `"architectures": ["Gemma3ForCausalLM"]`

---

### Size of the dictionary and the stream

| Field               | Means (short)                                                             | Used for                                                  | If missing                  |
|---------------------|---------------------------------------------------------------------------|-----------------------------------------------------------|-----------------------------|
| `vocab_size`        | How many distinct token ids the model knows                               | Width of embedding table and LM head (rows)               | Treated as 0 → broken model |
| `hidden_size`       | How many numbers describe **each token’s ongoing state** inside the stack | The main “working page width” for attention and residuals | 0 → broken                  |
| `intermediate_size` | How wide the **temporary expansion** is inside each layer’s MLP rewrite   | Gate/up and down projection sizes                         | 0 → broken                  |

### What `hidden_size` really is

Imagine each token, while it travels through the model, carries a **fixed-length dossier** — not one score, but a long
list of numbers (a vector). That list’s length is `hidden_size`.

- After the embedding lookup, token id `42` becomes a dossier of length `hidden_size`.
- Attention mixes dossiers, but each position still leaves with a dossier of the **same** length.
- Residuals add edits onto that same-width stream.
- At the very end, that dossier is compared against every vocabulary row to score the next token.

So `hidden_size` is the **width of the model’s internal working page** — how many slots of “description” each place in
the text keeps from layer to layer. Papers sometimes call it *d_model* or *model dimension*.

It is **not**:

- the number of layers (that is `num_hidden_layers`);
- the dictionary size (that is `vocab_size`);
- how many tokens of context you may use (that is `max_position_embeddings` / your `maxModelLen`).

```text
  token "cat"  →  [ n₁, n₂, n₃, … , n_H ]
                   ◄────── hidden_size H ──────►

  same width after attention, after MLP, after every layer
```

Larger `hidden_size` → richer dossiers → usually more capacity and more memory/compute per token. Smaller → leaner.

### What `intermediate_size` really is

Inside **each** layer, after attention, there is a private rewrite block (the MLP / feed-forward net). That block does
**not** stay at width `hidden_size` the whole time. It typically:

1. **Expands** the dossier from `hidden_size` to `intermediate_size` (often about 2×–4× wider),
2. Applies a nonlinearity / gate (SiLU, GELU, …),
3. **Shrinks** back to `hidden_size` before joining the residual stream again.

So `intermediate_size` is the width of that **temporary wide workshop** used only inside the MLP. The main hallway of
the model stays `hidden_size`; the side workshop is `intermediate_size`.

```text
  dossier (hidden_size H)
        │
        ▼
  expand to intermediate_size I   ◄── wide workshop (temporary)
        │
     gate / activate
        │
  shrink back to H
        │
        ▼
  add onto residual (still width H)
```

Why expand? A wider workshop gives the layer more room to recombine features before returning to the shared stream. Most
of a decoder layer’s **weight bulk** often sits in these expand/shrink matrices — that is why
`intermediate_size` matters so much for file size and RAM.

### How the two fit together

|                            | `hidden_size` (H)                                              | `intermediate_size` (I)                                  |
|----------------------------|----------------------------------------------------------------|----------------------------------------------------------|
| Lives where?               | Everywhere: embeddings, attention mixes, residuals, final norm | Only inside each layer’s MLP                             |
| Stays for the whole stack? | Yes — same H from first embed to last layer                    | No — temporary per layer                                 |
| Everyday metaphor          | Width of the manuscript page you keep rewriting                | Width of the blotter you use while editing one paragraph |
| Typical relation           | Baseline                                                       | Often roughly 2×–4× H (a design choice, not a law)       |

**Real values** (from the example model folders this project documents — not quality rankings):

|                         | Qwen3-0.6B       | Gemma3-270M        |
|-------------------------|------------------|--------------------|
| `vocab_size`            | 151936           | 262144             |
| `hidden_size` (H)       | **1024**         | **640**            |
| `intermediate_size` (I) | **3072** (= 3×H) | **2048** (≈ 3.2×H) |

So Qwen’s “page” is wider (1024 vs 640), and both use an MLP workshop about three times as wide as the page.

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
| `tie_word_embeddings` | If true, embedding table and LM head **share** the same numbers | Saves a huge matrix; load path may skip a separate `lm_head` | For Gemma `model_type`, default **true**; otherwise **false** |

**Real values:** Qwen3-0.6B sets `true`. Gemma often omits a separate LM-head tensor on disk; the loader’s Gemma default
matches that habit.

---

### Attention extras

| Field                   | Means                                         | Used for                                                                | If missing                  |
|-------------------------|-----------------------------------------------|-------------------------------------------------------------------------|-----------------------------|
| `attention_bias`        | Whether QKV projections add a bias vector     | Qwen path: if false, uses Q/K RMSNorm instead of bias                   | Default `false`             |
| `query_pre_attn_scalar` | Number used to scale attention scores (Gemma) | `attentionScale = 1 / sqrt(scalar)` (or `head_dim` if scalar absent/≤0) | 0 → fall back to `head_dim` |

**Real values:** both sample models set `attention_bias` false. Gemma3-270M sets `query_pre_attn_scalar` to 256 (same as
its `head_dim` here).

---

### RoPE — the position twist

| Field                  | Means                                                            | Used for                                                                  | If missing               |
|------------------------|------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------|
| `rope_theta`           | Base frequency for rotary position embeddings (global / default) | Build cos/sin tables for RoPE on Q and K                                  | Default `1000000`        |
| `rope_local_base_freq` | Separate base for **local / sliding** layers (Gemma)             | Sliding layers use this instead of `rope_theta`                           | Default `10000`          |
| `rope_scaling`         | Optional object describing long-context RoPE tricks              | If it contains `rope_theta`, that value can override the base (Qwen path) | `null` / absent → ignore |

Larger `rope_theta` is often used for longer contexts. Local layers with a smaller base keep nearby positions distinct
when the window is short.

**Real values:** both use `rope_theta` ≈ 1 000 000; Gemma also sets `rope_local_base_freq` to 10000; `rope_scaling` is
null in both samples.

---

### Sliding window and mixed layer types (Gemma)

| Field            | Means                                                            | Used for                                             | If missing                                                                                                       |
|------------------|------------------------------------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `sliding_window` | How many recent tokens a local layer may look back               | Window width for sliding attention                   | `0` or null → no windowing unless layer types say otherwise                                                      |
| `layer_types`    | Per-layer list: e.g. `"sliding_attention"` vs `"full_attention"` | `isSlidingLayer(i)` decides window + which RoPE base | If absent but `sliding_window` > 0: Gemma-style default — full attention every 6th layer (1-based), else sliding |

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

> Family qwen3; 28 rooms; each token keeps a **1024**-wide dossier (`hidden_size`); each MLP briefly widens to
> **3072** (`intermediate_size`, 3×); vocabulary ~152k; 16 Query heads with 8 KV groups
> (GQA); head size 128; SiLU MLP; embeddings tied to LM head; RoPE base 1e6; context claim up to 40960; weights stored
> as BF16 on disk, run as float32 here.

#### Gemma3-270M (compressed reading)

> Family gemma3_text; 18 rooms; dossier width **640**; MLP workshop **2048** (~3.2×) with GELU-tanh gate; vocabulary ~
> 262k; 4 Query
> heads sharing 1 KV head; head size 256; attention scaled by `query_pre_attn_scalar` 256; many layers only look back
> 512 tokens, every sixth layer looks globally; local RoPE base 10k, global 1e6; expect tied embeddings.

---

### Blueprint vs engine knobs (do not confuse)

| Comes from `config.json`                  | Comes from this program’s builder / runtime                                         |
|-------------------------------------------|-------------------------------------------------------------------------------------|
| Layer count, widths, heads, RoPE, windows | `maxModelLen` (capped by blueprint), KV page size, number of KV pages, batch limits |
| Which recipe (Qwen/Gemma)                 | `-Dnanovllm.arch=…` override                                                        |
| `torch_dtype` hint                        | Always float32 compute after load                                                   |

The blueprint says what the **model is**. The builder says how hard you ask your **machine** to run it.

**In the code:** `Config.HfConfig.load` fills the blueprint; `LLM.Builder` / `Config` set runtime knobs such as
`maxModelLen`, KV pages, and batch limits (chapter 16).

**Further reading:** configuration objects in the Python ecosystem —
[Hugging Face `PretrainedConfig`](https://huggingface.co/docs/transformers/main/en/main_classes/configuration); model
cards on the Hub usually ship the same `config.json` this chapter describes.


---

## 6. `tokenizer.json` — the dictionary file field by field

If `config.json` is the architect’s plan for the neural building, **`tokenizer.json`** is the **spelling book**: how
human text becomes token numbers and back. It is also JSON, but its job is linguistic plumbing, not layer widths.

This project loads it in `Tokenizer.fromPretrained`. Companion files often sit beside it:

| File                     | Role here                                                                  |
|--------------------------|----------------------------------------------------------------------------|
| `tokenizer.json`         | Vocab, merges, normalizer / pre-tokenizer / decoder pipeline, added tokens |
| `tokenizer_config.json`  | Often holds `chat_template`, `eos_token`, `pad_token` names                |
| `generation_config.json` | May list extra `eos_token_id` values used as stop ids                      |

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

### `model` — the BPE heart

| Subfield                                                  | Means                                                | Role                                                          |
|-----------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------|
| `type`                                                    | Almost always `"BPE"` here                           | Byte-Pair Encoding family                                     |
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

### Two encoding styles this project supports

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

> **`tokenizer.json` stores the vocabulary, the merge recipe, and the text-cleanup pipeline so strings become the
> integer ids the embedding table understands — and back again.**

**In the code:** `Tokenizer.fromPretrained` reads `tokenizer.json` (+ `tokenizer_config.json` chat template and stop
ids); encode/decode and `applyChatTemplate` live on `Tokenizer` (chapter 16).

**Further reading:** the JSON pipeline (normalizer → pre-tokenizer → model → decoder) is the
[Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) design; API overview of the `Tokenizer` class is
[here](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer).


---

## 7. `*.safetensors` — the weight crates: format and contents

The large files named `model.safetensors` or `model-00001-of-00003.safetensors` hold the **learned numbers**. This
project reads them with `SafetensorsReader` and pours them into the empty shelves built from `config.json`.

### Why this format exists

Older workflows used opaque pickles. **Safetensors** is a simple, mmap-friendly layout:

1. A small binary length prefix.
2. A JSON **catalog** naming every tensor.
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

**Important:** after load, compute is **float32** in this teaching engine. A BF16 file still becomes a large F32
resident image in RAM.

**Samples:** both Qwen3-0.6B and Gemma3-270M ship **all** listed tensors as `BF16` in the inspected files.

### What the contents *are* (names and roles)

Tensors are not mysterious blobs; their **names** say which shelf they fill. Typical patterns:

| Name pattern                                                                  | Content                                                                         |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `model.embed_tokens.weight`                                                   | Embedding table `[vocab_size, hidden_size]` — one portrait per token id         |
| `model.layers.i.input_layernorm.weight`                                       | RMSNorm scales for layer *i* (before attention)                                 |
| `model.layers.i.self_attn.q_proj.weight`                                      | Query projection (HF may store Q/K/V separately)                                |
| `model.layers.i.self_attn.k_proj.weight`                                      | Key projection                                                                  |
| `model.layers.i.self_attn.v_proj.weight`                                      | Value projection                                                                |
| `model.layers.i.self_attn.o_proj.weight`                                      | Output mix after attention                                                      |
| `model.layers.i.self_attn.q_norm.weight` / `k_norm.weight`                    | Optional per-head Q/K norms                                                     |
| `model.layers.i.post_attention_layernorm.weight`                              | Norm before MLP                                                                 |
| `model.layers.i.mlp.gate_proj.weight` / `up_proj.weight` / `down_proj.weight` | Gated MLP pieces                                                                |
| `model.layers.i.pre_feedforward_layernorm.weight` / `post_…`                  | Extra Gemma norms                                                               |
| `model.norm.weight`                                                           | Final norm before scoring                                                       |
| `lm_head.weight`                                                              | Vocab scorer `[vocab_size, hidden_size]` — may be absent if embeddings are tied |

**Counts in samples:** Qwen3-0.6B ≈ **311** tensors in one file; Gemma3-270M ≈ **236** (tied head → often no separate
`lm_head` tensor).

### Shards (several files)

Large models may split weights across `model-00001-of-0000N.safetensors`. The loader **lists all `*.safetensors`**,
opens them in sorted order, and plans every matching tensor. Each file has its **own** header + payload; names must not
collide across shards for the same parameter.

### How this project uses a tensor at load time

```text
  1. Read header → know names, dtypes, shapes, byte ranges
  2. Match name → WeightSlot on the live Java graph
       (rewrite q_proj/k_proj/v_proj → fused qkv_proj, etc.)
  3. mmap / read payload slice → convert to float[] Tensor
  4. slot.load(tensor) → copy into the layer’s weight storage
```

Unknown names are skipped. Missing **required** names leave empty shelves and break inference.

### What is *not* in `.safetensors`

- Tokenizer strings and merges.
- Chat templates.
- KV cache / attention notebooks (created empty at runtime).
- RoPE cos/sin tables (computed from `rope_theta` in config).

Only **trained parameters** (and sometimes buffers stored as tensors).

### Size intuition

Rough payload size ≈ sum over tensors of `bytes_per_element × numel`.

Example: one BF16 embedding `[151936, 1024]` is already hundreds of megabytes before layers. After expansion to F32 in
RAM, expect roughly **~2×** that payload for resident weights alone — plus KV pages and activations.

### A fair one-sentence summary

> **A `.safetensors` file is a labeled warehouse of matrices and vectors: a JSON index up front, raw numeric bytes
> afterward, poured into the model’s shelves at load time (and widened to float32 in this port).**

**In the code:** `utils.SafetensorsReader` parses the file; `utils.ModelLoader.loadModel` matches names (including
packed `q_proj`/`k_proj`/`v_proj` → `qkv_proj`) and calls `WeightSlot.load` (chapter 16).

**Further reading:** format overview in the
[Safetensors documentation](https://huggingface.co/docs/safetensors); binary layout notes in the
[safetensors GitHub README](https://github.com/huggingface/safetensors).


---

## 8. Attention: kinds of looking-back, and how they work

Attention is the part people mean when they say the model “pays attention” to something you wrote earlier. It is not a
spotlight of consciousness. It is a **rule for mixing other places in the text into this place**.

There is not only one kind. Models differ in *who may look at whom*, *how many separate glances run in parallel*, and
*how thriftily they store the notebooks*. This chapter walks those kinds in plain language, then says which ones this
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

Without this mask, the model could “cheat” by reading the answer it has not written. Bidirectional models (like the old
BERT-style readers) allow looking left *and* right because their job was understanding a finished sentence, not
continuing it. **Language models that generate text use causal self-attention.** This project does too.

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

Local layers may also use a different RoPE “twist speed” than global ones — a detail of Gemma’s recipe so near and far
looks do not fight each other.

**Further reading:** rotary positions —
[Su et al., *RoFormer* / RoPE](https://arxiv.org/abs/2104.09864).


---

### Kind 6 — Prefill attention vs decode attention (same rule, different workload)

The *rule* (causal, heads, GQA, window) stays the same; the *shape of the work* changes:

| Phase       | What attention does                                                               | Everyday picture                                     |
|-------------|-----------------------------------------------------------------------------------|------------------------------------------------------|
| **Prefill** | Many positions at once; each looks back over the prompt (and fills notebooks)     | Read the whole letter carefully once                 |
| **Decode**  | Usually **one** new position; it looks over past Keys/Values already in notebooks | Write the next sentence using notes you already took |

So “types” of attention in engineering talk sometimes means this **phase**, not a different philosophy. This engine
implements both; chapter 12 explains why notebooks make decode cheaper.

---

### Order still matters (RoPE) for all these kinds

Before Query and Key are compared, a **position-dependent twist** (RoPE) is applied so order counts: “dog bites man” is
not treated like “man bites dog.” Global and local Gemma layers may use different twist bases; the idea is the same —
sequence is baked into the match.

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
| Bidirectional BERT-style                  | No                                                |
| Fancy GPU kernels (flash-attention, etc.) | No — plain educational CPU math                   |

---

**In the code (kinds this port runs):** causal self-attention + GQA/MQA geometry in `layers.Attention`; Gemma sliding
window / global via model config in `Gemma3ForCausalLM`; RoPE in `Norms.RotaryEmbedding`; prefill vs decode branches
inside `Attention.forward` (chapter 16).

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
`Gemma3Attention.forward` (chapter 16).

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

| Sense | Name in this book                    | What it really is                                                                              | Visible as text?        |
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

**In the code (Sense A):** `LLM.generate` / `step` → `Scheduler.schedule` → `ModelRunner.run` → layer
`forward` stacks on `Qwen3ForCausalLM` / `Gemma3ForCausalLM` (chapter 16).

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

#### The format this chat path expects (Qwen-style chat)

For non-Gemma chat, the default system stage directions ask roughly:

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

#### How the program organizes Sense C around the model

```text
  1. Build chat history + system directions
  2. Apply chat template → one big prompt string
       (Gemma: thinking tags not relied on)
       (Qwen-style: enableThinking=true + system text invite <think>…)
  3. Prefill + decode (pure Sense A) until the turn ends
  4. Decode tokens → raw assistant text
  5. AssistantParts.parse splits raw text into:
        thinking  = inside <think>…</think>
        answer    = after the closed tag
        thinkOpen = tag never closed (incomplete)
  6. ChatSession may:
        • stream thinking to one output, answer to another
        • salvage an answer from thinking if the visible part is empty
        • finish the turn for history / UI
```

Important: **the model does not call a `Think()` function.** It emits the characters `<`, `t`, `h`, … as ordinary tokens
if sampling chose them. Parsing happens **after** generation (and incrementally while streaming).

#### Gemma vs Qwen organization in this project

|                                         | Qwen-style chat                                                                                | Gemma chat here                                                                                            |
|-----------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| System directions about `<think>`       | Yes (default `CHAT_SYSTEM`)                                                                    | No (empty system; avoids latching into filler)                                                             |
| `enableThinking` in `applyChatTemplate` | **true** — do **not** pre-insert an empty sealed `<think></think>`; the model may open its own | Flag is **false**, but Gemma uses its own turn template and ignores that im_start empty-think trick anyway |
| Reliable tagged scratchpad              | Encouraged by system text                                                                      | Not relied on                                                                                              |
| Sense A (layers)                        | Same kind of engine                                                                            | Same kind of engine                                                                                        |

So “thinking UI” is a **chat convention** on top of the same model machinery — stronger on Qwen-style templates here.
`enableThinking` is not a second brain switch; it only changes how the **prompt string** is wrapped before Sense A runs.

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
     parse → ChatReply(thinking, answer)
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

## 10. The math, said gently

You can skip this chapter and still understand the story. It exists for readers who want the *shape* of the arithmetic
without a textbook.

### Portraits as lists of numbers

Every token becomes a list of numbers — a **vector**. “Nearness” of meaning is often reflected (imperfectly) in how
these lists relate. Almost everything the model does is: take lists, mix them with stored tables of numbers, produce new
lists.

### A linear layer

“Mix every input trait into every output trait, using a big learned table, optionally add a bias.”  
That single pattern appears everywhere: building Query/Key/Value, mixing attention heads, expanding and shrinking in the
MLP, scoring the vocabulary.

### Embedding

Not mixing — **looking up**. Token number 42 → copy portrait number 42 from the card index.

### Softmax — turning scores into shares of attention (or probability)

If three candidates score 10, 3, and 1, softmax turns that into portions that **add up to 100%**, with the winner
getting most but not necessarily all. A technical trick subtracts the largest score first so the exponentials do not
overflow — same idea, safer calculation.

### Temperature and chance

Divide the scores by a **temperature** before making portions:

- low temperature → the leader wins almost always (stern, repetitive);
- high temperature → lesser candidates get more chance (wilder, more surprising).

This project always keeps a little randomness; pure “always pick the single top score” is disabled on purpose.

### Attention in one line

> Compare me to the past; turn comparisons into portions; take that weighted blend of past contents as my new note.

### The gated MLP in one line

> Split a widened portrait into two halves; activate one half; multiply by the other; shrink back.  
> (SiLU or a GELU-like curve is just the shape of that “activate.”)

None of this *is* understanding in the human sense. It is a procedure that, after training, often **imitates** fluent
continuation well enough to be useful — and misleading.

**In the code:** portraits and tables are `tensor.Tensor`; mix/lookup/activate live in `Ops` (`linear`, `embedding`,
`rmsNorm`, `siluAndMul` / `geluPytorchTanhAndMul`); SIMD helpers in `VectorMath`; softmax / temperature / draw in
`layers.Sampler` (chapter 16).

**Further reading:** [RMSNorm](https://arxiv.org/abs/1910.07467); gated MLP variants such
as [SwiGLU](https://arxiv.org/abs/2002.05202); softmax background
on [Wikipedia](https://en.wikipedia.org/wiki/Softmax_function).


---

## 11. Choosing a word: not always the most obvious one

After scoring the vocabulary, the program must **pick** one token.

Imagine a very large hat of slips of paper. Softmax writes how many copies of each slip go into the hat. Then:

- **Top-k** — keep only the *k* most popular slips; throw the rest away; refill proportions (`0` means “off”).
- **Top-p** — keep the smallest set of popular slips that together cover a large share of the hat (often about 0.9–0.95
  here); discard the long tail of oddities.

Then one slip is drawn. This project’s sampler uses a **Gumbel-max–style** draw over the remaining probabilities (not a
naive left-to-right walk of a cumulative table). Pure greedy decoding (temperature ≈ 0) is **rejected** by
`SamplingParams` — use a small positive temperature instead.

Default helpers (`SamplingDefaults`) use temperature `0.6` and top-p `0.95`; for Gemma chat they also set **top-k =
64**, which matches common Gemma sampling advice.

So the model is not forced to say the single most likely word every time. Controlled chance is why two answers to the
same question can differ — and why “creativity” settings exist in chat products.

You can ask for short or long answers by limiting **how many** tokens may be drawn before stopping. Special **end** /
stop token ids (from the tokenizer and optional `generation_config.json`) mean “the assistant considers this reply
finished.”

**Further reading:** nucleus (top-p) sampling —
[Holtzman et al., *The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751).

**In the code:** `SamplingParams` / `SamplingDefaults.forTokenizer`; draw in `Sampler.forward` (Gumbel-max style); stop
ids and `maxTokens` enforced in `Scheduler.postprocess` (chapter 16).


---

## 12. Why the program keeps a notebook of the past

Attention needs the Key and Value notes for **everything already seen**. Recomputing them from scratch for the whole
prompt on every single new word would be like rereading an entire novel each time you write the next sentence.

So the program keeps a **notebook** (the KV cache): once a position has been processed, its Key and Value are stored and
reused. Loading allocated the blank pages; answering **writes** into them.

### Pages, not one endless scroll

Instead of one giant notebook per conversation, this project (following the vLLM idea) uses **fixed-size pages**
(blocks). A conversation gets a list of page numbers — like a library call slip pointing to several short notebooks on a
shelf.

Why? So memory can be shared and recycled: many conversations, limited desk space, and sometimes **shared early pages**
when two prompts start with the same long prefix (same opening paragraph → reuse the same notes).

### Two phases of work

1. **Prefill** — read the whole prompt once, fill notebooks, produce the first new token.
2. **Decode** — for each later token, read mostly from notebooks, write one new page line, pick one new token.

That is why the first pause can feel longer than each following word: the opening read is heavier than the continuation.

**In the code:** KV pages in `engine.BlockManager` (`allocate`, `hashBlocks`, prefix reuse); prefill/decode scheduling
in `Scheduler`; cache write/read in `Attention.storeKvCache` / decode path (chapter 16).

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

Chat keeps a **history** of turns, wraps them in the model’s expected markers, and may trim old turns if the desk
(context length) is full — like abridgment when the binder only holds so many pages.

**System prompts** (“answer briefly and factually”) are stage directions to the assistant. Some models (notably Gemma in
this project’s default) do **not** have a proper “system” role; stuffing a long lecture into the first user line can
make a small model freeze into polite filler (“Okay, I’m ready”) instead of answering. Short, concrete user text works
better there.

This program’s chat helper can also separate “thinking out loud” tags from the visible answer when the model uses them —
Sense C from the thinking chapter: marked scratchpad versus fair copy.

**Further reading:** [chat templating](https://huggingface.co/docs/transformers/chat_templating) in Transformers.

**In the code:** chat is `ChatSession` (`send`, `streamTo`, history as `ChatMessage`); completion is
`LLM.complete` / raw `generate`; templates via `Tokenizer.applyChatTemplate`; defaults in `ChatPrompts.systemFor`
(chapter 16).

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
  LLM#builder → LLM.Builder#build → LLM.<init>
      → ModelRunner.<init>
          → CausalLMFactory#detect / #create
          → ModelLoader#loadModel          (+ SafetensorsReader per tensor)
          → ModelRunner#allocateKvCache    (empty notebooks on each Attention)
      → Tokenizer#fromPretrained
      → Scheduler.<init>                   (owns BlockManager)
      → LLM#warmup                         (optional tiny generate)

ONE TURN
  LLM#chat(maxTokens) → ChatSession#open → ChatSession#send("What is 2+2?")
      → ChatMessage#user / history.add
      → ChatMessages#truncateHistory
      → ChatSession#generateTurn
          → Tokenizer#applyChatTemplate
          → LLM#generate
              → LLM#addRequest → Tokenizer#encode → Scheduler#add (new Sequence)
              → loop while !LLM#isFinished:
                    LLM#step
                      → Scheduler#schedule          (BlockManager#allocate / #mayAppend)
                      → ModelRunner#run             (via ModelRunner#call("run", …))
                          → preparePrefill | prepareDecode
                          → CausalLM#forward        (Qwen3ForCausalLM or Gemma3ForCausalLM)
                          → CausalLM#computeLogits
                          → Sampler#forward
                      → Scheduler#postprocess       (append token / finish on stop)
                    onToken → AssistantParts#parse(Tokenizer#decode(partial))
          → AssistantParts#parse(Tokenizer#decode(final ids))
      → ChatSession#finishTurn
          → maybe AssistantParts#salvageFromThinking
          → ChatMessage#assistant(answer only) into history
```

That is the whole “2+2” turn in library terms. The subsections below zoom each box.

### You ask

A few lines of Java (or the example app) open the model folder and say, in effect: *chat with me; keep answers short.*

```java
try(LLM llm = LLM.builder(modelDir).maxModelLen(2048).build()){
ChatReply reply = llm.chat(256).send("What is 2+2?");
// reply.thinking() / reply.answer()
}
```

| Step            | Call                                 | Role                                              |
|-----------------|--------------------------------------|---------------------------------------------------|
| Start builder   | `LLM#builder`                        | Fluent open of a model directory                  |
| Finish open     | `LLM.Builder#build` → `LLM.<init>`   | Load + wire engine                                |
| Start session   | `LLM#chat(int)` → `ChatSession#open` | History + `SamplingDefaults#forTokenizer`         |
| Ask             | `ChatSession#send`                   | One user turn through template → generate → parse |
| CLI alternative | `Example#main`                       | Same ideas with `streamTo` on stderr/stdout       |

### Loading (once)

- Blueprint read → empty rooms built to the right sizes (`hidden_size`, layers, heads, …).
- Each weight name in the crates poured onto the matching shelf (Query/Key/Value packs merged where needed).
- Dictionary opened at the door.
- Blank Key/Value notebooks laid out for this session’s text.
- Learned shelves will not change; notebooks will.

**Attention’s role at load time:** none yet — only empty notebooks waiting.  
**Thinking’s role at load time:** none — no Sense A until a prompt runs.

| Step            | Call                                                                                 | Role                                                      |
|-----------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------|
| Blueprint       | `Config.HfConfig#load` (via `Config` / builder)                                      | Read `config.json`                                        |
| Empty graph     | `CausalLMFactory#detect` → `#create`                                                 | `Qwen3ForCausalLM` or `Gemma3ForCausalLM`                 |
| Pour weights    | `ModelLoader#loadModel` → `SafetensorsReader#getTensor` → `CausalLM.WeightSlot#load` | Fill shelves                                              |
| Dictionary      | `Tokenizer#fromPretrained`                                                           | `tokenizer.json` + chat template / stop ids               |
| Blank notebooks | `ModelRunner#allocateKvCache`                                                        | Attach `kCache`/`vCache` on each `Attention`              |
| Waiting room    | `Scheduler.<init>` → `BlockManager.<init>`                                           | Page pool for later allocate                              |
| Optional        | `LLM#warmup`                                                                         | Tiny generate so first real answer is not also cold-start |

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
| Stage directions | `Tokenizer#applyChatTemplate(…, enableThinking)`                        | Role markers; thinking invitation when not Gemma |
| To ids           | inside `LLM#generate` → `LLM#addRequest(String,…)` → `Tokenizer#encode` | Prompt string → token ids → `Sequence`           |

`ChatSession#generateTurn` is the private brick that calls `applyChatTemplate` then `LLM#generate`.

### Prefill — Sense A on the whole prompt (attention’s first big job)

Portraits for every prompt token walk through **every** reading room.

**Where attention works in prefill**

At each layer, for each position in the prompt (for example the token `2`, the token `+`, the later `2`, the `?`):

1. Build Query / Key / Value from that position’s dossier.
2. Compare this Query with Keys of **earlier** positions only (causal — no peeking at the future).
3. Mix Values from the strong matches into an updated dossier.
4. **Write** this position’s Key and Value into the KV notebooks for later decode.

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
  └─ LLM#step                         // first tick is usually prefill
       ├─ Scheduler#schedule          // BlockManager#canAllocate / #allocate; prefill=true
       ├─ ModelRunner#run(seqs, true)
       │    ├─ ModelRunner#preparePrefill   // ids, positions, Context slot maps / block tables
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
| Pack tensors | `ModelRunner#preparePrefill`                   | Build input ids / positions / `Context` |
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
  → RotaryEmbedding#forward     // position twist
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
LLM#step
  ├─ Scheduler#schedule              // prefill=false; BlockManager#mayAppend
  ├─ ModelRunner#run(seqs, false)
  │    ├─ ModelRunner#prepareDecode  // mostly the newest token + block tables
  │    ├─ CausalLM#forward
  │    │    └─ Attention#forward → #storeKvCache → #decode → #attendRange
  │    ├─ CausalLM#computeLogits
  │    └─ Sampler#forward
  └─ Scheduler#postprocess           // stop if eos / stop ids / maxTokens
```

`LLM#generate` keeps calling `LLM#step` until `Scheduler#isFinished` (via `LLM#isFinished`). Each appended token may
fire the `onToken` callback that `ChatSession#generateTurn` registered.

**Thinking’s three roles in this same decode loop**

| Sense                     | What happens on “What is 2+2?”                                                                                                   | Library home                                                                |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **A — silent**            | Every decode step is a full layer-walk (attention + MLP + sample). Always on, never shown as text.                               | `ModelRunner#run` → model `#forward` / `#computeLogits` → `Sampler#forward` |
| **B — written reasoning** | The model may emit ordinary words like “add” or “four” as intermediate text. Those words join the past.                          | Same loop; tokens land in `Sequence` via `Scheduler#postprocess`            |
| **C — tagged scratchpad** | On Qwen-style chat it may emit `<think> … </think>` then the answer. Same Sense A underneath; markers let the UI split channels. | `AssistantParts#parse` on decode/`finishTurn`                               |

Example timeline (one possible Qwen-style path — not guaranteed wording):

```text
  decode → "<think>"           Sense A; attention sees only the prompt so far
  decode → "User" "asks" …     Sense A; attention can also see prior think tokens
  decode → "2+2" "→" "4"       notes cite the question; attention binds them
  decode → "</think>"
  decode → "4"                 Sense A; attention may weigh notes + original "2" "+" "2"
  decode → end-of-turn
```

If there is **no** scratchpad (typical Gemma path here), Sense A still runs the same way; you only see the final answer
tokens. There is no missing “thinker” — only missing **visible** Sense B/C text. Gemma disables the thinking invitation
in `ChatSession#generateTurn` (`enableThinking = !tokenizer.isGemmaChat()`).

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
| Final split        | `AssistantParts#parse` → `ChatReply#from`                                        | `thinking` + `answer` + `thinkOpen`  |
| Recover            | `AssistantParts#salvageFromThinking`                                             | If answer blank but notes exist      |
| Commit history     | `ChatSession#finishTurn` → `ChatMessage#assistant(answer)`                       | **Answer only** stored for next turn |
| Close stream       | `StreamPrinter#closeTurn`                                                        | End of CLI printing for this reply   |

### One picture of the whole turn (story + calls)

```text
  LLM.<init> / ModelRunner.<init>     LOAD weights + empty KV notebooks
           │
           ▼
  Tokenizer#applyChatTemplate
  + Tokenizer#encode                  TEMPLATE + TOKENIZE  "What is 2+2?"
           │
           ▼
  LLM#step (prefill)                  PREFILL (Sense A)
  Scheduler#schedule
  ModelRunner#run → CausalLM#forward
  Attention#storeKvCache / prefill…      attention: every prompt place looks back; notebooks fill
  Sampler#forward                        thinking: silent only
           │
           ▼
  LLM#step (decode) × N               DECODE LOOP (Sense A each step)
  Attention#decode / #attendRange        attention: new token queries notebooks
  Sampler#forward                        thinking: optional <think> tokens (B/C)
  Scheduler#postprocess
           │
           ▼
  stop ids / maxTokens                SAMPLE "4" … STOP
           │
           ▼
  AssistantParts#parse
  ChatSession#finishTurn              PARSE / DISPLAY  thinking + answer "4"
```

### What this walk-through should leave you with

You should be able to point at **both** the story and the library:

| Story piece            | Primary `Class#method` homes                                                  |
|------------------------|-------------------------------------------------------------------------------|
| Open model             | `LLM#builder` / `LLM.<init>` → `ModelRunner.<init>` → `ModelLoader#loadModel` |
| Chat ask               | `ChatSession#send` → `#generateTurn` → `#finishTurn`                          |
| Template / ids         | `Tokenizer#applyChatTemplate` / `#encode` / `#decode`                         |
| Engine loop            | `LLM#generate` → `#step` → `Scheduler#schedule` / `#postprocess`              |
| One forward+sample     | `ModelRunner#run` → `CausalLM#forward` / `#computeLogits` → `Sampler#forward` |
| Attention + notebooks  | `Attention#forward` / `#storeKvCache` / `#decode` / `#attendRange`            |
| Visible thinking split | `AssistantParts#parse` (history keeps answer via `finishTurn`)                |

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
`src/main/java/com/igormaznitsa/nanollvm/`.

Every earlier chapter’s **In the code** note points here. Use this chapter when you want methods, samples, and paths in
one place.

You do not need to read every file. Use the tables to jump, then skim the named methods.

### Package layout (folders ↔ ideas)

| Folder / type                                                                          | Role in the story                                  |
|----------------------------------------------------------------------------------------|----------------------------------------------------|
| `LLM`, `LLM.Builder`, `EngineIo`, `SamplingParams`, `SamplingDefaults`                 | Front door, options, quiet vs CLI I/O              |
| `chat/` — `ChatSession`, `ChatMessage`, `ChatReply`, `AssistantParts`, `StreamPrinter` | Dialog, history, `<think>` split, streaming        |
| `tokenizer/Tokenizer`                                                                  | `tokenizer.json` → encode / decode / chat template |
| `Config`, `Config.HfConfig`                                                            | `config.json` blueprint                            |
| `utils/ModelLoader`, `utils/SafetensorsReader`, `utils/BundledModels`                  | Load weights; find default model dir               |
| `engine/Scheduler`, `Sequence`, `BlockManager`, `ModelRunner`                          | Prefill/decode loop, pages, one forward+sample     |
| `models/CausalLM`, `CausalLMFactory`, `Qwen3ForCausalLM`, `Gemma3ForCausalLM`          | Architecture graph                                 |
| `layers/Attention`, `Sampler`, `Linear`, `Norms`, …                                    | Attention, sampling, projections, RMSNorm/RoPE     |
| `tensor/Tensor`, `Ops`, `VectorMath`                                                   | Arrays and kernels                                 |
| `prompts/ChatPrompts`                                                                  | Default system text (Qwen vs empty Gemma)          |
| `Example`, `Bench`                                                                     | Runnable demos                                     |

### Concept → class → methods

| Story idea           | Primary type                           | Methods / entry points to open                                                    |
|----------------------|----------------------------------------|-----------------------------------------------------------------------------------|
| Open a model         | `LLM`, `LLM.Builder`                   | `LLM.builder(path)`, `.systemPrompt(…)`, `.withSystemIo()`, `.build()`            |
| Chat turn            | `ChatSession`                          | `llm.chat(maxTokens)`, `.send(user)`, `.streamTo(…)`, `.clear()`                  |
| One-shot / raw text  | `LLM`                                  | `chatOnce(…)`, `complete(…)`, `generate(…)`                                       |
| Cancel / timeout     | `LLM`                                  | `cancel()`; `generate(…, timeout, onToken)`                                       |
| Tokenize             | `Tokenizer`                            | `fromPretrained(dir)`, `encode`, `decode`, `applyChatTemplate(…, enableThinking)` |
| Blueprint            | `Config.HfConfig`                      | `HfConfig.load(config.json)`; `CausalLMFactory.detect/create`                     |
| Pour weights         | `ModelLoader`, `SafetensorsReader`     | `ModelLoader.loadModel`; `getTensor(name)`; `CausalLM.WeightSlot.load`            |
| One engine tick      | `LLM.step`, `Scheduler`, `ModelRunner` | `schedule` → `ModelRunner.run` → `postprocess`                                    |
| Forward + sample     | `ModelRunner`, `CausalLM`, `Sampler`   | `model.forward` → `computeLogits` → `sampler.forward`                             |
| Attention + KV write | `Attention`, `utils.Context`           | `Attention.forward`; `storeKvCache`; prefill/decode helpers                       |
| Pages / prefix reuse | `BlockManager`                         | `canAllocate`, `allocate`, `hashBlocks`, `mayAppend`                              |
| Split thinking UI    | `AssistantParts`, `ChatReply`          | `AssistantParts.parse`, `salvageFromThinking`                                     |
| Math bricks          | `Ops`                                  | `linear`, `embedding`, `rmsNorm`, `siluAndMul` / `geluPytorchTanhAndMul`          |

### Sample A — library use (what most apps call)

```java
import com.igormaznitsa.nanollvm.LLM;
import com.igormaznitsa.nanollvm.utils.BundledModels;

try(LLM llm = LLM.builder(BundledModels.resolveDefault())
        .maxModelLen(2048)
        .systemPrompt("Answer briefly and factually.")  // Qwen-style; Gemma often empty
        .build()){

// Multi-turn (history + template + optional <think> parse)
String reply = llm.chat(256).send("What is 2+2?").answer();

// One-shot chat
String once = llm.chatOnce("What is 2+2?");

// Raw continuation (no chat template)
String raw = llm.complete("The capital of France is");
}
```

Interactive CLI wiring lives in `Example.main`: `LLM.builder(…).withSystemIo().build()`, then
`llm.chat(…).streamTo(System.err, System.out, color)` and `chat.send(user)`.

### Sample B — one generate tick (Sense A loop)

`LLM.generate` repeatedly calls `step()` until the scheduler is idle. Each `step()` is:

```text
Scheduler.schedule()          // pick prefill or decode batch
    → ModelRunner.run(seqs, prefill)
        preparePrefill / prepareDecode  (+ Context.set slot maps, block tables)
        CausalLM.forward(inputIds, positions)
        CausalLM.computeLogits(hidden)
        Sampler.forward(logits, temperature, topK, topP)
    → Scheduler.postprocess(…)   // append token, finish on stop / maxTokens
```

Compressed from `LLM.step` / `ModelRunner.run`:

```java
// conceptual — names match the real methods
ScheduleResult scheduled = scheduler.schedule();
List<Integer> tokenIds = modelRunner.run(scheduled.sequences(), scheduled.prefill());
scheduler.

postprocess(scheduled.sequences(),tokenIds,scheduled.

prefill(),appendedOut);
```

Inside `ModelRunner.run`:

```java
Tensor logits = model.computeLogits(model.forward(inputIds, positions));
int[] tokenIds = sampler.forward(logits, temperatures, topKs, topPs);
```

### Sample C — attention stores and reads the notebook

`layers.Attention.forward(q, k, v)`:

1. If slot mapping is set, `storeKvCache` copies K/V into paged `kCache` / `vCache`.
2. Prefill → `prefillWithCache` or `prefillDense`.
3. Decode → `decode` (gather past K/V via `blockTables`, then `attendRange`).

That is the code behind “write notebooks at prefill; reread them at decode.”

### Sample D — loading shelves

```text
Config loads HfConfig from config.json
CausalLMFactory.create(hf) builds empty Qwen3ForCausalLM or Gemma3ForCausalLM
ModelLoader.loadModel(model, dir):
    for each *.safetensors name
        resolve packed q_proj/k_proj/… → qkv_proj (WeightSlot.qkv / merged)
        SafetensorsReader.getTensor(name)  // BF16/F16 → float[]
        model.getParameter(paramName).load(tensor, shardId)
ModelRunner.allocateKvCache() attaches empty K/V pages to each Attention
Tokenizer.fromPretrained(dir) loads tokenizer.json (+ tokenizer_config chat_template)
```

### Sample E — chat thinking path (Sense C)

```text
ChatSession.send(user)
  history.add(user message)
  truncateHistory(…)
  Tokenizer.applyChatTemplate(history, addGenerationPrompt=true, enableThinking=…)
  LLM.generate(prompt, sampling, onToken → AssistantParts.parse(partial decode))
  AssistantParts.parse(full decode) → ChatReply(thinking, answer, thinkOpen)
  finishTurn: maybe salvageFromThinking; history.add(assistant(answer only))
```

Key types: `ChatSession.send` / `generateTurn` / `finishTurn`, `AssistantParts.parse`, `ChatPrompts.systemFor`,
`SamplingDefaults.forTokenizer` (Gemma top-k 64).

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

### How to read the tree on disk

```text
src/main/java/com/igormaznitsa/nanollvm/
  LLM.java                 ← start here for API
  Example.java             ← start here for CLI chat
  chat/ChatSession.java    ← dialog + thinking split
  engine/ModelRunner.java  ← forward + sample
  engine/Scheduler.java    ← prefill/decode batches
  layers/Attention.java    ← QKV cache + attendRange
  models/Qwen3ForCausalLM.java / Gemma3ForCausalLM.java
  tokenizer/Tokenizer.java
  utils/ModelLoader.java / SafetensorsReader.java
```

### Threading reminder (API contract)

One `LLM` must not run concurrent `generate` / chat calls. `LLM.cancel()` is safe from another thread and clears
in-flight work via the scheduler.

---

## 17. Word list

Short glossary. For the Java home of each idea, prefer the **In the code** notes in earlier chapters and the map in
**chapter 16**.

| Term you may meet     | Plain meaning                                                                   |
|-----------------------|---------------------------------------------------------------------------------|
| Model                 | The finished “book” of learned numbers plus its dictionary and blueprint        |
| Loading               | Reading blueprint + dictionary + weights into memory and wiring them            |
| `config.json`         | Blueprint of sizes and recipe (not the learned weights)                         |
| `tokenizer.json`      | Vocab, BPE merges, and text pipeline (string ↔ token ids)                       |
| `.safetensors`        | Catalogued raw weight tensors (matrices/vectors) on disk                        |
| BPE                   | Byte-Pair Encoding: merge frequent pieces using an ordered merge list           |
| `data_offsets`        | Byte range of one tensor inside a safetensors payload                           |
| `hidden_size`         | Length of each token’s internal dossier (main working width through all layers) |
| `intermediate_size`   | Temporary wider width inside each layer’s MLP expand→shrink step                |
| `num_hidden_layers`   | How many stacked attention+MLP rooms                                            |
| GQA heads fields      | `num_attention_heads` vs `num_key_value_heads` (sharing of KV notebooks)        |
| Inference             | Using the book to produce text (not training it)                                |
| Token                 | A scrap of text with a number in the dictionary                                 |
| Vocabulary            | All scraps the model is allowed to emit                                         |
| Logits                | Raw preference scores for each vocabulary scrap, before turning into chances    |
| Softmax               | Turn raw scores into portions that add to 100%                                  |
| Attention             | Weighted reread of allowed places (Query matches Keys, mixes Values)            |
| Self-attention        | Looking within the same text (not at a second document)                         |
| Causal                | May only look at the past and present, not the future                           |
| Multi-head (MHA)      | Several parallel glances; each may have its own KV notebooks                    |
| GQA                   | Grouped-query: several Query heads share one Key/Value group                    |
| MQA                   | Multi-query: all Query heads share a single Key/Value pair                      |
| Sliding window        | May look back only a fixed recent stretch, not the whole past                   |
| Global attention      | May look back over the whole allowed past                                       |
| Query / Key / Value   | Search / label / content notes used by attention                                |
| Inner work (Sense A)  | Invisible stack of attention + rewrites for each next token                     |
| Chain of thought (B)  | Reasoning written as ordinary tokens in the reply                               |
| Tagged scratchpad (C) | Written reasoning inside `<think>…</think>` for UI splitting                    |
| ChatReply             | Parsed pair of thinking text + visible answer after a turn                      |
| KV cache / notebook   | Stored Keys and Values, reused while continuing                                 |
| Prefill               | First heavy read of the prompt                                                  |
| Decode                | Step-by-step production of later tokens                                         |
| Sampling              | Drawing the next token according to chances and filters                         |
| Temperature           | How strongly to favor the leading candidate                                     |
| Context length        | How much past text fits on the desk at once                                     |
| Weights               | The learned numbers on the shelves                                              |

---

## 18. Honest limits

This project is a **teaching instrument**, not a production cloud service.

- It runs on the ordinary processor and keeps numbers in a simple, memory-hungry form.
- It is slower than GPU systems you meet in products.
- Small models hallucinate, waffle, and latch onto polite filler — especially if prompts are vague.
- “Understanding,” “knowing,” “thinking,” and “meaning” here are **metaphors** for statistical continuation and inner
  arithmetic. A humanities reader is right to keep that distinction sharp.

If this book did its job, you can now explain to another non-specialist:

> Loading unpacks a fixed library of trained numbers. Attention is how each moment rereads allowed parts of the current
> page — causally, often with shared notebooks (GQA), sometimes through a sliding window. Thinking is organized as a
> loop: silent layer-walks for every next token, and optionally written notes (even tagged ones) that later attention
> can reuse. Then the program repeatedly draws the next scrap of text until the reply ends.

That is enough to understand what this Java program is doing — and what it is not.

**In the code (limits mirrored by design):** CPU kernels in `Ops` / `VectorMath`; no GPU path; one `LLM` is not safe for
concurrent `generate` — use `cancel()` from another thread only to abort (chapter 16).

---

## 19. External reading index

A single list of the links woven into earlier chapters. Prefer the in-chapter notes for context; use this as a bookmark
page.

Implementation homes stay in the **In the code** notes and **chapter 16**; this index is papers and format docs only.

| Topic                      | Link                                                                                                                               |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Transformer / attention    | [Vaswani et al. (arXiv)](https://arxiv.org/abs/1706.03762)                                                                         |
| Annotated Transformer      | [Harvard NLP notebook](https://nlp.seas.harvard.edu/annotated-transformer/)                                                        |
| BPE subwords               | [Sennrich et al. (arXiv)](https://arxiv.org/abs/1508.07909)                                                                        |
| Byte-level BPE / GPT-2     | [OpenAI GPT-2 report (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| HF Tokenizers              | [Documentation](https://huggingface.co/docs/tokenizers/index)                                                                      |
| Chat templates             | [Transformers guide](https://huggingface.co/docs/transformers/chat_templating)                                                     |
| Model `config` class       | [PretrainedConfig](https://huggingface.co/docs/transformers/main/en/main_classes/configuration)                                    |
| Safetensors                | [HF docs](https://huggingface.co/docs/safetensors) · [GitHub format notes](https://github.com/huggingface/safetensors)             |
| RoPE                       | [Su et al. / RoFormer (arXiv)](https://arxiv.org/abs/2104.09864)                                                                   |
| GQA                        | [Ainslie et al. (arXiv)](https://arxiv.org/abs/2305.13245)                                                                         |
| Multi-query attention      | [Shazeer (arXiv)](https://arxiv.org/abs/1911.02150)                                                                                |
| PagedAttention / vLLM      | [Kwon et al. (arXiv)](https://arxiv.org/abs/2309.06180) · [vLLM GitHub](https://github.com/vllm-project/vllm)                      |
| nano-vllm (upstream idea)  | [GitHub](https://github.com/GeeeekExplorer/nano-vllm)                                                                              |
| Chain-of-thought prompting | [Wei et al. (arXiv)](https://arxiv.org/abs/2201.11903)                                                                             |
| RMSNorm                    | [Zhang & Sennrich (arXiv)](https://arxiv.org/abs/1910.07467)                                                                       |
| SwiGLU                     | [Shazeer (arXiv)](https://arxiv.org/abs/2002.05202)                                                                                |
| Nucleus (top-p) sampling   | [Holtzman et al. (arXiv)](https://arxiv.org/abs/1904.09751)                                                                        |
| Softmax                    | [Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)                                                                        |

---
