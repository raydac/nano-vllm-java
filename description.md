# How a Language Model Works

### A plain-language guide to this Java project (Nano-vLLM)

This little book is written for a curious reader who is comfortable with ideas, stories, and careful argument — **not**
for someone who already thinks in code or equations.

You do not need to know Java, matrices, or “AI engineering.”  
If a term is unavoidable, it is explained the first time in everyday language.

The project this book describes is a small program that can **load a ready-made language model** and use it to continue
text or hold a short conversation — on an ordinary computer, without special graphics hardware.

---

## Table of contents

1. [What this is really about](#1-what-this-is-really-about)
2. [The one trick: guessing the next piece of text](#2-the-one-trick-guessing-the-next-piece-of-text)
3. [Cutting language into pieces the machine can count](#3-cutting-language-into-pieces-the-machine-can-count)
4. [Loading a model: opening the library box](#4-loading-a-model-opening-the-library-box)
5. [Attention: how the model looks back while it writes](#5-attention-how-the-model-looks-back-while-it-writes)
6. [What “thinking” means inside a model](#6-what-thinking-means-inside-a-model)
7. [The math, said gently](#7-the-math-said-gently)
8. [Choosing a word: not always the most obvious one](#8-choosing-a-word-not-always-the-most-obvious-one)
9. [Why the program keeps a notebook of the past](#9-why-the-program-keeps-a-notebook-of-the-past)
10. [Serving several conversations without chaos](#10-serving-several-conversations-without-chaos)
11. [Chat versus finishing a sentence](#11-chat-versus-finishing-a-sentence)
12. [A full walk-through: “What is 2+2?”](#12-a-full-walk-through-what-is-22)
13. [If you later open the code](#13-if-you-later-open-the-code)
14. [Word list](#14-word-list)
15. [Honest limits](#15-honest-limits)

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

Two families of models are supported here (different “editions” of the book, same kind of reading process): **Qwen3**
and **Gemma3**. You usually need not care which; the program detects which files you pointed it at.

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
how large the furniture must be.

**2. The dictionary (`tokenizer.json` and friends)**  
How human text becomes token numbers and back, plus special markers for “user” and “assistant.”  
This is the spelling system and the stage-direction language — still not the “knowledge.”

**3. The learned numbers (`*.safetensors`)**  
The big cargo. Millions or billions of numbers shaped by training. These are what make one model sound different from
another.

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

### Why the first load feels slow and heavy

- The crates are large (often gigabytes).
- Every number may be expanded to a fuller format in RAM.
- The program must touch essentially the whole model once before it can answer.

That cost is paid **at open time**, not on every word (the notebooks exist precisely so later words are cheaper).

---

## 5. Attention: how the model looks back while it writes

Attention is the part people mean when they say the model “pays attention” to something you wrote earlier. It is not a
spotlight of consciousness. It is a **rule for mixing the past into the present**.

### A small story

Suppose the text so far is:

> Mary gave Susan a book because she

When the model is about to continue after *she*, a human reader wonders: Mary or Susan? The model does not “wonder” in
our sense. At the position of *she*, it builds a **Query** (“what am I trying to link?”) and compares it with **Keys**
attached to earlier words (*Mary*, *Susan*, *book*, …). Whichever earlier places match strongly contribute more of their
**Values** into the new portrait for *she*.

If training taught useful patterns, *Mary* or *Susan* may light up appropriately. If not, the model may guess like a
distracted reader.

So attention is:

> For this place in the text, **how much should each earlier place influence me?** Then take that blended influence as
> part of my updated meaning.

### The three notes (Query, Key, Value)

For every token position, from the same portrait, the model builds three derived notes:

| Note      | Everyday question                                              |
|-----------|----------------------------------------------------------------|
| **Query** | What am I looking for *from here*?                             |
| **Key**   | How should *this* place advertise itself to searchers?         |
| **Value** | If someone chooses me, what content should they actually take? |

Comparison is between **this Query** and **other Keys**. The Values of the winners are mixed into the result.

A library metaphor: Query is your catalog search; Key is the card on the spine; Value is the book you pull off the shelf
when the card matches.

### Only the past (the causal rule)

When writing forward, a position may look at itself and what came **before**, never at the future. That rule is what
“causal” means here. Without it, the model could cheat by reading the answer it has not yet written.

```text
  positions:  1    2    3    4    5
              The  cat  sat  on   the
                                ↑
                         position 5 may look at 1…5
                         not at words that do not exist yet
```

### Many glances at once (heads)

Instead of one comparison, the model runs **several attentions in parallel** (“heads”). One head may specialize in
nearby grammar; another in names; another in punctuation patterns — not because someone labeled them that way, but
because training pushed different heads toward different habits.

Afterwards the glances are **merged** back into one portrait.

Some models use a thrifty variant: many Queries share fewer Key/Value notebooks (grouped-query attention). Same idea,
less duplicate memory.

### Order matters (a twist called RoPE)

Without position, “dog bites man” and “man bites dog” could look too similar. Before comparing Query and Key, the model
applies a **position-dependent twist** (RoPE) so that *where* something stood in the sentence affects the match.
Sequence is not a footnote; it is baked into the comparison.

### Attention is not yet “the answer”

Attention updates **inner portraits** of the current text. It does not directly print words. Words come later, when the
final portrait is scored against the vocabulary and one token is drawn. Attention is the **inward rereading**; speaking
aloud is a later stage.

### After attention: the private rewrite

Each reading room also contains a large feed-forward block (MLP). Rough picture:

1. Attention — *consult the current passage.*
2. MLP — *rewrite this position using trained habits that are not a simple glance at neighbors.*

Both happen in **every** room, stacked many times. Early rooms tend to handle local pattern; later rooms work with
richer mixtures — again as a tendency, not a guaranteed map of “where facts live.”

---

## 6. What “thinking” means inside a model

People say models “think.” That word covers at least **three different things**. Keeping them apart prevents
disappointment and mysticism.

### Sense A — Silent inner work (always on)

Every time the model produces even one token, the text’s portraits walk through **all** the reading rooms: many rounds
of attention + rewrite. That whole journey *is* the computation. There is no separate little person inside.

Call this **inner work**. It is mandatory, invisible as text, and happens for “Hi” as much as for a proof in geometry.
You do not see it; you only see the tokens that come out afterward.

```text
  your prompt (as tokens)
        │
        │  inner work: room 1 → room 2 → … → room N
        │  (attention looks back; MLP rewrites; again and again)
        ▼
  scores for every possible next scrap
        │
        ▼
  pick one scrap → show it (or keep it for the next round)
```

### Sense B — Thinking *as writing* (chain of thought)

Sometimes the model is trained or prompted to **write intermediate reasoning into the reply** (“First I note that…
Therefore…”). That is not a second brain. It is still next-token guessing — but the guessed tokens are *about
reasoning*, and those tokens then sit in the passage so later attention can look back at them.

In other words: **writing a draft of an argument can help the same machine continue better**, because attention can
reread that draft. The “thinking” became part of the text on the desk.

This can help on multi-step questions. It can also produce fluent nonsense that *looks* careful. The form of reasoning
is easier to imitate than the discipline of truth.

### Sense C — Tagged scratchpad in chat (what this project may show)

Some chat setups ask the assistant to put private notes between markers such as `<think> … </think>` and then give the
user-facing answer. This project’s chat helper can **split** those: show or hide the scratchpad, keep the clean answer.

Again: the scratchpad is **more generated text**, not a window into Sense A. Sense A already ran to create each token of
the scratchpad.

| Sense                | What it is                              | Can you read it?         |
|----------------------|-----------------------------------------|--------------------------|
| A. Inner work        | Stacked attention + rewrites on numbers | No — only effects        |
| B. Written reasoning | Ordinary tokens that narrate steps      | Yes — it’s in the reply  |
| C. Tagged scratchpad | Specially marked written reasoning      | Yes — if the UI shows it |

### What thinking is *not*

- Not a search of the internet (unless some other tool is added — this project does not).
- Not a guarantee of truth or self-knowledge.
- Not “the model paused to reflect” as a human does; it predicted tokens, including tokens that *sound like* reflection.
- Not stored as a diary inside the weight shelves; shelves are fixed at load time.

### A fair humanities summary

> **Loading** gives you a fixed library of trained habits.  
> **Attention** is how each new moment of that library rereads the current page.  
> **Thinking**, in the everyday chat sense, is either invisible arithmetic (Sense A) or more language about reasoning
> (Senses B and C) that attention can later use — never a ghost in the machine.

---

## 7. The math, said gently

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

---

## 8. Choosing a word: not always the most obvious one

After scoring the vocabulary, the program must **pick** one token.

Imagine a very large hat of slips of paper. Softmax writes how many copies of each slip go into the hat. Then:

- **Top-k** — keep only the *k* most popular slips; throw the rest away; refill proportions.
- **Top-p** — keep the smallest set of popular slips that together cover, say, 90% of the hat; discard the long tail of
  oddities.

Then one slip is drawn.

So the model is not forced to say the single most likely word every time. Controlled chance is why two answers to the
same question can differ — and why “creativity” settings exist in chat products.

You can ask for short or long answers by limiting **how many** tokens may be drawn before stopping. Special **end**
tokens mean “the assistant considers this reply finished.”

---

## 9. Why the program keeps a notebook of the past

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

---

## 10. Serving several conversations without chaos

A naïve program would finish Alice’s entire answer before Bob gets a turn. This engine instead keeps a **waiting room**
and a **work floor**:

- new prompts wait until there is room;
- several conversations can advance a little on the same “tick”;
- if memory pages run short, one conversation may be politely paused and restarted later.

You can think of a restaurant kitchen preparing several dishes in interleaved steps, not cooking one meal completely
before lighting the next stove. That idea is called **continuous batching**. For a single chat on your laptop you may
barely notice it; it matters when many requests share one model.

---

## 11. Chat versus finishing a sentence

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

---

## 12. A full walk-through: “What is 2+2?”

### You ask

A few lines of Java (or the example app) open the model folder and say, in effect: *chat with me; keep answers short.*

### Loading (once)

- Blueprint read → empty rooms built to the right sizes.
- Each weight name in the crates poured onto the matching shelf (Query/Key/Value packs merged where needed).
- Dictionary opened at the door.
- Blank Key/Value notebooks laid out for this session’s text.
- Learned shelves will not change; notebooks will.

### Your sentence becomes numbers

The chat template wraps your question with “user” / “assistant” markers. The tokenizer turns that into a list of token
numbers.

### Prefill (inner work on the whole prompt)

Portraits walk through every reading room. In each room, attention looks back through the prompt; the MLP rewrites. Keys
and Values are written into the notebooks. At the end, vocabulary scores appear; one first answer token is drawn.

### Decode (inner work, one scrap at a time)

Each further step:

1. take only the newest token through the rooms;
2. attention looks back using the notebooks (and writes a new line into them);
3. score the vocabulary;
4. draw one more token;
5. stop on an end marker or a length limit.

If the reply contains a tagged scratchpad, that text was also produced this way — token by token — not retrieved from a
hidden diary.

### Back to language

The answer tokens are looked up in the dictionary and shown to you as text: `4` or `The answer is 4.`

Nothing mystical happened — unpacking a crate, then a long chain of rereads, rewrites, and draws among likely scraps of
text. The *impression* of understanding is an effect of that chain, shaped by training you do not see.

---

## 13. If you later open the code

You still do not need to. But if curiosity leads you there, these are the “rooms” of the program, named for humans:

| Human idea                   | Place in the project                    |
|------------------------------|-----------------------------------------|
| Front door                   | `LLM` — open a model, chat, or complete |
| Conversation manners         | `chat.ChatSession`                      |
| Dictionary                   | `tokenizer.Tokenizer`                   |
| Blueprint                    | `Config` / Hugging Face config fields   |
| Filling the shelves          | `ModelLoader`, `SafetensorsReader`      |
| Kitchen tick (schedule work) | `Scheduler`, `Sequence`, `BlockManager` |
| Run the reading rooms        | `ModelRunner` + `Qwen3…` / `Gemma3…`    |
| The glance backward          | `Attention`                             |
| Draw from the hat            | `Sampler`                               |
| Arithmetic worksheets        | `tensor.Ops`, `Tensor`                  |

One `LLM` instance is like **one desk**: do not ask it to write two full answers at once from different threads. You may
ask it to **stop** from another thread if a reply is taking too long.

---

## 14. Word list

| Term you may meet   | Plain meaning                                                                |
|---------------------|------------------------------------------------------------------------------|
| Model               | The finished “book” of learned numbers plus its dictionary and blueprint     |
| Loading             | Reading blueprint + dictionary + weights into memory and wiring them         |
| Inference           | Using the book to produce text (not training it)                             |
| Token               | A scrap of text with a number in the dictionary                              |
| Vocabulary          | All scraps the model is allowed to emit                                      |
| Logits              | Raw preference scores for each vocabulary scrap, before turning into chances |
| Softmax             | Turn raw scores into portions that add to 100%                               |
| Attention           | Weighted reread of the text so far (Query matches Keys, mixes Values)        |
| Query / Key / Value | Search / label / content notes used by attention                             |
| Inner work          | Invisible stack of attention + rewrites for each step                        |
| Chain of thought    | Reasoning written as ordinary tokens in the reply                            |
| KV cache / notebook | Stored Keys and Values, reused while continuing                              |
| Prefill             | First heavy read of the prompt                                               |
| Decode              | Step-by-step production of later tokens                                      |
| Sampling            | Drawing the next token according to chances and filters                      |
| Temperature         | How strongly to favor the leading candidate                                  |
| Context length      | How much past text fits on the desk at once                                  |
| Weights             | The learned numbers on the shelves                                           |
| Causal              | May only look at the past, not the future                                    |

---

## 15. Honest limits

This project is a **teaching instrument**, not a production cloud service.

- It runs on the ordinary processor and keeps numbers in a simple, memory-hungry form.
- It is slower than GPU systems you meet in products.
- Small models hallucinate, waffle, and latch onto polite filler — especially if prompts are vague.
- “Understanding,” “knowing,” “thinking,” and “meaning” here are **metaphors** for statistical continuation and inner
  arithmetic. A humanities reader is right to keep that distinction sharp.

If this book did its job, you can now explain to another non-specialist:

> Loading unpacks a fixed library of trained numbers. Attention is how each moment rereads the current page. What we
> call thinking is either that silent stacked work, or more language about reasoning that the same attention can later
> use. Then the program repeatedly draws the next scrap of text until the reply ends.

That is enough to understand what this Java program is doing — and what it is not.
