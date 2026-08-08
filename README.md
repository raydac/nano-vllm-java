![Banner](assets/banner.png)

[![License Apache 2.0](https://img.shields.io/badge/license-Apache%20License%202.0-green.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Java 21+](https://img.shields.io/badge/java-21.0%2b-green.svg)](https://bell-sw.com/pages/downloads/)
[![Maven 3.9+](https://img.shields.io/badge/maven-3.9%2b-green.svg)](https://maven.apache.org/)   
[![Arthur's acres sanctuary donation](assets/arthur_sanctuary_banner.png)](https://www.arthursacresanimalsanctuary.org/donate)

# Nano-vLLM Java

Pure **Java 21+** LLM inference library: continuous batching, paged KV cache, and Hugging Face–compatible weight
loading on **CPU only** — no CUDA, PyTorch, or native runtime bindings. Add it to a Maven or Gradle app and call it
from ordinary Java.

Ideas in this project were inspired by the Python [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) educational
engine.

For a guided tour of the design (scheduler, attention, tensors, RAG), see [`description.md`](description.md).

## Hello World — Gemma3 log triage in your app

This is the usual path for library users: declare the dependency, point at a **local Gemma3** folder (any path you
choose), ask a short business question, print the answer.

### 1. Add the dependency

**Maven**

```xml
<dependency>
  <groupId>com.igormaznitsa</groupId>
  <artifactId>nano-vllm-java</artifactId>
  <version>1.0.0-SNAPSHOT</version>
</dependency>
```

**Gradle (Groovy)**

```gradle
implementation 'com.igormaznitsa:nano-vllm-java:1.0.0-SNAPSHOT'
```

JPMS module name: `com.igormaznitsa.nanollvm` (`requires com.igormaznitsa.nanollvm;`).
Runtime: **JDK 21+**, enough heap for the checkpoint (Gemma3-270M is typically fine with a few GB).

Download a Gemma3 instruct snapshot once (HF license + token), for example into `/opt/models/Gemma3-270M` — see
[Download scripts](#download-scripts) or Hugging Face
[google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it).

### 2. Load Gemma3 from your path and triage a log snippet

```java
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;

import java.nio.file.Path;

public final class LogTriageHelloWorld {

  public static void main(String[] args) {
    // Custom install location — not tied to this repo's ./models layout
    Path gemmaDir = Path.of("/opt/models/Gemma3-270M");

    LlmModel model = LlmModelFactory.make(gemmaDir);

    String logExcerpt = """
        2026-08-07 22:14:01 WARN  payment-api - retry 1/3 for order=99102 cause=SocketTimeoutException
        2026-08-07 22:14:04 WARN  payment-api - retry 2/3 for order=99102 cause=SocketTimeoutException
        2026-08-07 22:14:08 ERROR payment-api - give up order=99102 after 3 timeouts upstream=billing-svc:8443
        2026-08-07 22:14:08 INFO  payment-api - marked order=99102 status=PAYMENT_FAILED
        """;

    String prompt = """
        You are helping an on-call engineer. Read the log lines and reply in three short bullets:
        1) what failed
        2) likely cause
        3) one concrete next check
        Do not invent hosts or error codes that are not in the log.

        Log:
        %s
        """.formatted(logExcerpt);

    try (LLM llm = LLM.builder(model)
        .noSystemPrompt()          // Gemma chat path: keep the system role empty
        .maxModelLen(2048)
        .build()) {

      String advice = llm.chat(128).send(prompt).answer();
      System.out.println(advice);
    }
  }
}
```

What this shows: **pure Java** in / out, **your** model directory, one `LlmModelFactory.make` + `LLM.builder` +
`chat(…).send(…).answer()` — no Python sidecar. Swap the path for another Gemma3 layout or use Qwen3 the same way
(with `.systemPrompt(…)` if you want a fixed role).

In this repository the same program lives as non-exported
`com.igormaznitsa.nanollvm.samples.LogTriageHelloWorld` (defaults to `models/Gemma3-270M` via
`samples.utils.BundledModels`; optional first arg overrides the path):

```bash
mvn -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.LogTriageHelloWorld
# or: … -Dexec.args=/opt/models/Gemma3-270M
```

More API samples (streaming, RAG, GGUF, advisors) are in [Library quick start](#library-quick-start).

## Key features

- Continuous batching scheduler with paged KV cache and prefix caching
- **Qwen3** (default), **Gemma3**, and **LFM2** (hybrid short-conv + GQA) causal LMs
- Loads HF `config.json` + `.safetensors`, or a single **`.gguf`** file (dequantized to float32)
- Optional multi-thread CPU matmul (`cpuThreads` / `allCpuThreads` / `disableMultiCpu`); default = all processors
- GPT-2 byte BPE, Gemma Metaspace BPE, and GGUF-embedded tokenizers
- Optional **BM25 text RAG** over a local `rag/` corpus (used automatically by the Example CLI)
- Optional **advisors** before each chat/RAG turn (`LLM.setAdvisors`)

## Requirements

| Requirement                          | Notes                                                                      |
|--------------------------------------|----------------------------------------------------------------------------|
| **JDK 21+**                          | Language and runtime                                                       |
| **Maven 3.9+**                       | Build and `exec:java` (`mvn` on `PATH`)                                    |
| **~2–8 GB heap**                     | Enough for Qwen3-0.6B / Gemma3-270M                                         |
| **~16 GB heap**                      | Default in [`.mvn/jvm.config`](.mvn/jvm.config) (`-Xmx16g`) for LFM2 GGUF    |
| **Optional:** `jdk.incubator.vector` | Faster kernels; enabled via [`.mvn/jvm.config`](.mvn/jvm.config) for Maven |

## Build

Clone the repository, then compile and test:

```bash
cd nano-vllm-java
mvn test
mvn package
```

Artifacts:

- `target/nano-vllm-java-1.0.0-SNAPSHOT.jar` — library JAR (JPMS module `com.igormaznitsa.nanollvm`)
- `target/classes/` — compiled module for development runs

Tests use the Vector incubator module (`jvm.module.args` in the POM). Production runs should use the same flags
(see [Run from the CLI](#run-from-the-cli)).

### Use as a dependency

See [Hello World](#hello-world--gemma3-log-triage-in-your-app) for Maven / Gradle coordinates and a complete Gemma3
example. Snapshot builds from this repository still use version `1.0.0-SNAPSHOT` until you publish.

```xml
<dependency>
  <groupId>com.igormaznitsa</groupId>
  <artifactId>nano-vllm-java</artifactId>
  <version>1.0.0-SNAPSHOT</version>
</dependency>
```

On the module path:

```text
requires com.igormaznitsa.nanollvm;
```

Public API packages: `models`, `llm`, `chat`, `rag`, `tokenizer`, `prompts`, `utils`, `exceptions`.
`samples` (`Example`, `Bench`, `LogTriageHelloWorld`, `samples.utils`), `engine`, `layers`, `tensor`, and
`internal` are **not** exported — use `LlmModelFactory` / `LLM` / `RagFactory` from application code. Demos
remain runnable as main classes
(`mvn exec:java` or `java -m com.igormaznitsa.nanollvm/com.igormaznitsa.nanollvm.samples.Example`).

## Download and load models

Weights are **not** committed to git. They live under the project-root `models/` directory (see [
`models/README.md`](models/README.md)).

### What a valid model directory contains

Each checkpoint folder must look like a standard Hugging Face snapshot:

```
models/Qwen3-0.6B/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors          # one or more *.safetensors
  …                          # merges.txt / vocab.json as needed
```

At load time, `LlmModelFactory` reads `config.json`, builds the graph (Qwen3 or Gemma3), merges packed weights from
safetensors, and constructs the tokenizer. Architecture is inferred from `model_type` / `architectures` unless you set
`-Dnanovllm.arch=qwen3|gemma3|lfm2`.

### GGUF (LFM2)

A single `.gguf` file is also valid. Example: LiquidAI [LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
`Q4_K_M` (~1.67 GB on disk). Weights are **dequantized to float32** at load (~10 GB weights alone; plan on
**~16 GB heap** with KV/JVM) and run on the same CPU kernels; there is no quantized matmul path.

```bash
./models/download-lfm2.5-2.6b-gguf.sh
mvn -q exec:java \
  -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Example \
  -Dexec.args=models/LFM2.5-2.6B-Q4_K_M.gguf
```

Supported GGUF dtypes for this path: `Q4_K`, `Q4_0`, `Q6_K`, `Q8_0`, `F16`, `BF16`, `F32`. Architecture must be
`lfm2` (hybrid short-convolution + attention). Heap defaults to **16 GB** via `.mvn/jvm.config` (override with
`MAVEN_OPTS` if you need more).

### Download scripts

**Qwen3-0.6B (default, ~1.5 GB, no HF gate)**

```bash
./models/download-qwen3-0.6b.sh
```

**Gemma3-270M (optional, license on Hugging Face)**

Accept terms at [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it), then:

```bash
export HF_TOKEN=hf_…   # or: huggingface-cli login
./models/download-gemma3-270m.sh
```

**LFM2.5 GGUF** — see [GGUF (LFM2)](#gguf-lfm2) above.

**Windows:** `.\models\download-qwen3-0.6b.ps1` / `.cmd` and the matching Gemma / LFM scripts under `models/`.

You can also point the engine at **any** local HF-style directory (your own path or another download).

### How the default model path is chosen

`samples.utils.BundledModels.resolveDefault()` (used by `Example` and `Bench`; not a library API) picks the model in this order:

1. **First CLI argument** — model path or name (e.g. `models/Gemma3-270M`)
2. **System property** `-Dnanovllm.model=…`
3. **Environment** `NANOVLLM_MODEL=…`
4. **Default** `models/Qwen3-0.6B` under the models root

The models root itself defaults to `./models`, overridable with `-Dnanovllm.models.dir=…` or `NANOVLLM_MODELS_DIR`.

| Mechanism          | Example                                                             |
|--------------------|---------------------------------------------------------------------|
| CLI arg            | `mvn … -Dexec.args=models/Gemma3-270M`                              |
| Property           | `-Dnanovllm.model=/data/hf/Qwen3-0.6B`                              |
| Environment        | `NANOVLLM_MODEL=models/Gemma3-270M`                                 |
| Models root        | `-Dnanovllm.models.dir=/opt/models`                                 |
| Force architecture | `-Dnanovllm.arch=gemma3` (when auto-detect is wrong)                |
| RAG corpus dir     | `-Dnanovllm.rag.dir=./docs` or `NANOVLLM_RAG_DIR` (default `./rag`) |
| CPU matmul threads | `-Dnanovllm.cpu.threads=N` or `.cpuThreads(N)` / `.allCpuThreads()` / `.disableMultiCpu()` |

If you start **without** any of (1)–(3), the Example CLI shows an interactive menu (**Qwen3 / Gemma3 / LFM2 / Exit**).

## Run from the CLI

### Interactive chat (`Example`)

Recommended entry point: multi-turn chat with streaming output. Thinking tokens go to **stderr** (dim cyan when color is
enabled); the reply goes to **stdout**.

```bash
# After downloading a model — heap defaults to -Xmx16g via .mvn/jvm.config
mvn -q exec:java
```

Pick a model interactively, or pass it explicitly:

```bash
mvn -q exec:java -Dexec.args="models/Gemma3-270M"
```

```bash
NANOVLLM_MODEL=models/Qwen3-0.6B mvn -q exec:java
```

**RAG mode:** if the directory `rag/` exists (the repo ships Grimm / Little Red Riding Hood `.txt` and fact cards),
Example builds a shared BM25 index and uses the `rag?>` prompt. Otherwise it uses plain chat (`?>`).

**Advisors (Example only):** after load, advisors are wired by architecture — **Gemma** 3 roles (PARALLEL), **Qwen**
2 roles (PARALLEL), **LFM** none. Advisor notes appear on the thinking stream; grounded RAG mixes only Context-supported
hints into the main prompt.

Example session (ask about the demo corpus):

```text
Loading model from …/models/Qwen3-0.6B
RAG: prepared BM25 over …/rag (… chunks, shared index)
Advisors: 2 (practical, abstract) PARALLEL for Qwen.
Type a message and press Enter. Commands: /exit  /quit  /clear

rag?> who are the grimm brothers?
assistant> …
(retrieved 3 chunk(s): facts-brothers-grimm.md)

rag?> who was their father?
assistant> …
(retrieved … chunk(s): …)

rag?> /clear
(conversation cleared; RAG index kept)

rag?> /exit
```

Session recording (Gemma3 load + RAG questions about the Grimm brothers and their father):

![Gemma3 RAG session recording](assets/java_nano_llvm_session_1.gif)

| Command                          | Action                                      |
|----------------------------------|---------------------------------------------|
| `/exit`, `/quit`, `exit`, `quit` | Leave the program                           |
| `/clear`                         | Reset chat history (RAG index stays loaded) |

**Display:** set `NO_COLOR=1` or `-Dnanovllm.color=false` to disable ANSI colors.

Maven note: `exec:java` runs in the **same JVM as Maven**. Vector API flags and heap come from [
`.mvn/jvm.config`](.mvn/jvm.config) (`--add-modules=jdk.incubator.vector`, `-Xmx16g`). The exec plugin does not fork, so
`<jvmArgs>` in the POM are not applied — override with `MAVEN_OPTS` when needed.

### Run the packaged JAR (module path)

After `mvn package`:

```bash
java --add-modules jdk.incubator.vector \
  -Xmx16g \
  -p target/nano-vllm-java-1.0.0-SNAPSHOT.jar \
  -m com.igormaznitsa.nanollvm/com.igormaznitsa.nanollvm.samples.Example \
  models/Qwen3-0.6B
```

Replace the main class with `com.igormaznitsa.nanollvm.samples.Bench` for throughput smoke tests.

### Benchmark (`Bench`)

Loads the same default model and runs random token-id batches (scheduler / KV stress):

```bash
mvn -q exec:java \
  -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Bench \
  -Dexec.args="models/Qwen3-0.6B 8"
```

Second argument is the number of concurrent sequences (default `8`).

## Library quick start

Load once, share `LlmModel` across many `LLM` instances. One `LLM` is **not** safe for concurrent `generate` / chat —
use one instance per thread, or call sequentially. `LLM.cancel()` is safe from another thread.

### Chat, one-shot, and completion

```java
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.llm.LLM;

import java.nio.file.Path;

Path modelDir = Path.of("models/Qwen3-0.6B"); // your local HF (or .gguf) path
LlmModel model = LlmModelFactory.make(modelDir);    // quiet; LlmModelFactory.make(dir, LlmListeners.toSystem()) for progress

try (LLM llm = LLM.builder(model)
    .enforceEager(true)
    .maxModelLen(2048)
    .systemPrompt("Answer briefly and factually.") // Qwen-style; prefer .noSystemPrompt() on Gemma
    .build()) {

  String reply = llm.chat(256).send("Hello.").answer();
  String once = llm.chatOnce("What is 2+2?");
  String completion = llm.complete("The capital of France is");
}
```

Path convenience (private `LlmModel` inside the builder):

```java
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.LLM;

import java.nio.file.Path;

try (LLM llm = LLM.builder(Path.of("models/Qwen3-0.6B")).listen(LlmListeners.toSystem()).build()) {
  System.out.println(llm.chatOnce("Say hi in one sentence."));
}
```

### GGUF / LFM2

```java
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;

import java.nio.file.Path;

LlmModel model = LlmModelFactory.make(Path.of("models/LFM2.5-2.6B-Q4_K_M.gguf"), LlmListeners.toSystem());
try (LLM llm = LLM.builder(model)
    .maxModelLen(2048)
    .allCpuThreads()
    .build()) {
  System.out.println(llm.chatOnce("Hello"));
}
```

### Streaming chat

One `LlmListener` covers status (`STATUS_INFO` / `STATUS_PROGRESS`) and chat text
(`TEXT_THINKING` / `TEXT_ASSISTANT` / …). CLI PrintStreams remain sugar over the same path:

```java
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.chat.LlmTextKind;

try (LLM llm = LLM.builder(model)
    .listen(LlmListeners.toSystem())  // status → stderr/stdout
    .build()) {
  llm.chat(256)
      .listen((source, event) -> {
        switch (event.kind()) {
          case TEXT_THINKING -> System.err.print(event.text());
          case TEXT_ASSISTANT -> System.out.print(event.text());
          case TEXT_ADVISOR_NOTE -> System.err.printf("[advisor %d] %s%n", event.slot(), event.text());
          case TEXT_DIAGNOSTICS -> System.err.println(event.text());
          case STATUS_INFO, STATUS_PROGRESS -> { /* already handled by toSystem on the LLM */ }
        }
      })
      .send("Explain paged KV cache in one short paragraph.");
}

// Equivalent CLI helper for chat channels:
try (LLM llm = LLM.builder(model).listen(LlmListeners.toSystem()).build()) {
  llm.chat(256)
      .streamTo(System.err, System.out, true)  // → LlmListeners.toPrintStreams(...)
      .send("Explain paged KV cache in one short paragraph.");
}
```

### Cancel and timeout

```java
import java.time.Duration;

try (LLM llm = LLM.builder(model).build()) {
  var chat = llm.chat(512).timeout(Duration.ofSeconds(30));
  // from another thread while generate runs:
  // llm.cancel();
  chat.send("Write a long answer…");
}
```

### Advisors

Isolated advisor generates run **before** each chat/RAG turn (no history). Notes show on the thinking stream; for RAG
hits, only Context-grounded notes are mixed into the main prompt.

```java
import com.igormaznitsa.nanollvm.llm.AdvisorMode;
import com.igormaznitsa.nanollvm.prompts.AdvisorPrompts;

try (LLM llm = LLM.builder(model).build()) {
  llm.setAdvisors(AdvisorMode.PARALLEL, AdvisorPrompts.demoRolesQwen());
  // Gemma: AdvisorPrompts.demoRolesGemma() — three roles
  // Clear: llm.setAdvisors();

  System.out.println(llm.chat(256).send("Summarize the user question briefly.").answer());
}
```

`PARALLEL` batches all advisors in one `generate`; `SEQUENTIAL` runs one generate per role (slower, same `LLM` lock).

### Text RAG

Index documents once (UTF-8 `.txt` / `.md` / … and `.pdf` via `PdfTextExtractor`), share `PreparedRag` across LLMs:

```java
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.rag.PreparedRag;
import com.igormaznitsa.nanollvm.rag.RagFactory;
import com.igormaznitsa.nanollvm.rag.RagLoadOptions;

import java.nio.file.Path;

PreparedRag rag = RagFactory.make(Path.of("rag")); // silent
// progress: RagFactory.make(Path.of("rag"), RagLoadOptions.defaults(), LlmListeners.toSystem());
// tiny models: RagFactory.make(Path.of("rag"), RagLoadOptions.forTinyModels());

try (LLM llm = LLM.builder(model).build()) {
  String answer = llm.rag(rag)
      .topK(4)
      .maxContextChars(3500)
      .ask("Who are the Grimm brothers?");

  var session = llm.rag(rag, 256).topK(3);
  session.send("Who are the Grimm brothers?");
  session.send("Who was their father?"); // short follow-up: rewrite / Prior fallback
  System.out.println(session.lastHits());
}
```

Inline corpus without files:

```java
PreparedRag rag = RagFactory.of(
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.");
```

Retrieval is **lexical BM25** (no embedding model). Short anaphoric follow-ups may rewrite to keywords; if the rewrite
returns `NONE`, the session falls back to Prior + follow-up instead of aborting. Off-topic queries with contentful
out-of-vocabulary terms tend to yield no hits.

See [`description.md`](description.md) §17 and package `com.igormaznitsa.nanollvm.rag`. Prompt wording lives in
`prompts.RagPrompts`; `RagSession.formatUserMessage` builds the model-facing turn.

## Further reading

| Doc | Contents |
|-----|----------|
| [`description.md`](description.md) | Design tour: attention, tensors, scheduler, GGUF, call chain, RAG |
| [`models/README.md`](models/README.md) | Download scripts and model layout |
| [`rag/`](rag/) | Demo corpus (fairy tales + fact cards) |

## License

Apache License 2.0 — see [LICENSE](LICENSE) (or the badge at the top of this page).
