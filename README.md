![Banner](assets/banner.png)

[![License Apache 2.0](https://img.shields.io/badge/license-Apache%20License%202.0-green.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Java 21+](https://img.shields.io/badge/java-21.0%2b-green.svg)](https://bell-sw.com/pages/downloads/)
[![Maven 3.9+](https://img.shields.io/badge/maven-3.9%2b-green.svg)](https://maven.apache.org/)

# Nano-vLLM (Java)

Pure **Java 21+** LLM inference engine: continuous batching, paged KV cache, and Hugging Face–compatible weight loading
on **CPU only** — no CUDA, PyTorch, or third-party runtime libraries.

Ideas in this project were inspired by the Python [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) educational
engine.

For a guided tour of the design (scheduler, attention, tensors, RAG), see [`description.md`](description.md).

## Key features

- Continuous batching scheduler with paged KV cache and prefix caching
- **Qwen3** (default) and **Gemma3** text causal LMs; architecture from `config.json` or `-Dnanovllm.arch`
- Loads HF `config.json`, `tokenizer.json`, and `.safetensors` weights
- GPT-2 byte BPE and Gemma Metaspace BPE tokenizers
- Optional **BM25 text RAG** over a local `rag/` corpus (used automatically by the Example CLI)
- In-project JSON parser (no Gson or other JSON dependency)

## Requirements

| Requirement                          | Notes                                                                      |
|--------------------------------------|----------------------------------------------------------------------------|
| **JDK 21+**                          | Language and runtime                                                       |
| **Maven 3.9+**                       | Build and `exec:java` (`mvn` on `PATH`)                                    |
| **~2–8 GB heap**                     | Model load; use `MAVEN_OPTS=-Xmx8g` for Qwen3-0.6B on modest machines      |
| **Optional:** `jdk.incubator.vector` | Faster kernels; enabled via [`.mvn/jvm.config`](.mvn/jvm.config) for Maven |

## Build

Clone the repository, then compile and test:

```bash
cd nano-vllm-java
mvn test
mvn package
```

Artifacts:

- `target/nano-vllm-java-0.2.0-SNAPSHOT.jar` — library JAR (JPMS module `com.igormaznitsa.nanollvm`)
- `target/classes/` — compiled module for development runs

Tests use the Vector incubator module (`jvm.module.args` in the POM). Production runs should use the same flags
(see [Run from the CLI](#run-from-the-cli)).

### Use as a dependency

```xml
<dependency>
  <groupId>com.igormaznitsa</groupId>
  <artifactId>nano-vllm-java</artifactId>
  <version>0.2.0-SNAPSHOT</version>
</dependency>
```

On the module path:

```text
requires com.igormaznitsa.nanollvm;
```

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

At load time, `ModelFactory` reads `config.json`, builds the graph (Qwen3 or Gemma3), merges packed weights from
safetensors, and constructs the tokenizer. Architecture is inferred from `model_type` / `architectures` unless you set
`-Dnanovllm.arch=qwen3|gemma3`.

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

**Windows:** `.\models\download-qwen3-0.6b.ps1` / `.cmd` and the matching Gemma scripts.

You can also point the engine at **any** local HF-style directory (your own path or another download).

### How the default model path is chosen

`BundledModels.resolveDefault()` (used by `Example` and `Bench`) picks the model in this order:

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

If you start **without** any of (1)– (3), the Example CLI shows an interactive menu (Qwen / Gemma / exit).

## Run from the CLI

### Interactive chat (`Example`)

Recommended entry point: multi-turn chat with streaming output. Thinking tokens go to **stderr** (dim cyan when color is
enabled); the reply goes to **stdout**.

```bash
# After downloading Qwen3-0.6B — use enough heap for load + inference
MAVEN_OPTS="-Xmx8g" mvn -q exec:java
```

Pick a model interactively, or pass it explicitly:

```bash
MAVEN_OPTS="-Xmx8g" mvn -q exec:java -Dexec.args="models/Gemma3-270M"
```

```bash
NANOVLLM_MODEL=models/Qwen3-0.6B MAVEN_OPTS="-Xmx8g" mvn -q exec:java
```

**RAG mode:** if the directory `rag/` exists (the repo includes sample fairy-tale texts and fact cards), Example builds
a shared BM25 index and uses the `rag?>` prompt. Otherwise it uses plain chat (`?>`).

Example session:

```text
Loading model from …/models/Qwen3-0.6B
RAG: prepared BM25 over …/rag (… chunks, shared index)
Type a message and press Enter. Commands: /exit  /quit  /clear

rag?> What does nano-vllm-java run on?
assistant> …
(retrieved 2 chunk(s): facts-….md)

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

Maven note: `exec:java` runs in the **same JVM as Maven**; Vector API flags come from [
`.mvn/jvm.config`](.mvn/jvm.config) (`--add-modules=jdk.incubator.vector`). The exec plugin does not fork, so
`<jvmArgs>` in the POM are not applied — use `MAVEN_OPTS` for heap.

### Run the packaged JAR (module path)

After `mvn package`:

```bash
java --add-modules jdk.incubator.vector \
  -Xmx8g \
  -p target/nano-vllm-java-0.2.0-SNAPSHOT.jar \
  -m com.igormaznitsa.nanollvm/com.igormaznitsa.nanollvm.Example \
  models/Qwen3-0.6B
```

Replace the main class with `com.igormaznitsa.nanollvm.Bench` for throughput smoke tests.

### Benchmark (`Bench`)

Loads the same default model and runs random token-id batches (scheduler / KV stress):

```bash
MAVEN_OPTS="-Xmx8g" mvn -q exec:java \
  -Dexec.mainClass=com.igormaznitsa.nanollvm.Bench \
  -Dexec.args="models/Qwen3-0.6B 8"
```

Second argument is the number of concurrent sequences (default `8`).

## Library quick start

Load once, share `Model` across many `LLM` instances; generation is not concurrent on a single `LLM` (use one instance
per thread or call sequentially).

```java
import com.igormaznitsa.nanollvm.models.Model;
import com.igormaznitsa.nanollvm.models.ModelFactory;
import com.igormaznitsa.nanollvm.llm.EngineIo;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.utils.BundledModels;
import java.nio.file.Path;

Path modelDir = BundledModels.resolveDefault(); // or Path.of("models/Qwen3-0.6B")
Model model = ModelFactory.make(modelDir);      // quiet; use EngineIo.system() for progress

try(
LLM llm = LLM.builder(model)
    .enforceEager(true)
    .maxModelLen(2048)
    .systemPrompt("Answer briefly and factually.")
    .build()){

String reply = llm.chat(256).send("Hello.").answer();
String once = llm.chatOnce("What is 2+2?");
String completion = llm.complete("The capital of France is");
}
```

**RAG:** index documents once, attach to any LLM:

```java
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.rag.*;

PreparedRag rag = RagFactory.make(Path.of("rag"));
try(
LLM llm = LLM.builder(model).build()){
String answer = llm.rag(rag).topK(4).ask("Your question");
}
```

See [`description.md`](description.md) §17 and package `com.igormaznitsa.nanollvm.rag` for retrieval options.

## Source layout

```
com.igormaznitsa.nanollvm
├── Example, Bench, …
├── models/     Model, ModelFactory, Qwen3 / Gemma3 causal LM graphs
├── llm/        LLM, Config, EngineIo, SamplingParams, SamplingDefaults
├── chat/       ChatSession, streaming, templates
├── rag/        BM25 index, RagSession, RagFactory
├── engine/     Scheduler, KV cache, ModelRunner
├── layers/     Attention, Linear, norms, sampler
├── tensor/     Tensor, Ops, Vector/scalar kernels
├── tokenizer/  HF BPE loader
├── internal/   Safetensors loader, weight merge, inference Context (module-private)
└── utils/      Json, BundledModels, BundledRag, NanoVllmProps
```

## License

[Apache License 2.0](LICENSE).
