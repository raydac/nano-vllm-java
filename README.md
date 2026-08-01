# Nano-vLLM (Java)

Pure Java 21+ port of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — a lightweight vLLM-style offline
inference engine.

## Key Features

* Continuous batching scheduler with paged KV cache and prefix caching
* Pluggable causal LMs: **Qwen3** (default) and **Gemma3** (text), selected via path / config auto-detect /
  `-Dnanovllm.arch`
* HuggingFace `config.json` + `.safetensors` weight loading
* BPE tokenizer loader for `tokenizer.json` (GPT-2 byte BPE and Metaspace/`▁` for Gemma)
* **No native / CUDA / PyTorch** — JDK 21+ plus **Gson** for JSON (HF configs / tokenizer / safetensors headers)

## Requirements

* JDK 21+
* Maven 3.9+ (`mvn` on `PATH`)
* Gson (pulled by Maven)

## Maven coordinates

|               |                                                                         |
|---------------|-------------------------------------------------------------------------|
| GroupId       | `com.igormaznitsa`                                                      |
| ArtifactId    | `nano-vllm-java`                                                        |
| Version       | `0.2.0-SNAPSHOT`                                                        |
| JPMS module   | `com.igormaznitsa.nanollvm`                                             |
| Java packages | `com.igormaznitsa.nanollvm` (+ `chat`, `tokenizer`, `prompts`, `utils`) |

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

Optional Vector API (faster kernels): add `jdk.incubator.vector` / `--add-modules jdk.incubator.vector`. Scalar kernels
are used when that module is absent.

## Build

```bash
cd nano-vllm-java
mvn -q test
mvn -q package
```

## Model

Weights live outside `src/` in the project-root `models/` folder (gitignored checkpoints, tracked download scripts):

```
models/
  README.md
  download-qwen3-0.6b.sh
  download-gemma3-270m.sh
  Qwen3-0.6B/          # default (~1.5GB)
  Gemma3-270M/         # optional; HF license + HF_TOKEN
```

```bash
./models/download-qwen3-0.6b.sh
# optional Gemma (accept license, then HF_TOKEN / huggingface-cli login):
./models/download-gemma3-270m.sh
```

Windows: `.\models\download-qwen3-0.6b.ps1` / `.\models\download-gemma3-270m.ps1` (or `.cmd`).

`Example` / `Bench` resolve `models/Qwen3-0.6B` by default via `BundledModels`.

| Override    | Example                                                        |
|-------------|----------------------------------------------------------------|
| CLI         | `mvn … -Dexec.args=models/Gemma3-270M`                         |
| Property    | `-Dnanovllm.model=models/Gemma3-270M`                          |
| Force arch  | `-Dnanovllm.arch=gemma3` or `qwen3`                            |
| Env         | `NANOVLLM_MODEL=models/Gemma3-270M`                            |
| Models root | `-Dnanovllm.models.dir=/other/models` or `NANOVLLM_MODELS_DIR` |

## Quick Start

Interactive dialog (after model load):

```text
?> introduce yourself
assistant> ...
?> list primes within 20
assistant> ...
?> /exit
```

Or in code (library — quiet by default):

```java
try(LLM llm = LLM.builder(BundledModels.resolveDefault())
        .enforceEager(true)
        .maxModelLen(2048)
        .systemPrompt("Answer briefly and factually.") // optional
        .build()){

// Multi-turn chat (history + template + truncation)
String reply = llm.chat(256).send("Hello, Nano-vLLM.").answer();

// One-shot chat
String once = llm.chatOnce("What is 2+2?");

// Raw completion (no chat template)
String raw = llm.complete("The capital of France is");
}

// CLI / tools that want load progress:
// LLM.builder(path).withSystemIo().build()
```

**Library notes:** one `LLM` per concurrent generate; call `llm.cancel()` to abort; optional
`chat.timeout(Duration.ofSeconds(30))`; load failures throw `ModelLoadException`.

```bash
# .mvn/jvm.config already adds jdk.incubator.vector for this project.
# Extra heap for model load:
MAVEN_OPTS="-Xmx8g" mvn -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.Example
```

Requires JDK 21+ with the Vector incubator module available. Surefire gets
`--add-modules jdk.incubator.vector` via `jvm.module.args` in the POM; `exec:java`
inherits JVM flags from `.mvn/jvm.config` (it cannot take `<jvmArgs>` — that goal is not forked).

## Differences from Python

| Area            | Python             | Java                          |
|-----------------|--------------------|-------------------------------|
| Device          | CUDA + flash-attn  | CPU float32                   |
| Tensor parallel | NCCL multi-process | `tensor_parallel_size=1` only |
| CUDA graphs     | Supported          | Eager only                    |
| Hash            | xxhash             | Pure-JDK block hash           |
| Throughput      | GPU-class          | Educational / CPU baseline    |

## Layout

```
com.igormaznitsa.nanollvm          (~24 teaching-oriented sources)
├── LLM / EngineIo / SamplingParams / SamplingDefaults / Example / Bench
├── chat/       ChatSession, ChatMessage, ChatRole, StreamPrinter
├── engine/     Scheduler, Sequence, BlockManager, ModelRunner
├── layers/     Attention, Linear(+Qkv/Merged/…), Norms, Sampler, Embedding
├── models/     Qwen3ForCausalLM, Gemma3ForCausalLM
├── tensor/     Tensor, Ops, VectorMath
├── tokenizer/  HuggingFace BPE
└── utils/      Json, SafetensorsReader, ModelLoader, Context, BundledModels
```

## License

MIT (same as upstream nano-vllm).
