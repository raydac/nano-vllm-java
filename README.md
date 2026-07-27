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
* Maven via `mvn` on this machine
* Gson (pulled by Maven)

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

```bash
MAVEN_OPTS="-Xmx8g" mvn -q exec:java -Dexec.mainClass=io.nanovllm.Example
```

Requires `--add-modules jdk.incubator.vector` (already set in the POM for `test` / `exec:java`).

Or in code:

```java
try(LLM llm = new LLM(BundledModels.resolveDefault(), Map.of("enforce_eager", true))){
SamplingParams sp = new SamplingParams(0.6f, 256);
var outputs = llm.generate(List.of("Hello, Nano-vLLM."), sp);
    System.out.

println(outputs.get(0).

text());
    }
```

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
io.nanovllm          (~24 teaching-oriented sources)
├── LLM / SamplingParams / Config / Example / Bench
├── engine/     Scheduler, Sequence, BlockManager, ModelRunner
├── layers/     Attention, Linear(+Qkv/Merged/…), Norms, Sampler, Embedding
├── models/     Qwen3ForCausalLM
├── tensor/     Tensor, Ops, VectorMath
├── tokenizer/  HuggingFace BPE
└── utils/      Json, SafetensorsReader, ModelLoader, Context, BundledModels
```

## License

MIT (same as upstream nano-vllm).
