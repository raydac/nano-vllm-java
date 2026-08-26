# External model weights (not packaged in the JAR / not committed)

## Qwen3-0.6B (default)

**Linux / macOS**

```bash
./models/download-qwen3-0.6b.sh
```

**Windows** (PowerShell or Command Prompt; needs `curl.exe`, included on Windows 10+)

```powershell
.\models\download-qwen3-0.6b.ps1
```

```bat
models\download-qwen3-0.6b.cmd
```

Creates `models/Qwen3-0.6B/` (~1.5GB). Used by default when no model path is set.

Qwen3 also loads from a **single `.gguf`** (`general.architecture=qwen3`, **since 1.1.0**). There is no download
script for that crate — place a file yourself (for example
[Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)) and pass its path to `LlmModelFactory.make`.

## Gemma3-270M (optional)

Instruction-tuned text Gemma 3 (270M). **License-gated** on Hugging Face — accept terms at
https://huggingface.co/google/gemma-3-270m-it then authenticate (`huggingface-cli login` or `HF_TOKEN`).

```bash
export HF_TOKEN=hf_...   # if needed
./models/download-gemma3-270m.sh
```

Windows: `.\models\download-gemma3-270m.ps1` or `models\download-gemma3-270m.cmd`.

Creates `models/Gemma3-270M/`.

## Gemma4-E2B-IT-QAT-Mobile (optional)

[google/gemma-4-E2B-it-qat-mobile-transformers](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers)
— instruction-tuned Gemma 4 E2B, QAT mobile (`wNa8o8`) safetensors. Apache 2.0, ungated. About **2.3 GB**.
Default **16 GB heap** in [`.mvn/jvm.config`](../.mvn/jvm.config) is the intended headroom.

```bash
./models/download-gemma4-e2b-qat-mobile.sh
```

Windows: `.\models\download-gemma4-e2b-qat-mobile.ps1` or `models\download-gemma4-e2b-qat-mobile.cmd`.

Creates `models/Gemma4-E2B-IT-QAT-Mobile/`. Resume-safe (`curl -C -`). Optional `HF_TOKEN` if Hugging Face
rate-limits you.

`LlmModelFactory.make` loads this folder as **text-only** chat (packed QAT). Vision and audio towers in the
same checkpoint are unused.

## LFM2.5-2.6B GGUF (optional)

Hybrid Liquid LFM2 checkpoint as a single GGUF file. Weights stay packed in RAM (dequant on matmul); default
**~16 GB heap** in [`.mvn/jvm.config`](../.mvn/jvm.config) is safe headroom for KV / activations.

```bash
./models/download-lfm2.5-2.6b-gguf.sh
```

Windows: `.\models\download-lfm2.5-2.6b-gguf.ps1` or `models\download-lfm2.5-2.6b-gguf.cmd`.

Creates `models/LFM2.5-2.6B-Q4_K_M.gguf`.

## Tiny-LLM-ONNX (optional, Llama ONNX demo)

[onnx-community/Tiny-LLM-ONNX](https://huggingface.co/onnx-community/Tiny-LLM-ONNX) — ~10M Llama, ONNX weights under
`onnx/`. Tokenizer files are pulled from [arnir0/Tiny-LLM](https://huggingface.co/arnir0/Tiny-LLM). Uses fp32
`onnx/model.onnx` (quantized community files are ignored by the loader).

```bash
./models/download-tiny-llm-onnx.sh
```

Windows: `.\models\download-tiny-llm-onnx.ps1` or `models\download-tiny-llm-onnx.cmd`.

Creates `models/Tiny-LLM-ONNX/`.

Sample: `nano-vllm-java-samples` → `com.igormaznitsa.nanollvm.samples.NextTokenHelloWorld`
(encode a seed, print the next tokens, continue the text). Use `generate(LlmInText, TEXT)` /
`generateTokenIds`, not chat templates.

## SmolLM2-135M-Instruct-ONNX (optional, Llama ChatML ONNX)

[onnx-community/SmolLM2-135M-Instruct-ONNX](https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX) —
instruct-tuned SmolLM2 (~135M, ChatML). Downloads `onnx/model_fp16.onnx` (~270 MiB; quantized `*_q4*` /
`*_int8*` files are ignored). For plain completion (not chat), see
[onnx-community/SmolLM2-135M-ONNX](https://huggingface.co/onnx-community/SmolLM2-135M-ONNX) instead.

```bash
./models/download-smollm2-135m-instruct-onnx.sh
```

Windows: `.\models\download-smollm2-135m-instruct-onnx.ps1` or `models\download-smollm2-135m-instruct-onnx.cmd`.

Creates `models/SmolLM2-135M-Instruct-ONNX/`.

## gte-small GGUF (optional, embeddings)

[ChristianAzinn/gte-small-gguf](https://huggingface.co/ChristianAzinn/gte-small-gguf) — Alibaba GTE-small as GGUF
(**BERT embedding / feature extraction**, not a causal chat model). Scripts fetch the **smallest** quant
(`gte-small.Q2_K.gguf`, ~25 MB). MIT license; no HF gate.

```bash
./models/download-gte-small-gguf.sh
```

Windows: `.\models\download-gte-small-gguf.ps1` or `models\download-gte-small-gguf.cmd`.

Creates `models/gte-small.Q2_K.gguf`. Context length up to 512 tokens.

Load like any other model, then embed:

```java
try (LlmModel model = LlmModelFactory.make(Path.of("models/gte-small.Q2_K.gguf"))) {
  LlmOutEmbedding v = (LlmOutEmbedding) model.generate(LlmInText.of("hello world"), LlmModality.EMBEDDING);
}
```

Sample: pass this GGUF to `EmbeddingsHelloWorld` if you want a tiny English checkpoint instead of E5.
Prefer `LLM.builder` for the shared matmul pool; do not use chat.

## multilingual-e5-small ONNX (optional, embeddings)

[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) — multilingual E5
(BERT encoder, XLM-RoBERTa Unigram tokenizer, 94 languages). Hugging Face **safetensors BERT is not
loaded** by this library; the scripts fetch `onnx/model.onnx` (fp32, ~470 MB) plus tokenizer sidecars.
MIT license; no HF gate. Skip quantized `*_qint8*` / OpenVINO files.

```bash
./models/download-multilingual-e5-small.sh
```

Windows: `.\models\download-multilingual-e5-small.ps1` or `models\download-multilingual-e5-small.cmd`.

Creates `models/multilingual-e5-small/`. Context length up to 512 tokens. E5 expects prefixes
`query: …` and `passage: …`.

```java
try (LlmModel model = LlmModelFactory.make(Path.of("models/multilingual-e5-small"))) {
  LlmOutEmbedding v = (LlmOutEmbedding) model.generate(LlmInText.of("query: hello world"), LlmModality.EMBEDDING);
}
```

Prefer `LLM.builder` for the shared matmul pool; do not use chat. Sample:
`nano-vllm-java-samples` → `com.igormaznitsa.nanollvm.samples.EmbeddingsHelloWorld`
(default folder; non-retrieval texts get `query: `).

## xlm-roberta-base ONNX (optional, embeddings)

[FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) — multilingual
XLM-RoBERTa encoder (94 languages, fill-mask checkpoint). Hugging Face **safetensors is not
loaded**; the scripts fetch Hub `model.onnx` (~1.9 GB fp32) and save it as `onnx/model.onnx` so
the folder matches other ONNX layouts. MIT license; no HF gate. Optional `HF_TOKEN` if Hugging Face
rate-limits the large file.

```bash
./models/download-xlm-roberta-base.sh
```

Windows: `.\models\download-xlm-roberta-base.ps1` or `models\download-xlm-roberta-base.cmd`.

Creates `models/xlm-roberta-base/`. Context length up to 512 tokens. Mean-pooled embeddings via
`generate(LlmInText, EMBEDDING)` — not a dedicated sentence-embedding model like E5 (no
`query:` / `passage:` prefixes). Raw cosine between unrelated strings is often ~0.99
(anisotropy). In `Example`, pick **Classify** after load, teach `label | text` (or `/demo` /
`/load nano-vllm-java-samples/classify-labels.example.txt`), then predict — a centered prototype
probe, not a sequence-classification head.

```java
try (LlmModel model = LlmModelFactory.make(Path.of("models/xlm-roberta-base"))) {
  LlmOutEmbedding v = (LlmOutEmbedding) model.generate(LlmInText.of("hello world"), LlmModality.EMBEDDING);
}
```

Prefer `LLM.builder` for the shared matmul pool; do not use chat.

## whisper-base / whisper-tiny (optional, speech-to-text)

[openai/whisper-base](https://huggingface.co/openai/whisper-base) (~290 MB) and
[openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) (~150 MB) — Hugging Face
**safetensors** Whisper (`config.json` + `model.safetensors` + tokenizer). MIT license; no HF gate.
CTranslate2 / faster-whisper `model.bin`, Whisper GGUF, and Whisper ONNX are not loaded.

```bash
./models/download-whisper-base.sh
```

Windows: `.\models\download-whisper-base.ps1` or `models\download-whisper-base.cmd`.

Tiny: `./models/download-whisper-tiny.sh` (or `.ps1` / `.cmd`). Creates `models/whisper-base/` or
`models/whisper-tiny/`. Audio is uncompressed WAV or 16 kHz mono PCM. Greedy multilingual decode;
optional language; 30 s chunking. No MP3, VAD, beam search, or word timestamps.

```java
try (LlmModel model = LlmModelFactory.make(Path.of("models/whisper-base"))) {
  LlmOutText text = (LlmOutText) model.generate(LlmInSound.ofWav(Files.readAllBytes(Path.of("clip.wav"))), LlmModality.TEXT);
}
```

Prefer `LLM.builder` for the shared matmul pool; do not use chat. Sample:
`nano-vllm-java-samples` → `com.igormaznitsa.nanollvm.samples.TranscribeHelloWorld`.

## piper-en-lessac-medium (optional, English text-to-speech)

[rhasspy/piper-voices Lessac medium](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)
— US English Piper VITS (`*.onnx` + `*.onnx.json`, ~63 MB) plus espeak-ng-data
(`lang/` and `dictsource/` list+rules). Same load path as Irina.

```bash
./models/download-piper-en-lessac-medium.sh
```

Windows: `.\models\download-piper-en-lessac-medium.ps1` or `models\download-piper-en-lessac-medium.cmd`.

Creates `models/piper-en-lessac-medium/`.

```java
try (LlmModel model = LlmModelFactory.open(Path.of("models/piper-en-lessac-medium"))
    .optionalData(LlmOptionalData.ESPEAK_DATA, Path.of("models/piper-en-lessac-medium/espeak-ng-data"))
    .make()) {
  LlmOutSoundData wav = (LlmOutSoundData) model.generate(LlmInText.of("Hello world"), LlmModality.AUDIO);
}
```

Sample: `nano-vllm-java-samples` → `SynthesizeHelloWorld` (uses Lessac when that folder exists).
In `Example`, pick the Lessac row, then at `tts?>` type `Hello world`.

## piper-ru-irina-medium (optional, text-to-speech)

[rhasspy/piper-voices Irina medium](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU/irina/medium)
— Piper VITS voice (`*.onnx` + `*.onnx.json`, ~63 MB) plus espeak-ng-data (GPL data, downloaded
next to the voice, not in the library JAR: `lang/` plus `dictsource/` lists and rules). Not a chat model. Not ONNX Runtime.

```bash
./models/download-piper-ru-irina-medium.sh
```

Windows: `.\models\download-piper-ru-irina-medium.ps1` or `models\download-piper-ru-irina-medium.cmd`.

Creates `models/piper-ru-irina-medium/`. Point espeak-ng-data with typed load extras (or keep the
default `{model}/espeak-ng-data`):

```java
try (LlmModel model = LlmModelFactory.open(Path.of("models/piper-ru-irina-medium"))
    .optionalData(LlmOptionalData.ESPEAK_DATA, Path.of("models/piper-ru-irina-medium/espeak-ng-data"))
    .make()) {
  LlmOutSoundData wav = (LlmOutSoundData) model.generate(LlmInText.of("Привет, мир"), LlmModality.AUDIO);
}
```

Do not use with `LLM.builder` / chat. Sample: `nano-vllm-java-samples` →
`com.igormaznitsa.nanollvm.samples.SynthesizeHelloWorld`.

## Selecting a model

| Mechanism          | Example                                                                                           |
|--------------------|---------------------------------------------------------------------------------------------------|
| CLI arg            | `… Example models/Gemma3-270M` or `… Example models/LFM2.5-2.6B-Q4_K_M.gguf`                      |
| System property    | `-Dnanollvm.model=models/Gemma3-270M`                                                             |
| Force architecture | `-Dnanollvm.arch=gemma3` or `gemma4` (must match the checkpoint)                              |
| Env                | `NANOLLVM_MODEL=models/Gemma3-270M`                                                               |
| Models root        | `-Dnanollvm.models.dir=/other/models` or `NANOLLVM_MODELS_DIR` (default loads `<dir>/Qwen3-0.6B`) |

Architecture is detected exactly from `config.json` / GGUF `general.architecture` (`ModelSupport`).
`-Dnanollvm.arch` may only confirm a matching family (`qwen3`, `gemma3`, `gemma4`, `llama`, `lfm2`, `bert`, `whisper`, `piper`);
look-alike checkpoints fail with `UnsupportedModelException`.
