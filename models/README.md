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

## Gemma3-270M (optional)

Instruction-tuned text Gemma 3 (270M). **License-gated** on Hugging Face — accept terms at
https://huggingface.co/google/gemma-3-270m-it then authenticate (`huggingface-cli login` or `HF_TOKEN`).

```bash
export HF_TOKEN=hf_...   # if needed
./models/download-gemma3-270m.sh
```

Windows: `.\models\download-gemma3-270m.ps1` or `models\download-gemma3-270m.cmd`.

Creates `models/Gemma3-270M/`.

## LFM2.5-2.6B GGUF (optional)

Hybrid Liquid LFM2 checkpoint as a single GGUF file. Weights stay packed in RAM (dequant on matmul); default
**~16 GB heap** in [`.mvn/jvm.config`](../.mvn/jvm.config) is safe headroom for KV / activations.

```bash
./models/download-lfm2.5-2.6b-gguf.sh
```

Windows: `.\models\download-lfm2.5-2.6b-gguf.ps1` or `models\download-lfm2.5-2.6b-gguf.cmd`.

Creates `models/LFM2.5-2.6B-Q4_K_M.gguf`.

## Selecting a model

| Mechanism          | Example                                                                                           |
|--------------------|---------------------------------------------------------------------------------------------------|
| CLI arg            | `… Example models/Gemma3-270M` or `… Example models/LFM2.5-2.6B-Q4_K_M.gguf`                      |
| System property    | `-Dnanovllm.model=models/Gemma3-270M`                                                             |
| Force architecture | `-Dnanovllm.arch=gemma3` (or `qwen3` / `lfm2`)                                                    |
| Env                | `NANOVLLM_MODEL=models/Gemma3-270M`                                                               |
| Models root        | `-Dnanovllm.models.dir=/other/models` or `NANOVLLM_MODELS_DIR` (default loads `<dir>/Qwen3-0.6B`) |

Architecture is auto-detected from `config.json` or GGUF `general.architecture` unless `-Dnanovllm.arch` is set.
