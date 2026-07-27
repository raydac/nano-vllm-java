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

## Selecting a model

| Mechanism          | Example                                                                                           |
|--------------------|---------------------------------------------------------------------------------------------------|
| CLI arg            | `… Example models/Gemma3-270M`                                                                    |
| System property    | `-Dnanovllm.model=models/Gemma3-270M`                                                             |
| Force architecture | `-Dnanovllm.arch=gemma3` (or `qwen3`)                                                             |
| Env                | `NANOVLLM_MODEL=models/Gemma3-270M`                                                               |
| Models root        | `-Dnanovllm.models.dir=/other/models` or `NANOVLLM_MODELS_DIR` (default loads `<dir>/Qwen3-0.6B`) |

Architecture is auto-detected from `config.json` (`model_type` / `architectures`) unless `-Dnanovllm.arch` is set.
