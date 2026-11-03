![Banner](assets/banner.png)

[![License Apache 2.0](https://img.shields.io/badge/license-Apache%20License%202.0-green.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Java 21+](https://img.shields.io/badge/java-21.0%2b-green.svg)](https://bell-sw.com/pages/downloads/)
[![Maven 3.8+](https://img.shields.io/badge/maven-3.8%2b-green.svg)](https://maven.apache.org/)
[![Maven central](https://img.shields.io/badge/Maven%20central-1.4.0-green.svg)](http://search.maven.org/#artifactdetails|com.igormaznitsa|nano-vllm-java|1.4.0|jar)
[![Arthur's acres sanctuary donation](assets/arthur_sanctuary_banner.png)](https://www.arthursacresanimalsanctuary.org/donate)

# Nano-vLLM Java

A Java library that runs a pretrained model on the CPU and returns an answer. You add a Maven or Gradle dependency and call it from ordinary Java. The call stays in the JVM: no Python process, no CUDA, and no PyTorch.

It does not train models. It loads files you already have and runs inference.

**For** a Java application that needs a small local model: a short chat reply, a sentence vector, a transcript, a spoken WAV, a language label, or a choice with a probability.

**Supports** Qwen3, Qwen3.5 text, Gemma 3, Gemma 4 text, Llama, LFM2, BERT embeddings, Whisper, Piper, and fastText. The full list is [below](#supported-models).

**Attach** `com.igormaznitsa:nano-vllm-java:1.4.0` on JDK 21+. Coordinates and a first call are under [Add it to an application](#add-it-to-an-application).

**Try a sample** from this repository after one download. Qwen3-0.6B needs no Hugging Face gate:

```bash
./models/download-qwen3-0.6b.sh
mvn -pl nano-vllm-java-samples -q exec:java
```

That opens the interactive menu (chat, search, typed decisions). Every other sample, including weather and Russian checkers, is under [Run the samples](#run-the-samples).

**1.4.0** is the release on Maven Central. This repository is **1.5.0-SNAPSHOT** and also includes Qwen3.5 text, typed decisions (Jev), the checkers sample, and a weather sample. The longer design tour (scheduler, attention, tensors, formats) is [`description.md`](description.md).

Ideas here were inspired by the Python [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) educational engine.

## Contents

- [What it is for](#what-it-is-for)
- [Supported models](#supported-models)
- [Add it to an application](#add-it-to-an-application)
- [Chat](#chat)
- [Run the samples](#run-the-samples)
- [Other calls](#other-calls)
- [Threads](#threads)
- [Limits](#limits)
- [Build from source](#build-from-source)

## What it is for

| You have | You get | Call |
|----------|---------|------|
| A question or a log snippet | A text reply | `llm.chatOnce(…)` or `llm.chat(…).send(…).answer()` |
| A sentence | A vector (for search) | `generate(text, EMBEDDING)` |
| A WAV clip | A transcript | `generate(sound, TEXT)` |
| A sentence | A WAV file | `generate(text, AUDIO)` |
| A sentence | Ranked labels (language id, and similar) | `generate(text, LABELS)` |
| A situation and a fixed list of options | The chosen option and a probability for each | `jev.decide(…)` (snapshot) |
| Your own notes on disk | A reply that can quote those notes | `llm.rag(…).ask(…)` |

One `LLM` runs one call at a time. Share the loaded `LlmModel` across engines when calls should overlap. The rules are in [Threads](#threads).

## Supported models

`LlmModelFactory.make(path)` accepts a Hugging Face folder, a `.gguf` file, or a fastText `*.bin` / `*.ftz`. This is a curated set of families.

| Family | Typical file | Job |
|--------|----------------|-----|
| Qwen3, Gemma 3, Llama | Safetensors folder, or ONNX weights in that folder | Chat |
| Qwen3 | Also a `.gguf` file | Chat |
| Qwen3.5 text | Safetensors only (snapshot). The vision tower is skipped | Chat, and Jev |
| Gemma 4 text (QAT mobile) | Safetensors only | Chat |
| LFM2 | `.gguf` only | Chat |
| BERT (gte, E5, XLM-RoBERTa, …) | `.gguf` or ONNX | Embeddings |
| Whisper | Safetensors folder | Speech to text |
| Piper | `*.onnx` + `*.onnx.json` | Text to speech |
| fastText | `*.bin` or `*.ftz` | Labels |

If a folder has both safetensors and ONNX, safetensors is used. An unsupported file fails with `UnsupportedModelException` and the list of families this build knows.

Download scripts and the folder each one creates are in [`models/README.md`](models/README.md). A good first chat model with no Hugging Face gate is Qwen3-0.6B (`./models/download-qwen3-0.6b.sh`). Gemma3-270M is smaller and needs you to accept the model license on Hugging Face first (`./models/download-gemma3-270m.sh`).

## Add it to an application

JDK **21+**. A small chat model (Gemma3-270M, Qwen3-0.6B) wants a few gigabytes of heap. Larger checkpoints want more. The in-repo demos expect about 16 GB.

**Maven**

```xml
<dependency>
  <groupId>com.igormaznitsa</groupId>
  <artifactId>nano-vllm-java</artifactId>
  <version>1.4.0</version>
</dependency>
```

**Gradle**

```gradle
implementation 'com.igormaznitsa:nano-vllm-java:1.4.0'
```

On the module path: `requires com.igormaznitsa.nanollvm;`.

Weights are not in the published jar and not in this git tree. Point `LlmModelFactory` at a folder you downloaded (`config.json`, tokenizer, `*.safetensors`) or at a single `.gguf` file.

## Chat

```java
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;

import java.nio.file.Path;

Path modelDir = Path.of("/opt/models/Gemma3-270M");

try (LlmModel model = LlmModelFactory.make(modelDir);
     LLM llm = LLM.builder(model)
         .noSystemPrompt()   // Gemma chat path; Qwen can take .systemPrompt("…")
         .build()) {

  String reply = llm.chatOnce("""
      Read the log and reply in three short bullets: what failed, likely cause, one next check.
      Do not invent hosts or error codes that are not in the log.

      2026-08-07 22:14:08 ERROR payment-api - give up order=99102 after 3 timeouts upstream=billing-svc:8443
      """);
  System.out.println(reply);
}
```

The same program is `LogTriageHelloWorld` in this repo (default folder `models/Gemma3-270M`):

```bash
mvn -pl nano-vllm-java-samples -q exec:java \
  -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.LogTriageHelloWorld
```

## Run the samples

Run from the repository root after the matching download in [`models/README.md`](models/README.md). An optional first argument overrides the model path (`-Dexec.args=…`).

Any row in the table:

```bash
mvn -pl nano-vllm-java-samples -q exec:java \
  -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.HelloWorld
```

Swap the class name. `Example` is the default main, so this is enough after a download:

```bash
mvn -pl nano-vllm-java-samples -q exec:java
```

| Sample | What it shows | Default model |
|--------|----------------|---------------|
| `HelloWorld` | One hello, one reply | Gemma3-270M |
| `ToolsHelloWorld` | OS name and a directory listing via tools | Qwen3-0.6B |
| `LogTriageHelloWorld` | Three bullets from a log excerpt | Gemma3-270M |
| `NextTokenHelloWorld` | Continue a seed | Tiny-LLM-ONNX |
| `EmbeddingsHelloWorld` | Sentence vector | multilingual-e5-small |
| `TranscribeHelloWorld` | WAV or a tone → text | whisper-base |
| `SynthesizeHelloWorld` | Text → `piper-hello.wav` | piper-en-lessac-medium |
| `VoiceReplyHelloWorld` | Transcribe, then chat, then speak | whisper + chat + piper |
| `LanguageIdHelloWorld` | Language label | fasttext lid.176 |
| `AdvisorRagHelloWorld` | Named advisors plus search over `rag/` | Gemma3-270M |
| `RagTunerHelloWorld` | Read a bundled EPUB, then ask about the play | Qwen3-0.6B |
| `JevHelloWorld` | Choice, yes/no, and a score (snapshot) | Gemma3-270M |
| `RussianCheckersHelloWorld` | Russian draughts in the terminal; you pick a colour (snapshot) | Jeff-Qwen3.5-0.8B |
| `Example` | Interactive menu: chat, search, Jev | first model on disk |
| `Bench` | Throughput smoke test | Qwen3-0.6B |

### Interactive menu

`Example` lists the models it finds, then chats. `/exit` leaves. `/clear` drops the conversation and keeps the document index. `NO_COLOR=1` turns ANSI colors off.

![Example: pick Qwen3 or Gemma 3, then a Gemma 3 chat about the Grimm brothers](assets/java_nano_llvm_session_1.gif)

*Pick a model, then ask. This run loads Gemma 3 and asks about the Grimm brothers.*

### Weather

`nano-vllm-java-weather-sample` is a separate module. Qwen3-0.6B asks where this computer is, then what the weather is there. It prints the temperature and the sky, and Piper speaks that line on the default audio output. Once the coordinates are known it prints a 64-column world map, Kavrayskiy VII over the Natural Earth coastline, with that position marked. The log gives the wall-clock time of each step.

If the chat model or a Piper voice is missing, the sample stops before loading and prints the download command for each one. The two questions stay separate because this checkpoint calls one tool when a question names both. Gemma 3 has no tool-call tokens. Qwen3.5, LFM2, and Gemma 4 can replace it via the first argument.

The place is this computer's public network address (ipwho.is, then geojs, then ipapi.co). The weather is [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). Neither needs an API key.

```bash
./models/download-qwen3-0.6b.sh
./models/download-piper-en-lessac-medium.sh
mvn -pl nano-vllm-java-weather-sample -am -q exec:java
```

The Russian Irina voice can stand in for Lessac. Windows download scripts are in [`models/README.md`](models/README.md).

![Weather sample: Qwen3 locates this computer, prints the map and the forecast, then Piper speaks the line](assets/weatherstation_03nov2027.gif)

*Qwen3 locates this computer, prints the map and the forecast, then Piper speaks the line.*

### Russian checkers

You pick a colour. Enter selects White. Jeff (Qwen3.5) plays the other colour. A move is `a3 b4` or `A 3 B 4`. `q` quits.

```bash
./models/download-jeff-qwen3.5-0.8b.sh
mvn -pl nano-vllm-java-samples -q exec:java \
  -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.RussianCheckersHelloWorld
```

![Russian checkers: you play White, Jeff answers with Qwen3.5, and the game ends when your king is taken](assets/russian_checkers_03nov26.gif)

*You play White. Jeff answers with Qwen3.5. The game ends when your king is taken.*

## Other calls

Each snippet loads a model the same way as chat, then asks for a different result. `generate` returns the concrete type for that job (`LlmOutText`, `LlmOutEmbedding`, `LlmOutSoundData`, `LlmOutLabels`).

**Sentence vector** — download with `./models/download-multilingual-e5-small.sh`.

```java
LlmOutEmbedding vector = llm.generate(
    LlmInText.of("query: hello world"), LlmModality.EMBEDDING);
```

**Speech to text** — mono PCM in `[-1, 1]`, or an uncompressed WAV. `./models/download-whisper-base.sh`.

```java
LlmOutText transcript = llm.generate(
    LlmInSound.ofWav(Files.readAllBytes(Path.of("clip.wav"))), LlmModality.TEXT);
```

**Text to speech** — PCM16 WAV bytes. `./models/download-piper-en-lessac-medium.sh`.

```java
Path voice = Path.of("models/piper-en-lessac-medium");
try (LlmModel model = LlmModelFactory.open(voice)
         .optionalData(LlmOptionalData.ESPEAK_DATA, voice.resolve("espeak-ng-data"))
         .make();
     LLM llm = LLM.builder(model).build()) {
  byte[] wav = llm.generate(LlmInText.of("Hello world"), LlmModality.AUDIO).wav();
}
```

**Language id** — `./models/download-fasttext-lid-176.sh`.

```java
LlmOutLabels labels = llm.generate(
    LlmInText.of("Bonjour, comment allez-vous ?"), LlmModality.LABELS);
System.out.println(labels.topLabel());
```

**A choice with probabilities** (this snapshot, any causal chat model). Nothing is parsed from the reply; the engine reads the next-token scores.

```java
try (LlmModel model = LlmModelFactory.make(Path.of("models/Gemma3-270M"));
     Jev jev = LLM.builder(model).toJev()) {
  JevDecision route = jev.decide(
      "My card was charged twice for one order.",
      JevQuestion.choice("Which team should handle this?",
          List.of("billing", "technical", "sales", "other")));
  System.out.println(route.answer());
}
```

**Notes beside the model.** UTF-8 `.txt` / `.md` files. The model still writes the answer; retrieval only puts passages into the prompt.

```java
PreparedRag notes = RagFactory.builder()
    .addFolders(Path.of("docs"))
    .add("hours", "Live chat is available 9–17 UTC.")
    .build();

String answer = llm.rag(notes).topK(4).ask("When is live chat open?");
```

## Threads

Weights are shared. A conversation is not. Load the model once, then give each concurrent caller its own engine.

| Object | Across threads |
|--------|----------------|
| `LlmModel` | Yes. Weights, tokenizer, and load options stay frozen. Many `LLM` engines bind the same model. |
| `PreparedRag`, a `RagIndex` | Yes. `retrieve` from many threads. A dense index keeps the embedding model open and embeds each query on its own. |
| `LLM` | One call at a time. A second `generate`, chat, RAG ask, or `jev.decide` waits, then runs, and returns its own result. |
| `ChatSession`, `RagSession`, `Jev` | No. One thread each. History, and a Jev level-0 prior, live on that object. |

Two chats at once are two engines on one model. Each engine has its own KV cache, so two engines use about twice the cache RAM.

```java
Path modelDir = Path.of("/opt/models/Qwen3-0.6B");

try (LlmModel model = LlmModelFactory.make(modelDir);
     LLM first = LLM.builder(model).build();
     LLM second = LLM.builder(model).build()) {

  ChatSession left = first.chat();
  ChatSession right = second.chat();
  // left.send(...) and right.send(...) may run together
}
```

Close every `LLM` before `model.close()`. Closing an engine does not unload the weights. `Jev.close()` closes its engine and leaves the model open.

One shared `LLM` is a queue. Several `ChatSession`s may sit on it; their turns run one after another. A tool-calling turn lets the next waiter in between model calls. Each waiter still gets its own reply.

`cancel()` belongs to the engine, not to the caller. It aborts whoever is generating on that `LLM` (`GenerationCancelledException`). A cancel that arrives before a call has started is dropped when the call starts. `close()` from any thread cancels the in-flight call; a later call throws `IllegalStateException`.

A token callback runs inside the call that is still generating. Do not call `generate`, chat, or RAG from that callback.

Pass `Builder.random` only to one engine. Two concurrent engines need two `Random` instances.

With `cpuThreads` above 1, engines share a process-wide pool named `nanollvm-matmul-*` unless you change it. `matmulExecutor` uses a pool you own and shut down. `dedicatedMatmulPool` uses a pool that closes with that engine. `disableMultiCpu` keeps the work on the calling thread.

## Limits

- **CPU, small models.** Answers are slower than a GPU server, and a 270M–0.8B model will miss facts, waffle, or invent a detail. Search over your notes can supply a passage; the model can still garble it.
- **Inference only.** The files do not change while you chat.
- **Text in, and the jobs in the table above.** No image or audio understanding beyond Whisper. No image generation. Qwen3.5 vision, LoRA adapters, mixture-of-experts, and Qwen3-Next are refused.
- **Whisper** reads uncompressed WAV or PCM. No MP3, no timestamps, no voice-activity detection.
- **Piper** returns one WAV per call. No streaming audio and no SSML.
- **GGUF chat** is Qwen3 and LFM2. **GGUF embeddings** are BERT. Other GGUF architectures are out.
- **ONNX** contributes weights to the Java engine. Quantized community exports and graphs that need ONNX Runtime are skipped.
- **Documents for search** are UTF-8 text. PDF and EPUB need a small extractor you register (`RagTunerHelloWorld` shows an EPUB).
- **One call at a time per `LLM`.** Overlap needs one engine per caller on a shared `LlmModel`. See [Threads](#threads).

Format tables, quant types, and the engine internals are in [`description.md`](description.md), especially chapters **7–7g** and **19**.

## Build from source

JDK 21+ and Maven 3.8+, from the repository root:

```bash
mvn test
mvn package
```

The library jar is `nano-vllm-java/target/nano-vllm-java-1.5.0-SNAPSHOT.jar` (module `com.igormaznitsa.nanollvm`). Samples stay in `nano-vllm-java-samples` and are not published. Tests skip when a local checkpoint is missing.

## Further reading

| Doc | What is there |
|-----|----------------|
| [`description.md`](description.md) | How the engine works: attention, tensors, scheduler, formats, RAG, honest limits |
| [`models/README.md`](models/README.md) | Every download script and the folder it creates |
| [`CHANGELOG.md`](CHANGELOG.md) | What each release added |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
