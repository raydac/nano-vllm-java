package com.igormaznitsa.nanollvm.testsupport;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODELS_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_RAG_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODELS_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_RAG_DIR;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Test-only helper: skip resource-dependent tests when optional local weights or corpus are absent.
 * Not part of the published library — models are never shipped with the JAR.
 */
public final class OptionalModelAssumptions {

  private static final String MODELS_DIR = "models";
  private static final String RAG_DIR = "rag";
  private static final String QWEN3_0_6B = "Qwen3-0.6B";
  private static final String GEMMA3_270M = "Gemma3-270M";
  private static final String GEMMA4_E2B_IT_QAT_MOBILE = "Gemma4-E2B-IT-QAT-Mobile";
  private static final String LFM2_5_2_6B_GGUF = "LFM2.5-2.6B-Q4_K_M.gguf";
  private static final String GTE_SMALL_GGUF = "gte-small.Q2_K.gguf";
  private static final String MULTILINGUAL_E5_SMALL = "multilingual-e5-small";
  private static final String SMOLLM2_135M_INSTRUCT_ONNX = "SmolLM2-135M-Instruct-ONNX";
  private static final String WHISPER_BASE = "whisper-base";
  private static final String WHISPER_TINY = "whisper-tiny";
  private static final String PIPER_RU_IRINA_MEDIUM = "piper-ru-irina-medium";
  private static final String PIPER_EN_LESSAC_MEDIUM = "piper-en-lessac-medium";
  private static final String FASTTEXT_LID_176 = "fasttext-lid-176";

  private OptionalModelAssumptions() {
  }

  public static Path requireQwen3() {
    return require(
      findModel(QWEN3_0_6B),
      "Qwen3-0.6B",
      "models/download-qwen3-0.6b.sh");
  }

  public static Path requireGemma3() {
    return require(
      findModel(GEMMA3_270M),
      "Gemma3-270M",
      "models/download-gemma3-270m.sh (HF license + HF_TOKEN)");
  }

  public static Path requireGemma4E2bQatMobile() {
    return require(
      findModel(GEMMA4_E2B_IT_QAT_MOBILE),
      "Gemma4-E2B-IT-QAT-Mobile",
      "models/download-gemma4-e2b-qat-mobile.sh");
  }

  public static Path requireLfm2Gguf() {
    return require(
      findModel(LFM2_5_2_6B_GGUF),
      "LFM2.5-2.6B GGUF",
      "models/download-lfm2.5-2.6b-gguf.sh");
  }

  public static Path requireGteSmallGguf() {
    return require(
      findModel(GTE_SMALL_GGUF),
      "gte-small GGUF",
      "models/download-gte-small-gguf.sh");
  }

  public static Path requireMultilingualE5Small() {
    return require(
      findModel(MULTILINGUAL_E5_SMALL),
      "multilingual-e5-small ONNX",
      "models/download-multilingual-e5-small.sh");
  }

  public static Path requireSmolLm2InstructOnnx() {
    return require(
      findModel(SMOLLM2_135M_INSTRUCT_ONNX),
      "SmolLM2-135M-Instruct-ONNX",
      "models/download-smollm2-135m-instruct-onnx.sh");
  }

  public static Path requireWhisper() {
    Optional<Path> base = findModel(WHISPER_BASE);
    if (base.isPresent()) {
      return base.get();
    }
    return require(
      findModel(WHISPER_TINY),
      "whisper-base or whisper-tiny",
      "models/download-whisper-base.sh or models/download-whisper-tiny.sh");
  }

  public static Path requirePiper() {
    Optional<Path> english = findModel(PIPER_EN_LESSAC_MEDIUM);
    if (english.isPresent()) {
      return english.get();
    }
    Optional<Path> russian = findModel(PIPER_RU_IRINA_MEDIUM);
    if (russian.isPresent()) {
      return russian.get();
    }
    return require(
        Optional.empty(),
        "piper-en-lessac-medium or piper-ru-irina-medium",
        "models/download-piper-en-lessac-medium.sh or models/download-piper-ru-irina-medium.sh");
  }

  public static Path requirePiperRussian() {
    return require(
      findModel(PIPER_RU_IRINA_MEDIUM),
      "piper-ru-irina-medium",
      "models/download-piper-ru-irina-medium.sh");
  }

  public static Path requireFastTextLid() {
    return require(
      findModel(FASTTEXT_LID_176),
      "fasttext-lid-176",
      "models/download-fasttext-lid-176.sh");
  }

  public static Path requireLocalRag() {
    return require(
      findRag(),
      "local RAG corpus at " + ragRoot(),
      "create ./rag with corpus files or set -Dnanollvm.rag.dir=…");
  }

  public static Path require(
    final Optional<Path> path,
    final String label,
    final String downloadHint
  ) {
    if (path.isPresent()) {
      return path.get();
    }
    String message = "Skipping test: %s not available (%s)".formatted(label, downloadHint);
    System.err.println("[nanollvm] WARNING: " + message);
    assumeTrue(false, message);
    throw new IllegalStateException("unreachable");
  }

  private static Optional<Path> findModel(final String bareName) {
    Path candidate = modelsRoot().resolve(bareName);
    return isModel(candidate) ? Optional.of(candidate.toAbsolutePath().normalize()) :
      Optional.empty();
  }

  private static Optional<Path> findRag() {
    Path root = ragRoot();
    return Files.isDirectory(root) ? Optional.of(root) : Optional.empty();
  }

  private static Path modelsRoot() {
    String prop = NanoLlvmProps.systemProperty(PROP_MODELS_DIR);
    if (prop != null) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = NanoLlvmProps.environment(ENV_MODELS_DIR);
    if (env != null) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return Path.of("").toAbsolutePath().normalize().resolve(MODELS_DIR);
  }

  private static Path ragRoot() {
    String prop = NanoLlvmProps.systemProperty(PROP_RAG_DIR);
    if (prop != null) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = NanoLlvmProps.environment(ENV_RAG_DIR);
    if (env != null) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return Path.of("").toAbsolutePath().normalize().resolve(RAG_DIR);
  }

  private static boolean isModel(final Path path) {
    return isGgufFile(path) || isHfModelDir(path) || isDirWithGguf(path)
      || ModelSupport.isSynthesisCheckpoint(path)
      || ModelSupport.isClassificationCheckpoint(path);
  }

  private static boolean isGgufFile(final Path path) {
    return Files.isRegularFile(path)
      && path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
  }

  private static boolean isHfModelDir(final Path dir) {
    return Files.isDirectory(dir)
      && Files.isRegularFile(dir.resolve(CONFIG_JSON))
      && (hasSafetensors(dir) || hasOnnx(dir));
  }

  private static boolean isDirWithGguf(final Path dir) {
    if (!Files.isDirectory(dir)) {
      return false;
    }
    try (Stream<Path> stream = Files.list(dir)) {
      return stream.anyMatch(OptionalModelAssumptions::isGgufFile);
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasSafetensors(final Path dir) {
    try (var stream = Files.list(dir)) {
      return stream.anyMatch(p -> p.getFileName().toString().endsWith(".safetensors"));
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasOnnx(final Path dir) {
    if (Files.isRegularFile(dir.resolve("model.onnx"))
      || Files.isRegularFile(dir.resolve("model_fp16.onnx"))
      || Files.isRegularFile(dir.resolve("onnx").resolve("model.onnx"))
      || Files.isRegularFile(dir.resolve("onnx").resolve("model_fp16.onnx"))) {
      return true;
    }
    try (var stream = Files.list(dir)) {
      if (stream.anyMatch(
        p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".onnx"))) {
        return true;
      }
    } catch (Exception ignored) {
      // fall through
    }
    Path onnxDir = dir.resolve("onnx");
    if (!Files.isDirectory(onnxDir)) {
      return false;
    }
    try (var stream = Files.list(onnxDir)) {
      return stream.anyMatch(
        p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".onnx"));
    } catch (Exception e) {
      return false;
    }
  }
}
