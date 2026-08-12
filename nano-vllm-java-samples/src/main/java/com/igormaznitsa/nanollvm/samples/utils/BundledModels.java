package com.igormaznitsa.nanollvm.samples.utils;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODELS_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODELS_DIR;

import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Resolves on-disk HuggingFace model directories ({@code .safetensors} or {@code .onnx}) or
 * {@code .gguf} files. Default root is {@code ./models} (see {@code models/download-*.sh}).
 */
public final class BundledModels {

  public static final String DEFAULT_MODELS_DIR = "models";
  public static final String DEFAULT_MODEL_NAME = "Qwen3-0.6B";
  public static final String QWEN3_0_6B = DEFAULT_MODELS_DIR + "/" + DEFAULT_MODEL_NAME;
  public static final String GEMMA3_270M = DEFAULT_MODELS_DIR + "/Gemma3-270M";
  public static final String LFM2_5_2_6B_GGUF =
    DEFAULT_MODELS_DIR + "/LFM2.5-2.6B-Q4_K_M.gguf";
  /**
   * @since 1.1.0
   */
  public static final String GTE_SMALL_GGUF =
    DEFAULT_MODELS_DIR + "/gte-small.Q2_K.gguf";
  /**
   * @since 1.1.0
   */
  public static final String TINY_LLM_ONNX = DEFAULT_MODELS_DIR + "/Tiny-LLM-ONNX";
  /**
   * @since 1.1.0
   */
  public static final String SMOLLM2_135M_INSTRUCT_ONNX =
    DEFAULT_MODELS_DIR + "/SmolLM2-135M-Instruct-ONNX";

  private BundledModels() {
  }

  /**
   * Resolution order: CLI path → {@code -Dnanollvm.model} → {@code NANOLLVM_MODEL}
   * → {@code <modelsRoot>/Qwen3-0.6B}.
   */
  public static Path resolveDefault(String... cliArgs) {
    if (cliArgs != null && cliArgs.length > 0 && cliArgs[0] != null && !cliArgs[0].isBlank()) {
      return Path.of(cliArgs[0]).toAbsolutePath().normalize();
    }
    String prop = NanoLlvmProps.systemProperty(PROP_MODEL);
    if (prop != null) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = NanoLlvmProps.environment(ENV_MODEL);
    if (env != null) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return require(QWEN3_0_6B);
  }

  /**
   * Models root: {@code -Dnanollvm.models.dir} → {@code NANOLLVM_MODELS_DIR} → {@code ./models}.
   */
  public static Path modelsRoot() {
    String prop = NanoLlvmProps.systemProperty(PROP_MODELS_DIR);
    if (prop != null) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = NanoLlvmProps.environment(ENV_MODELS_DIR);
    if (env != null) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return Path.of("").toAbsolutePath().normalize().resolve(DEFAULT_MODELS_DIR);
  }

  public static Path require(final String modelPathOrName) {
    return find(modelPathOrName).orElseThrow(() -> new IllegalStateException(
      "model not found: " + modelPathOrName
        + " (expected under " + modelsRoot()
        + "). Run models/download-qwen3-0.6b.sh, models/download-gemma3-270m.sh, "
        + "models/download-lfm2.5-2.6b-gguf.sh, models/download-gte-small-gguf.sh, "
        + "or models/download-tiny-llm-onnx.sh, "
        + "models/download-smollm2-135m-instruct-onnx.sh, "
        + "or pass a model path / -D" + PROP_MODEL + "=… / " + ENV_MODEL + "."
    ));
  }

  /**
   * @param modelPathOrName absolute/relative path, {@code Qwen3-0.6B}, {@code Gemma3-270M},
   *                        {@code Tiny-LLM-ONNX}, {@code SmolLM2-135M-Instruct-ONNX},
   *                        {@code LFM2.5-2.6B-Q4_K_M.gguf}, or {@code models/…}
   */
  public static Optional<Path> find(final String modelPathOrName) {
    if (modelPathOrName == null || modelPathOrName.isBlank()) {
      return Optional.empty();
    }
    String key = modelPathOrName.strip();
    if (key.startsWith("/")) {
      key = key.substring(1);
    }

    Path asPath = Path.of(key);
    if (asPath.isAbsolute() && isModel(asPath)) {
      return Optional.of(asPath.normalize());
    }

    Path cwd = Path.of("").toAbsolutePath().normalize();
    Path root = modelsRoot();
    String bare = stripModelsPrefix(key);

    Path[] candidates = {
      root.resolve(bare),
      cwd.resolve(key),
      cwd.resolve(DEFAULT_MODELS_DIR).resolve(bare),
    };
    for (Path candidate : candidates) {
      if (isModel(candidate)) {
        return Optional.of(candidate.toAbsolutePath().normalize());
      }
    }
    return Optional.empty();
  }

  private static String stripModelsPrefix(final String key) {
    if (key.startsWith(DEFAULT_MODELS_DIR + "/")) {
      return key.substring(DEFAULT_MODELS_DIR.length() + 1);
    }
    return key;
  }

  private static boolean isModel(final Path path) {
    return isGgufFile(path) || isHfModelDir(path) || isDirWithGguf(path);
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
      return stream.anyMatch(BundledModels::isGgufFile);
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
