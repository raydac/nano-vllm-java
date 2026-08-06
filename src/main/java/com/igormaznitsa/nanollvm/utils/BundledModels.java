package com.igormaznitsa.nanollvm.utils;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.CONFIG_JSON;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.ENV_MODELS_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.PROP_MODELS_DIR;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/**
 * Resolves on-disk HuggingFace model directories.
 * Default root is {@code ./models} (see {@code models/download-*.sh}).
 */
public final class BundledModels {

  public static final String DEFAULT_MODELS_DIR = "models";
  public static final String DEFAULT_MODEL_NAME = "Qwen3-0.6B";
  public static final String QWEN3_0_6B = DEFAULT_MODELS_DIR + "/" + DEFAULT_MODEL_NAME;
  public static final String GEMMA3_270M = DEFAULT_MODELS_DIR + "/Gemma3-270M";

  private BundledModels() {
  }

  /**
   * Resolution order: CLI path → {@code -Dnanovllm.model} → {@code NANOVLLM_MODEL}
   * → {@code <modelsRoot>/Qwen3-0.6B}.
   */
  public static Path resolveDefault(String... cliArgs) {
    if (cliArgs != null && cliArgs.length > 0 && cliArgs[0] != null && !cliArgs[0].isBlank()) {
      return Path.of(cliArgs[0]).toAbsolutePath().normalize();
    }
    String prop = System.getProperty(PROP_MODEL);
    if (prop != null && !prop.isBlank()) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = System.getenv(ENV_MODEL);
    if (env != null && !env.isBlank()) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return require(QWEN3_0_6B);
  }

  /**
   * Models root: {@code -Dnanovllm.models.dir} → {@code NANOVLLM_MODELS_DIR} → {@code ./models}.
   */
  public static Path modelsRoot() {
    String prop = System.getProperty(PROP_MODELS_DIR);
    if (prop != null && !prop.isBlank()) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = System.getenv(ENV_MODELS_DIR);
    if (env != null && !env.isBlank()) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return Path.of("").toAbsolutePath().normalize().resolve(DEFAULT_MODELS_DIR);
  }

  public static Path require(final String modelPathOrName) {
    return find(modelPathOrName).orElseThrow(() -> new IllegalStateException(
        "model not found: " + modelPathOrName
            + " (expected under " + modelsRoot()
            + "). Run models/download-qwen3-0.6b.sh or models/download-gemma3-270m.sh, "
            + "or pass a model path / -D" + PROP_MODEL + "=… / " + ENV_MODEL + "."
    ));
  }

  /**
   * @param modelPathOrName absolute/relative path, {@code Qwen3-0.6B}, {@code Gemma3-270M},
   *                        or {@code models/…}
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
    if (asPath.isAbsolute() && isModelDir(asPath)) {
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
      if (isModelDir(candidate)) {
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

  private static boolean isModelDir(final Path dir) {
    return Files.isDirectory(dir)
        && Files.isRegularFile(dir.resolve(CONFIG_JSON))
        && hasSafetensors(dir);
  }

  private static boolean hasSafetensors(final Path dir) {
    try (var stream = Files.list(dir)) {
      return stream.anyMatch(p -> p.getFileName().toString().endsWith(".safetensors"));
    } catch (Exception e) {
      return false;
    }
  }
}
