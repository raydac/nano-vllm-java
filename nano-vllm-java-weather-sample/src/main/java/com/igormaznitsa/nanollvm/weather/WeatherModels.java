package com.igormaznitsa.nanollvm.weather;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.ENV_MODELS_DIR;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODEL;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_MODELS_DIR;

import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Chat checkpoint and Piper voice under {@code models/}.
 * Default chat model is Qwen3-0.6B: Gemma 3 has no tool-call tokens, and this is the
 * smallest checkpoint in the repo that emits them.
 */
public final class WeatherModels {

  private WeatherModels() {
  }

  public static Required require(final String[] args) {
    List<String> missing = new ArrayList<>();
    Optional<Path> chat = foundOrNotice(() -> chatModel(args), missing);
    Optional<Path> voice = foundOrNotice(WeatherModels::piperVoice, missing);
    if (!missing.isEmpty()) {
      throw new MissingModels("This sample cannot start.\n\n" + String.join("\n\n", missing));
    }
    return new Required(chat.orElseThrow(), voice.orElseThrow());
  }

  static Path chatModel(final String[] args) {
    Path absolute = explicit(args).orElseGet(WeatherModels::configuredOrDefault)
      .toAbsolutePath()
      .normalize();
    if (!isChatCheckpoint(absolute)) {
      throw new MissingModels(chatNotice(absolute));
    }
    return absolute;
  }

  static Path piperVoice() {
    Path root = modelsRoot();
    Path lessac = root.resolve("piper-en-lessac-medium");
    if (ModelSupport.isSynthesisCheckpoint(lessac)) {
      return lessac.toAbsolutePath().normalize();
    }
    throw new MissingModels(voiceNotice(root));
  }

  static Path modelsRoot() {
    String prop = NanoLlvmProps.systemProperty(PROP_MODELS_DIR);
    if (prop != null) {
      return Path.of(prop).toAbsolutePath().normalize();
    }
    String env = NanoLlvmProps.environment(ENV_MODELS_DIR);
    if (env != null) {
      return Path.of(env).toAbsolutePath().normalize();
    }
    return Path.of("").toAbsolutePath().normalize().resolve("models");
  }

  private static Optional<Path> foundOrNotice(
    final Supplier<Path> lookup,
    final List<String> missing
  ) {
    try {
      return Optional.of(lookup.get());
    } catch (MissingModels ex) {
      missing.add(ex.getMessage());
      return Optional.empty();
    }
  }

  private static Optional<Path> explicit(final String[] args) {
    if (args == null || args.length == 0 || args[0] == null || args[0].isBlank()) {
      return Optional.empty();
    }
    return Optional.of(Path.of(args[0].strip()));
  }

  private static String chatNotice(final Path path) {
    String problem = Files.exists(path)
      ? "Chat model is not a checkpoint: " + path
      + "\nExpected a folder with config.json and weight files, or a .gguf file."
      : "Chat model was not found: " + path;
    return """
      %s
      Download the default checkpoint with models/download-qwen3-0.6b.sh.
      A tool-capable checkpoint can also be passed as the first argument (Qwen3, Qwen3.5, LFM2, or Gemma 4)."""
      .formatted(problem)
      .strip();
  }

  private static String voiceNotice(final Path root) {
    return """
      Piper voice was not found under %s
      Download models/download-piper-en-lessac-medium.sh.""".formatted(root).strip();
  }

  private static boolean isChatCheckpoint(final Path path) {
    if (!Files.exists(path)) {
      return false;
    }
    if (isGguf(path)) {
      return true;
    }
    if (!Files.isDirectory(path)) {
      return false;
    }
    if (hasWeight(path, ".gguf")) {
      return true;
    }
    return Files.isRegularFile(path.resolve("config.json"))
      && (hasWeight(path, ".safetensors")
      || hasWeight(path, ".onnx")
      || hasWeight(path.resolve("onnx"), ".onnx"));
  }

  private static boolean isGguf(final Path path) {
    Path name = path.getFileName();
    return Files.isRegularFile(path)
      && name != null
      && name.toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
  }

  private static boolean hasWeight(final Path dir, final String suffix) {
    if (!Files.isDirectory(dir)) {
      return false;
    }
    try (Stream<Path> files = Files.list(dir)) {
      return files.anyMatch(file -> isWeightFile(file, suffix));
    } catch (IOException ex) {
      return false;
    }
  }

  private static boolean isWeightFile(final Path file, final String suffix) {
    Path name = file.getFileName();
    return name != null
      && Files.isRegularFile(file)
      && name.toString().toLowerCase(Locale.ROOT).endsWith(suffix);
  }

  private static Path configuredOrDefault() {
    String prop = NanoLlvmProps.systemProperty(PROP_MODEL);
    if (prop != null) {
      return Path.of(prop);
    }
    String env = NanoLlvmProps.environment(ENV_MODEL);
    if (env != null) {
      return Path.of(env);
    }
    return modelsRoot().resolve("Qwen3-0.6B");
  }

  /**
   * Chat checkpoint and Piper voice, checked together before either is loaded.
   *
   * @param chat  tool-capable chat checkpoint
   * @param voice Piper voice folder
   */
  public record Required(Path chat, Path voice) {
  }

  static final class MissingModels extends IllegalStateException {
    MissingModels(final String message) {
      super(message);
    }
  }
}
