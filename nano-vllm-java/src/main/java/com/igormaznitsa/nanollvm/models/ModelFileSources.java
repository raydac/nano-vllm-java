package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Adapters that turn a filesystem path, classpath folder, or classpath GGUF file into a
 * {@link ModelFileSource}.
 *
 * @since 1.1.0
 */
public final class ModelFileSources {

  private ModelFileSources() {
  }

  /**
   * HF model <em>folder</em> or a single {@code .gguf} <em>file</em> on disk (for tests / uniform
   * stream API). {@link LlmModelFactory#make(Path)} does not route through this.
   *
   * @param modelPath filesystem path to an HF folder or a {@code .gguf} file
   * @return stream source over that path
   * @since 1.1.0
   */
  public static ModelFileSource fromPath(final Path modelPath) {
    Path path = requireNonNull(modelPath, "modelPath").toAbsolutePath().normalize();
    if (isGgufFile(path)) {
      return new GgufFileSource(path);
    }
    if (!Files.isDirectory(path)) {
      throw new IllegalArgumentException(
        "model path is not an HF model folder or .gguf file: " + path);
    }
    return new FolderSource(path);
  }

  /**
   * Maps {@link ModelFileId#fileName()} and weight shard names under a classpath <em>folder</em>
   * (no leading slash; e.g. {@code models/MyChatModel}).
   *
   * @param loader         class loader that owns the resources
   * @param resourceFolder classpath folder prefix (not a file path)
   * @return stream source for that folder
   * @since 1.1.0
   */
  public static ModelFileSource classpath(final ClassLoader loader, final String resourceFolder) {
    requireNonNull(loader, "loader");
    String folder = normalizeResourcePath(resourceFolder, "resourceFolder");
    return new ClasspathFolderSource(loader, folder);
  }

  /**
   * Single GGUF <em>file</em> on the classpath (exact path, e.g. {@code models/gte-small.Q2_K.gguf}).
   *
   * @param loader           class loader that owns the resource
   * @param ggufResourceFile classpath path to one {@code .gguf} file (not a folder)
   * @return stream source for that file
   * @since 1.1.0
   */
  public static ModelFileSource classpathGguf(
    final ClassLoader loader,
    final String ggufResourceFile
  ) {
    requireNonNull(loader, "loader");
    String file = normalizeResourcePath(ggufResourceFile, "ggufResourceFile");
    return new ClasspathGgufFileSource(loader, file);
  }

  private static boolean isGgufFile(final Path path) {
    Path name = path.getFileName();
    return Files.isRegularFile(path)
      && name != null
      && name.toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
  }

  private static String normalizeResourcePath(final String resourcePath, final String paramName) {
    requireNonNull(resourcePath, paramName);
    String trimmed = resourcePath.strip();
    if (trimmed.isEmpty()) {
      throw new IllegalArgumentException(paramName + " must not be blank");
    }
    while (trimmed.startsWith("/")) {
      trimmed = trimmed.substring(1);
    }
    while (trimmed.endsWith("/")) {
      trimmed = trimmed.substring(0, trimmed.length() - 1);
    }
    if (trimmed.isEmpty()) {
      throw new IllegalArgumentException(paramName + " must not be blank");
    }
    return trimmed;
  }

  private record FolderSource(Path folder) implements ModelFileSource {

    @Override
    public InputStream open(final ModelFileId id) throws IOException {
      Path file = this.folder.resolve(id.fileName());
      if (Files.isRegularFile(file)) {
        return Files.newInputStream(file);
      }
      if (id == ModelFileId.MODEL_ONNX || id == ModelFileId.MODEL_ONNX_FP16) {
        Path nested = this.folder.resolve("onnx").resolve(id.fileName());
        if (Files.isRegularFile(nested)) {
          return Files.newInputStream(nested);
        }
      }
      return null;
    }

    @Override
    public InputStream openWeightShard(final String fileName) throws IOException {
      Path file = this.folder.resolve(requireNonNull(fileName, "fileName"));
      if (!file.normalize().startsWith(this.folder)) {
        throw new IllegalArgumentException("weight shard escapes model folder: " + fileName);
      }
      return Files.isRegularFile(file) ? Files.newInputStream(file) : null;
    }

    @Override
    public String displayName() {
      return this.folder.toString();
    }
  }

  private record GgufFileSource(Path ggufFile) implements ModelFileSource {

    @Override
    public InputStream open(final ModelFileId id) throws IOException {
      return id == ModelFileId.GGUF ? Files.newInputStream(this.ggufFile) : null;
    }

    @Override
    public String displayName() {
      return this.ggufFile.toString();
    }
  }

  private record ClasspathFolderSource(ClassLoader loader, String resourceFolder)
    implements ModelFileSource {

    @Override
    public InputStream open(final ModelFileId id) {
      InputStream in = this.loader.getResourceAsStream(this.resourceFolder + "/" + id.fileName());
      if (in != null) {
        return in;
      }
      if (id == ModelFileId.MODEL_ONNX || id == ModelFileId.MODEL_ONNX_FP16) {
        return this.loader.getResourceAsStream(
          this.resourceFolder + "/onnx/" + id.fileName());
      }
      return null;
    }

    @Override
    public InputStream openWeightShard(final String fileName) {
      requireNonNull(fileName, "fileName");
      if (fileName.indexOf('/') >= 0 || fileName.indexOf('\\') >= 0 || fileName.contains("..")) {
        throw new IllegalArgumentException("illegal weight shard name: " + fileName);
      }
      return this.loader.getResourceAsStream(this.resourceFolder + "/" + fileName);
    }

    @Override
    public String displayName() {
      return "classpath:" + this.resourceFolder;
    }
  }

  private record ClasspathGgufFileSource(ClassLoader loader, String ggufResourceFile)
    implements ModelFileSource {

    @Override
    public InputStream open(final ModelFileId id) {
      return id == ModelFileId.GGUF
        ? this.loader.getResourceAsStream(this.ggufResourceFile)
        : null;
    }

    @Override
    public String displayName() {
      return "classpath:" + this.ggufResourceFile;
    }
  }
}
