package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

/**
 * One document candidate during RAG load: a disk file or a classpath resource.
 *
 * <p>{@link RagTuner#isRagResourceAllowed(RagResource)} may see a resource before bytes are read
 * ({@link #content()} empty). {@link RagTuner#extractRagText(RagResource)} always sees loaded
 * bytes. Construct via {@link #file(Path)} / {@link #classpath(String)} (tests and custom tuners).
 *
 * @since 1.2.0
 */
public final class RagResource {

  private final Kind kind;
  private final String source;
  private final Path path;
  private final String classpathPath;
  private final byte[] content;

  /**
   * @param kind          file or classpath
   * @param source        chunk source label; never {@code null}
   * @param path          absolute file path, or {@code null} for classpath
   * @param classpathPath path without {@code classpath:}, or {@code null} for files
   * @param content       loaded bytes, or {@code null} when not yet read
   */
  private RagResource(
    final Kind kind,
    final String source,
    final Path path,
    final String classpathPath,
    final byte[] content
  ) {
    this.kind = requireNonNull(kind, "kind");
    this.source = requireNonNull(source, "source");
    this.path = path;
    this.classpathPath = classpathPath;
    this.content = content;
  }

  /**
   * Disk file identity (content not loaded yet).
   *
   * @param path file path; normalized to absolute; must not be {@code null}
   * @return resource whose {@link #content()} is empty
   * @throws NullPointerException if {@code path} is {@code null}
   */
  public static RagResource file(final Path path) {
    return file(path, null);
  }

  /**
   * Disk file with optional loaded bytes.
   *
   * @param path    file path; normalized to absolute; must not be {@code null}
   * @param content loaded bytes, or {@code null} when not yet read (copied when present)
   * @return resource for {@code path}
   * @throws NullPointerException if {@code path} is {@code null}
   */
  public static RagResource file(final Path path, final byte[] content) {
    requireNonNull(path, "path");
    Path normalized = path.toAbsolutePath().normalize();
    return new RagResource(
      Kind.FILE,
      normalized.toString(),
      normalized,
      null,
      copy(content));
  }

  /**
   * Classpath identity (content not loaded yet). Leading slashes are stripped.
   *
   * @param resourcePath absolute classpath path, e.g. {@code rag/facts.md}
   * @return resource whose {@link #source()} is {@code classpath:…} and {@link #content()} is empty
   * @throws NullPointerException     if {@code resourcePath} is {@code null}
   * @throws IllegalArgumentException if {@code resourcePath} is blank
   */
  public static RagResource classpath(final String resourcePath) {
    return classpath(resourcePath, null);
  }

  /**
   * Classpath resource with optional loaded bytes. Leading slashes are stripped.
   *
   * @param resourcePath absolute classpath path, e.g. {@code rag/facts.md}
   * @param content      loaded bytes, or {@code null} when not yet read (copied when present)
   * @return resource for {@code resourcePath}
   * @throws NullPointerException     if {@code resourcePath} is {@code null}
   * @throws IllegalArgumentException if {@code resourcePath} is blank
   */
  public static RagResource classpath(final String resourcePath, final byte[] content) {
    String path = requireClasspathPath(resourcePath);
    return new RagResource(
      Kind.CLASSPATH,
      "classpath:" + path,
      null,
      path,
      copy(content));
  }

  /**
   * Strips leading slashes and rejects blank classpath paths.
   *
   * @param resourcePath raw classpath path; must not be {@code null}
   * @return path without a leading {@code /}
   * @throws NullPointerException     if {@code resourcePath} is {@code null}
   * @throws IllegalArgumentException if the stripped path is blank
   */
  private static String requireClasspathPath(final String resourcePath) {
    requireNonNull(resourcePath, "resourcePath");
    String trimmed = resourcePath.strip();
    while (trimmed.startsWith("/")) {
      trimmed = trimmed.substring(1);
    }
    if (trimmed.isEmpty()) {
      throw new IllegalArgumentException("resourcePath must not be blank");
    }
    return trimmed;
  }

  /**
   * Defensive copy, or {@code null} when {@code content} is {@code null}.
   *
   * @param content bytes to copy; may be {@code null}
   * @return a clone of {@code content}, or {@code null}
   */
  private static byte[] copy(final byte[] content) {
    return content == null ? null : content.clone();
  }

  /**
   * {@link Kind#FILE} or {@link Kind#CLASSPATH}.
   *
   * @return origin of this document; never {@code null}
   */
  public Kind kind() {
    return this.kind;
  }

  /**
   * Canonical label used as the chunk {@link TextChunk#source()}: absolute file path or
   * {@code classpath:…}.
   *
   * @return source label; never {@code null}
   */
  public String source() {
    return this.source;
  }

  /**
   * File name with extension ({@code facts.md}), never a directory prefix.
   *
   * @return basename of the file or classpath path; never {@code null}
   */
  public String fileName() {
    if (this.path != null) {
      Path name = this.path.getFileName();
      return name == null ? this.source : name.toString();
    }
    int slash = Math.max(this.classpathPath.lastIndexOf('/'), this.classpathPath.lastIndexOf('\\'));
    return slash >= 0 ? this.classpathPath.substring(slash + 1) : this.classpathPath;
  }

  /**
   * Absolute normalized path when {@link #kind()} is {@link Kind#FILE}.
   *
   * @return the file path, or empty for classpath resources
   */
  public Optional<Path> path() {
    return Optional.ofNullable(this.path);
  }

  /**
   * Classpath path without {@code classpath:} when {@link #kind()} is {@link Kind#CLASSPATH}.
   *
   * @return the classpath path, or empty for disk files
   */
  public Optional<String> classpathPath() {
    return Optional.ofNullable(this.classpathPath);
  }

  /**
   * Loaded bytes, copied; empty when the resource has not been read yet (filter phase).
   *
   * @return a copy of the payload, or empty before read
   */
  public Optional<byte[]> content() {
    return this.content == null ? Optional.empty() : Optional.of(this.content.clone());
  }

  /**
   * {@code true} when {@link #kind()} is {@link Kind#FILE}.
   *
   * @return {@code true} for a disk file
   */
  public boolean isFile() {
    return this.kind == Kind.FILE;
  }

  /**
   * {@code true} when {@link #kind()} is {@link Kind#CLASSPATH}.
   *
   * @return {@code true} for a classpath resource
   */
  public boolean isClasspath() {
    return this.kind == Kind.CLASSPATH;
  }

  /**
   * {@code true} when bytes have been read for extraction.
   *
   * @return {@code true} if {@link #content()} is present
   */
  public boolean hasContent() {
    return this.content != null;
  }

  /**
   * Internal payload for the built-in UTF-8 loader (not copied).
   *
   * @return the loaded bytes
   * @throws IllegalStateException if content has not been read yet
   */
  byte[] rawContent() {
    if (this.content == null) {
      throw new IllegalStateException("content is not loaded: " + this.source);
    }
    return this.content;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean equals(final Object obj) {
    return this == obj
      || (obj instanceof RagResource other
      && this.kind == other.kind
      && this.source.equals(other.source)
      && Objects.equals(this.path, other.path)
      && Objects.equals(this.classpathPath, other.classpathPath)
      && Arrays.equals(this.content, other.content));
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int hashCode() {
    return Objects.hash(this.kind, this.source, this.path, this.classpathPath,
      Arrays.hashCode(this.content));
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public String toString() {
    return "RagResource{kind=%s, source=%s, loaded=%s}".formatted(
      this.kind, this.source, this.content != null);
  }

  /**
   * Where the document was found.
   *
   * @since 1.2.0
   */
  public enum Kind {
    /**
     * Regular file on disk ({@link RagResource#path()} is present).
     */
    FILE,
    /**
     * Classpath resource ({@link RagResource#classpathPath()} is present).
     */
    CLASSPATH
  }
}
