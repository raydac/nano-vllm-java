package com.igormaznitsa.nanollvm.exceptions;

import static java.util.Objects.requireNonNull;

import java.util.List;

/**
 * The checkpoint’s architecture or weight container is not one this library can run.
 *
 * @since 1.1.0
 */
public final class UnsupportedModelException extends ModelLoadException {

  private final String modelType;
  private final List<String> architectures;

  /**
   * Load failure for an architecture or container this library cannot run.
   *
   * @param message       user-facing reason (catalog is usually appended by {@link com.igormaznitsa.nanollvm.models.ModelSupport})
   * @param modelType     HF {@code model_type} or GGUF architecture; may be blank
   * @param architectures HF {@code architectures} list; may be empty
   * @since 1.1.0
   */
  public UnsupportedModelException(
    final String message,
    final String modelType,
    final List<String> architectures
  ) {
    super(requireNonNull(message, "message"));
    this.modelType = modelType == null ? "" : modelType;
    this.architectures = architectures == null ? List.of() : List.copyOf(architectures);
  }

  /**
   * HF {@code model_type} or GGUF architecture string (may be blank).
   *
   * @since 1.1.0
   */
  public String modelType() {
    return this.modelType;
  }

  /**
   * HF {@code architectures} list (never {@code null}; may be empty).
   *
   * @since 1.1.0
   */
  public List<String> architectures() {
    return this.architectures;
  }
}
