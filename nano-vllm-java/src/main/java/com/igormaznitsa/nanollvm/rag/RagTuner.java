package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

/**
 * Load-time hook for {@link RagFactory.Builder#addProcessor(RagTuner...)}.
 *
 * <p>Several tuners run in registration order:
 * <ul>
 *   <li>{@link #isRagResourceAllowed(RagResource)} — all must return {@code true} or the file /
 *       classpath document is skipped (inline {@link RagFactory.Builder#add(String) add} is not
 *       filtered)</li>
 *   <li>{@link #extractRagText(RagResource)} — first non-empty {@link Optional} becomes the
 *       document body; {@link Optional#empty()} leaves extraction to the next tuner, then the
 *       built-in UTF-8 / PDF loader</li>
 *   <li>{@link #preprocessRagText(String)} — each tuner transforms the body in order, then the
 *       usual {@link RagLoadOptions#preprocess()} sentence packing runs</li>
 * </ul>
 *
 * <p>Override only the methods you need; defaults allow everything, extract nothing extra, and
 * leave text unchanged. One-method tuners: {@link #allowing}, {@link #extracting},
 * {@link #preprocessing}.
 *
 * <pre>{@code
 * PreparedRag rag = RagFactory.builder()
 *     .addProcessor(
 *         RagTuner.allowing(resource -> !resource.fileName().startsWith("_")),
 *         RagTuner.extracting(resource -> resource.fileName().endsWith(".html")
 *             ? Optional.of(stripTags(resource))
 *             : Optional.empty()),
 *         RagTuner.preprocessing(String::strip))
 *     .addFolder(Path.of("docs"))
 *     .build();
 * }</pre>
 *
 * @since 1.1.1
 */
public interface RagTuner {

  /**
   * Tuner that indexes a resource only when {@code allowed} is {@code true}.
   *
   * @param allowed predicate over file / classpath identity (and content when already loaded)
   * @return a tuner that only implements {@link #isRagResourceAllowed(RagResource)}
   * @throws NullPointerException if {@code allowed} is {@code null}
   */
  static RagTuner allowing(final Predicate<RagResource> allowed) {
    requireNonNull(allowed, "allowed");
    return new RagTuner() {
      @Override
      public boolean isRagResourceAllowed(final RagResource resource) {
        return allowed.test(requireNonNull(resource, "resource"));
      }

      @Override
      public String toString() {
        return "RagTuner.allowing";
      }
    };
  }

  /**
   * Tuner that supplies document text when {@code extractor} returns a present value.
   *
   * @param extractor {@link Optional#empty()} means “not this format” (fall through)
   * @return a tuner that only implements {@link #extractRagText(RagResource)}
   * @throws NullPointerException if {@code extractor} is {@code null}
   */
  static RagTuner extracting(final Function<RagResource, Optional<String>> extractor) {
    requireNonNull(extractor, "extractor");
    return new RagTuner() {
      @Override
      public Optional<String> extractRagText(final RagResource resource) {
        return requireNonNull(
          extractor.apply(requireNonNull(resource, "resource")),
          "extractRagText");
      }

      @Override
      public String toString() {
        return "RagTuner.extracting";
      }
    };
  }

  /**
   * Tuner that transforms extracted text before chunking.
   *
   * @param transform must not return {@code null}
   * @return a tuner that only implements {@link #preprocessRagText(String)}
   * @throws NullPointerException if {@code transform} is {@code null}
   */
  static RagTuner preprocessing(final UnaryOperator<String> transform) {
    requireNonNull(transform, "transform");
    return new RagTuner() {
      @Override
      public String preprocessRagText(final String text) {
        return requireNonNull(transform.apply(requireNonNull(text, "text")), "preprocessRagText");
      }

      @Override
      public String toString() {
        return "RagTuner.preprocessing";
      }
    };
  }

  /**
   * Whether this file or classpath document should be indexed. Default allows every resource.
   * Inline strings are not passed here.
   *
   * @param resource identity, optionally with {@link RagResource#content()} already loaded
   * @return {@code false} to skip this document
   * @throws NullPointerException if {@code resource} is {@code null}
   */
  default boolean isRagResourceAllowed(final RagResource resource) {
    requireNonNull(resource, "resource");
    return true;
  }

  /**
   * Custom document text. Empty means the next tuner or the built-in UTF-8 / PDF loader should
   * run. A present value (including blank) is used as-is and later passed through
   * {@link #preprocessRagText(String)}.
   *
   * @param resource loaded file or classpath document ({@link RagResource#hasContent()} is
   *                 {@code true})
   * @return extracted body, or empty to keep the default loader
   * @throws NullPointerException if {@code resource} is {@code null}
   */
  default Optional<String> extractRagText(final RagResource resource) {
    requireNonNull(resource, "resource");
    return Optional.empty();
  }

  /**
   * Transforms extracted (or inline) text before {@link RagLoadOptions#preprocess()} chunking.
   * Default returns {@code text} unchanged. Must not return {@code null}.
   *
   * @param text document body after extraction; never {@code null}
   * @return text to chunk; never {@code null}
   * @throws NullPointerException if {@code text} is {@code null}
   */
  default String preprocessRagText(final String text) {
    requireNonNull(text, "text");
    return text;
  }
}
