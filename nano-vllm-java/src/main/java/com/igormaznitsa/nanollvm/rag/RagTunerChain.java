package com.igormaznitsa.nanollvm.rag;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Ordered {@link RagTuner} pipeline used by {@link CorpusLoader}.
 *
 * <p>Filter is AND, extract is first present {@link Optional}, preprocess is a sequential
 * transform. Registration order is preserved.
 */
final class RagTunerChain {

  private static final RagTunerChain EMPTY = new RagTunerChain(List.of());

  private final List<RagTuner> tuners;

  /**
   * @param tuners tuners in registration order; copied
   */
  private RagTunerChain(final List<RagTuner> tuners) {
    this.tuners = List.copyOf(tuners);
  }

  /**
   * Chain with no tuners (allow all, no custom extract, identity preprocess).
   *
   * @return the shared empty chain
   */
  static RagTunerChain empty() {
    return EMPTY;
  }

  /**
   * Appends {@code more} after the tuners already in this chain.
   *
   * @param more tuners to add; must not contain {@code null}
   * @return a new chain, or {@code this} when {@code more} is empty
   * @throws NullPointerException if {@code more} or an element is {@code null}
   */
  RagTunerChain plus(final RagTuner... more) {
    requireNonNull(more, "tuners");
    if (more.length == 0) {
      return this;
    }
    List<RagTuner> next = new ArrayList<>(this.tuners.size() + more.length);
    next.addAll(this.tuners);
    for (RagTuner tuner : more) {
      next.add(requireNonNull(tuner, "tuner"));
    }
    return new RagTunerChain(next);
  }

  /**
   * {@code true} when every tuner allows {@code resource}.
   *
   * @param resource file or classpath identity; must not be {@code null}
   * @return {@code true} if the document should be indexed
   * @throws NullPointerException if {@code resource} is {@code null}
   */
  boolean allows(final RagResource resource) {
    requireNonNull(resource, "resource");
    return this.tuners.stream().allMatch(tuner -> tuner.isRagResourceAllowed(resource));
  }

  /**
   * First non-empty {@link RagTuner#extractRagText(RagResource)} in registration order.
   *
   * @param resource loaded document; must not be {@code null}
   * @return custom body, or empty to use the built-in UTF-8 loader
   * @throws NullPointerException if {@code resource} is {@code null}
   */
  Optional<String> extract(final RagResource resource) {
    requireNonNull(resource, "resource");
    return this.tuners.stream()
      .map(tuner -> requireNonNull(tuner.extractRagText(resource), "extractRagText"))
      .flatMap(Optional::stream)
      .findFirst();
  }

  /**
   * Applies every tuner's {@link RagTuner#preprocessRagText(String)} in registration order.
   *
   * @param text extracted or inline body; must not be {@code null}
   * @return text to chunk; never {@code null}
   * @throws NullPointerException if {@code text} is {@code null}
   */
  String preprocess(final String text) {
    requireNonNull(text, "text");
    String current = text;
    for (RagTuner tuner : this.tuners) {
      current = requireNonNull(tuner.preprocessRagText(current), "preprocessRagText");
    }
    return current;
  }
}
