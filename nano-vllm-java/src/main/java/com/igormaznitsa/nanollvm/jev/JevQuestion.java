package com.igormaznitsa.nanollvm.jev;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A typed question: {@link #choice}, {@link #yesNo}, or {@link #score}. Options are the strings
 * the decision returns. The model is shown letters ({@code A}, {@code B}, …) for a choice, the
 * words {@code Yes} and {@code No} for a yes/no, and numbers ({@code 1}, {@code 2}, …) for a
 * score.
 *
 * @since 1.5.0
 */
public final class JevQuestion {

  static final int MAX_CHOICE_OPTIONS = 26;
  private static final int MIN_OPTIONS = 2;
  private static final int MAX_SCORE_BINS = 10;

  private final Kind kind;
  private final String text;
  private final List<String> options;
  private final String name;
  private final boolean ordered;
  private final boolean explicitLevels;
  private final double scaleLow;
  private final double scaleHigh;
  private final List<Double> centers;

  private JevQuestion(
    final Kind kind,
    final String text,
    final List<String> options,
    final String name,
    final boolean ordered,
    final boolean explicitLevels,
    final double scaleLow,
    final double scaleHigh,
    final List<Double> centers
  ) {
    this.kind = kind;
    this.text = text;
    this.options = options;
    this.name = name;
    this.ordered = ordered;
    this.explicitLevels = explicitLevels;
    this.scaleLow = scaleLow;
    this.scaleHigh = scaleHigh;
    this.centers = centers;
  }

  /**
   * Unordered choice of two to twenty-six distinct options. The model answers with a letter.
   *
   * @param text    question shown to the model; must not be blank
   * @param options answer strings, in any order; at least two, at most twenty-six, unique
   * @return a choice question with no name
   * @throws NullPointerException     if {@code text} or {@code options} is {@code null}
   * @throws IllegalArgumentException if the text is blank or the options are not a valid set
   * @since 1.5.0
   */
  public static JevQuestion choice(final String text, final List<String> options) {
    return new JevQuestion(
      Kind.CHOICE,
      requireText(text),
      requireOptions(options, MAX_CHOICE_OPTIONS),
      "",
      false,
      false,
      0d,
      1d,
      List.of());
  }

  /**
   * Yes or no. The two words stay attached to the answers when the phrasing order flips.
   *
   * @param text question shown to the model; must not be blank
   * @return a yes/no question with options {@code Yes} and {@code No}
   * @throws NullPointerException     if {@code text} is {@code null}
   * @throws IllegalArgumentException if {@code text} is blank
   * @since 1.5.0
   */
  public static JevQuestion yesNo(final String text) {
    return new JevQuestion(
      Kind.YES_NO,
      requireText(text),
      List.of("Yes", "No"),
      "",
      false,
      false,
      0d,
      1d,
      List.of());
  }

  /**
   * Five equal-width bins on the scale {@code [0, 1]}.
   *
   * @param text question shown to the model; must not be blank
   * @return an ordered score question
   * @throws NullPointerException     if {@code text} is {@code null}
   * @throws IllegalArgumentException if {@code text} is blank
   * @since 1.5.0
   */
  public static JevQuestion score(final String text) {
    return score(text, 5);
  }

  /**
   * Equal-width bins on the scale {@code [0, 1]}. Bins are ordered and are not rotated.
   *
   * @param text question shown to the model; must not be blank
   * @param bins number of bins, from 2 to 10
   * @return an ordered score question
   * @throws NullPointerException     if {@code text} is {@code null}
   * @throws IllegalArgumentException if the text or bin count is invalid
   * @since 1.5.0
   */
  public static JevQuestion score(final String text, final int bins) {
    return scoreOnScale(text, bins, 0d, 1d, "");
  }

  private static JevQuestion scoreOnScale(
    final String text,
    final int bins,
    final double low,
    final double high,
    final String name
  ) {
    if (bins < MIN_OPTIONS || bins > MAX_SCORE_BINS) {
      throw new IllegalArgumentException("score supports 2 to 10 bins");
    }
    if (!(high > low) || !Double.isFinite(low) || !Double.isFinite(high)) {
      throw new IllegalArgumentException("scale must be finite with high > low");
    }
    List<String> labels = IntStream.range(0, bins)
      .mapToObj(bin -> "%s to %s".formatted(compact(edge(low, high, bins, bin)),
        compact(edge(low, high, bins, bin + 1))))
      .toList();
    return new JevQuestion(
      Kind.SCORE, requireText(text), labels, name, true, false, low, high, List.of());
  }

  /**
   * Ordered level descriptions. The score value of level {@code i} is {@code i}, starting at 0.
   *
   * @param text   question shown to the model; must not be blank
   * @param levels lowest level first; 2 to 10 distinct descriptions
   * @return an ordered score question
   * @throws NullPointerException     if {@code text} or {@code levels} is {@code null}
   * @throws IllegalArgumentException if the text or levels are invalid
   * @since 1.5.0
   */
  public static JevQuestion score(final String text, final List<String> levels) {
    List<String> labels = requireOptions(levels, MAX_SCORE_BINS);
    List<Double> centers =
      IntStream.range(0, labels.size()).mapToObj(index -> (double) index).toList();
    return new JevQuestion(
      Kind.SCORE,
      requireText(text),
      labels,
      "",
      true,
      true,
      0d,
      labels.size() - 1d,
      centers);
  }

  private static String requireText(final String text) {
    requireNonNull(text, "text");
    String stripped = text.strip();
    if (stripped.isEmpty()) {
      throw new IllegalArgumentException("question text must not be blank");
    }
    return stripped;
  }

  private static List<String> requireOptions(final List<String> options, final int max) {
    requireNonNull(options, "options");
    if (options.size() < MIN_OPTIONS || options.size() > max) {
      throw new IllegalArgumentException(
        "expected 2 to %d options, got %d".formatted(max, options.size()));
    }
    List<String> copy = new ArrayList<>(options.size());
    Set<String> seen = new HashSet<>();
    for (String option : options) {
      requireNonNull(option, "option");
      String stripped = option.strip();
      if (stripped.isEmpty()) {
        throw new IllegalArgumentException("option must not be blank");
      }
      if (!seen.add(stripped)) {
        throw new IllegalArgumentException("options must be unique");
      }
      copy.add(stripped);
    }
    return List.copyOf(copy);
  }

  private static double edge(final double low, final double high, final int bins, final int index) {
    return low + (high - low) * index / bins;
  }

  private static String compact(final double value) {
    if (Double.isFinite(value) && value == Math.rint(value) && Math.abs(value) < 1e15d) {
      return Long.toString(Math.round(value));
    }
    return Double.toString(value);
  }

  /**
   * Same bins on {@code [low, high]}. Only for an equal-width score, not for explicit levels.
   *
   * @param low  inclusive start of the scale
   * @param high end of the scale; must be greater than {@code low}
   * @return a score question with relabeled bins
   * @throws IllegalStateException    if this is not an equal-width score
   * @throws IllegalArgumentException if the scale is invalid
   * @since 1.5.0
   */
  public JevQuestion between(final double low, final double high) {
    if (this.kind != Kind.SCORE || this.explicitLevels) {
      throw new IllegalStateException("between applies to equal-width score bins");
    }
    return scoreOnScale(this.text, this.options.size(), low, high, this.name);
  }

  /**
   * Same question with a caller name, returned on {@link JevDecision#name()}.
   *
   * @param name non-blank name
   * @return a copy of this question
   * @throws NullPointerException     if {@code name} is {@code null}
   * @throws IllegalArgumentException if {@code name} is blank
   * @since 1.5.0
   */
  public JevQuestion named(final String name) {
    requireNonNull(name, "name");
    if (name.isBlank()) {
      throw new IllegalArgumentException("name must not be blank");
    }
    return new JevQuestion(
      this.kind,
      this.text,
      this.options,
      name.strip(),
      this.ordered,
      this.explicitLevels,
      this.scaleLow,
      this.scaleHigh,
      this.centers);
  }

  /**
   * Question kind.
   *
   * @return choice, yes/no, or score
   */
  public Kind kind() {
    return this.kind;
  }

  /**
   * Question text shown to the model.
   *
   * @return stripped text
   */
  public String text() {
    return this.text;
  }

  /**
   * Option strings in caller order. A decision's distribution uses this order.
   *
   * @return unmodifiable options
   */
  public List<String> options() {
    return List.copyOf(this.options);
  }

  /**
   * Caller name, or empty when {@link #named(String)} was not used.
   *
   * @return name or {@code ""}
   */
  public String name() {
    return this.name;
  }

  /**
   * {@code true} for a score question.
   *
   * @return whether {@link JevDecision#score()} is present
   */
  public boolean isScore() {
    return this.kind == Kind.SCORE;
  }

  /**
   * Value each score bin stands for. Equal-width bins use the bin midpoint. Explicit levels use
   * {@code 0 .. n-1}.
   *
   * @return one center per option
   * @throws IllegalStateException if this is not a score question
   */
  public List<Double> binCenters() {
    if (this.kind != Kind.SCORE) {
      throw new IllegalStateException("bin centers are only defined for a score question");
    }
    if (!this.centers.isEmpty()) {
      return List.copyOf(this.centers);
    }
    int bins = this.options.size();
    return IntStream.range(0, bins)
      .mapToDouble(bin -> this.scaleLow + (this.scaleHigh - this.scaleLow) * (bin + 0.5d) / bins)
      .boxed()
      .toList();
  }

  boolean isOrdered() {
    return this.ordered;
  }

  boolean hasExplicitLevels() {
    return this.explicitLevels;
  }

  double scaleLow() {
    return this.scaleLow;
  }

  double scaleHigh() {
    return this.scaleHigh;
  }

  List<String> answerLabels() {
    if (this.kind == Kind.YES_NO) {
      return List.of("Yes", "No");
    }
    if (this.kind == Kind.SCORE) {
      return IntStream.rangeClosed(1, this.options.size()).mapToObj(Integer::toString).toList();
    }
    return this.letterLabels();
  }

  List<String> letterLabels() {
    return IntStream.range(0, this.options.size())
      .mapToObj(index -> Character.toString((char) ('A' + index)))
      .toList();
  }

  String priorKey() {
    String optionKey = this.ordered
      ? String.join("\u0001", this.options)
      : this.options.stream().sorted().collect(Collectors.joining("\u0001"));
    return this.kind.name() + "\u0000" + this.text + "\u0000" + optionKey;
  }

  /**
   * Which answers the model is asked to pick.
   *
   * @since 1.5.0
   */
  public enum Kind {
    /**
     * Several named options, answered with a letter.
     */
    CHOICE,
    /**
     * {@code Yes} or {@code No}.
     */
    YES_NO,
    /**
     * Ordered bins or levels, answered with a number (or a letter when the tokenizer has no
     * single-token digits).
     */
    SCORE
  }
}
