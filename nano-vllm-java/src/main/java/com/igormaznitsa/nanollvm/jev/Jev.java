package com.igormaznitsa.nanollvm.jev;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.NextTokenRead;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Decision view of one causal chat engine. {@link LLM.Builder#toJev()} builds the engine and
 * this view; {@link #close()} closes the engine and does not close the shared model.
 *
 * <p>One {@code Jev} must not {@link #decide} concurrently. The label prior for
 * {@link JevLevel#L0} accumulates on this instance across calls that ask the same question.
 *
 * @since 1.5.0
 */
public final class Jev implements AutoCloseable {

  private final LLM engine;
  private final JevLevel level;
  private final Map<String, ResolvedLabels> labelCache = new HashMap<>();
  private final Map<String, PriorAccumulator> priors = new HashMap<>();
  private final AtomicBoolean closed = new AtomicBoolean();

  /**
   * Binds a causal engine. Closing this view closes {@code engine}.
   *
   * @param engine causal chat engine; must not be {@code null}
   * @param level  readout level; must not be {@code null}
   * @throws NullPointerException if either argument is {@code null}
   */
  public Jev(final LLM engine, final JevLevel level) {
    this.engine = requireNonNull(engine, "engine");
    this.level = requireNonNull(level, "level");
  }

  /**
   * Decides one state.
   *
   * @param state    situation text; blank is read as an empty state
   * @param question typed question; must not be {@code null}
   * @return the decision for {@code state}
   * @throws NullPointerException     if {@code question} is {@code null}
   * @throws IllegalArgumentException if a label is not a single token, or the prompt does not fit
   * @throws IllegalStateException    if this view is closed or the engine is not a chat model
   * @since 1.5.0
   */
  public JevDecision decide(final String state, final JevQuestion question) {
    requireNonNull(state, "state");
    return this.decideAll(List.of(state), question).getFirst();
  }

  /**
   * Decides many states of one question. An L0 batch of at least eight states (including earlier
   * calls for the same question) estimates the label prior from those states.
   *
   * @param states   situation texts; must not be empty
   * @param question typed question; must not be {@code null}
   * @return one decision per state, same order
   * @throws NullPointerException     if {@code states} or {@code question} is {@code null}
   * @throws IllegalArgumentException if {@code states} is empty, a label is not a single token, or
   *                                  a prompt does not fit
   * @throws IllegalStateException    if this view is closed or the engine is not a chat model
   * @since 1.5.0
   */
  public List<JevDecision> decideAll(final List<String> states, final JevQuestion question) {
    this.requireOpen();
    List<String> texts = this.requireStates(states);
    JevQuestion asked = requireNonNull(question, "question");
    ResolvedLabels labels = this.labelsFor(asked);
    List<int[]> rotations = JevMarginal.rotations(asked, this.level);
    List<StateRead> reads = this.readStates(texts, asked, labels, rotations);
    double[][] prior = this.rememberPrior(asked, rotations.size(), reads);
    return this.assemble(asked, rotations, reads, prior);
  }

  /**
   * Readout level fixed at construction.
   *
   * @return {@link JevLevel#L0} unless the builder selected raw
   * @since 1.5.0
   */
  public JevLevel level() {
    return this.level;
  }

  /**
   * Closes the chat engine this view built. A second call does nothing. The loaded model stays
   * open.
   */
  @Override
  public void close() {
    if (this.closed.compareAndSet(false, true)) {
      this.engine.close();
    }
  }

  private List<String> requireStates(final List<String> states) {
    requireNonNull(states, "states");
    if (states.isEmpty()) {
      throw new IllegalArgumentException("states must not be empty");
    }
    List<String> texts = new ArrayList<>(states.size());
    for (String state : states) {
      texts.add(requireNonNull(state, "state"));
    }
    return List.copyOf(texts);
  }

  private ResolvedLabels labelsFor(final JevQuestion question) {
    return this.labelCache.computeIfAbsent(
      question.kind().name() + ":" + question.options().size() + ":" + question.answerLabels(),
      key -> ResolvedLabels.from(JevLabels.resolve(this.engine.tokenizer(), question)));
  }

  private List<StateRead> readStates(
    final List<String> states,
    final JevQuestion question,
    final ResolvedLabels labels,
    final List<int[]> rotations
  ) {
    List<StateRead> reads = new ArrayList<>(states.size());
    for (String state : states) {
      double[][] positions = new double[rotations.size()][];
      double[] masses = new double[rotations.size()];
      for (int rotation = 0; rotation < rotations.size(); rotation++) {
        int[] permutation = rotations.get(rotation);
        String user = JevPrompts.userText(state, question, permutation, labels.texts());
        List<Integer> prompt = JevPrompts.promptTokenIds(
          this.engine.tokenizer(), this.engine.model().thinkTags(), user);
        NextTokenRead read = this.engine.readNextToken(
          prompt, JevLabels.idsForRotation(question, labels.ids(), permutation));
        positions[rotation] = JevMarginal.softmax(read.logProbabilities());
        masses[rotation] = read.labelMass();
      }
      reads.add(new StateRead(positions, masses));
    }
    return reads;
  }

  private double[][] rememberPrior(
    final JevQuestion question,
    final int rotationCount,
    final List<StateRead> reads
  ) {
    if (this.level != JevLevel.L0) {
      return null;
    }
    int optionCount = question.options().size();
    PriorAccumulator accumulator = this.priors.computeIfAbsent(
      question.priorKey(), key -> new PriorAccumulator(rotationCount, optionCount));
    for (StateRead read : reads) {
      accumulator.add(read.positions());
    }
    if (accumulator.count() < JevMarginal.MIN_PRIOR_STATES) {
      return null;
    }
    return accumulator.prior();
  }

  private List<JevDecision> assemble(
    final JevQuestion question,
    final List<int[]> rotations,
    final List<StateRead> reads,
    final double[][] prior
  ) {
    List<String> options = question.options();
    List<Double> centers = question.isScore() ? question.binCenters() : List.of();
    List<JevDecision> decisions = new ArrayList<>(reads.size());
    for (StateRead read : reads) {
      double[][] positions = prior == null
        ? read.positions()
        : JevMarginal.applyPrior(read.positions(), prior, JevMarginal.PRIOR_STRENGTH);
      double[] probabilities = JevMarginal.marginalize(positions, rotations);
      int winner = JevMarginal.argmax(probabilities);
      List<JevOption> distribution = new ArrayList<>(options.size());
      for (int index = 0; index < options.size(); index++) {
        distribution.add(new JevOption(options.get(index), probabilities[index]));
      }
      double mass = Arrays.stream(read.masses()).average().orElse(0d);
      decisions.add(new JevDecision(
        question.name(),
        this.level,
        options.get(winner),
        distribution,
        mass,
        rotations.size(),
        JevMarginal.orderFlipRate(read.positions(), rotations),
        prior != null,
        centers));
    }
    return List.copyOf(decisions);
  }

  private void requireOpen() {
    if (this.closed.get()) {
      throw new IllegalStateException("Jev is closed");
    }
  }

  private record StateRead(double[][] positions, double[] masses) {
  }

  private record ResolvedLabels(List<String> texts, int[] ids) {

    private static ResolvedLabels from(final JevLabels.Resolved resolved) {
      return new ResolvedLabels(resolved.labels(), resolved.ids().clone());
    }
  }

  private static final class PriorAccumulator {

    private final double[][] sum;
    private int count;

    private PriorAccumulator(final int rotations, final int options) {
      this.sum = new double[rotations][options];
    }

    private void add(final double[][] positions) {
      for (int rotation = 0; rotation < this.sum.length; rotation++) {
        for (int option = 0; option < this.sum[rotation].length; option++) {
          this.sum[rotation][option] += positions[rotation][option];
        }
      }
      this.count++;
    }

    private int count() {
      return this.count;
    }

    private double[][] prior() {
      return JevMarginal.meanPrior(this.sum, this.count);
    }
  }
}
