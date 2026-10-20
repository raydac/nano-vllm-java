package com.igormaznitsa.nanollvm.jev;

/**
 * How strongly a {@link JevDecision} was read from the next-token distribution.
 *
 * <p>Set it with {@link com.igormaznitsa.nanollvm.llm.LLM.Builder#jevLevel(JevLevel)} before
 * {@link com.igormaznitsa.nanollvm.llm.LLM.Builder#toJev()}. The default is {@link #L0}.
 * {@link #RAW} is weaker: {@link JevDecision#require(JevLevel)} follows declaration order and
 * refuses a result below the level you name.
 *
 * <p>The model answers with a label ({@code A}, {@code Yes}, {@code 1}), not with the option
 * text, and it often prefers the first slot, the letter {@code A}, or the word {@code Yes}.
 * {@link #RAW} leaves that habit in the probabilities. {@link #L0} gives an unordered choice
 * every letter slot and a yes/no both phrasings, then averages in log space. After eight states
 * of the same question on the same {@link Jev}, L0 also divides out a label prior estimated from
 * those states. An ordered score stays one prefill at either level, because the bin order is the
 * scale; at L0 the prior still applies once eight states of that score have been seen.
 *
 * <p>{@link #L0} costs one prefill per option of a choice, and two prefills for a yes/no.
 * {@link #RAW} is always one prefill.
 *
 * <pre>{@code
 * try (LlmModel model = LlmModelFactory.make(path);
 *      Jev careful = LLM.builder(model).toJev();
 *      Jev quick = LLM.builder(model).jevLevel(JevLevel.RAW).toJev()) {
 *   JevQuestion route = JevQuestion.choice(
 *       "Which team should handle this?",
 *       List.of("billing", "technical", "sales"));
 *   JevDecision debiased = careful.decide(
 *       "My card was charged twice for one order.", route);
 *   String team = debiased.require(JevLevel.L0).answer();
 *
 *   JevDecision once = quick.decide(
 *       "My card was charged twice for one order.", route);
 *   once.answer();
 *   // once.require(JevLevel.L0) throws: this read was RAW
 * }
 * }</pre>
 *
 * @see JevDecision#rotations()
 * @see JevDecision#orderFlipRate()
 * @see JevDecision#priorApplied()
 * @since 1.5.0
 */
public enum JevLevel {

  /**
   * One prefill. Options stay in the order you passed, so a preference for the first slot or
   * for the letter {@code A} stays in the distribution.
   *
   * <p>{@link JevDecision#rotations()} is 1, {@link JevDecision#orderFlipRate()} is 0, and
   * {@link JevDecision#priorApplied()} stays false. {@code require(L0)} throws.
   */
  RAW,

  /**
   * Default. An unordered choice is read once per option: the options are sorted, then shifted
   * so each option sits under each letter once. A yes/no is read as {@code Yes or No} and as
   * {@code No or Yes}. Those probabilities are averaged in log space and mapped back onto the
   * options.
   *
   * <p>After eight states of the same question on this {@link Jev}, a label prior estimated from
   * those states is divided out at three-quarter strength and
   * {@link JevDecision#priorApplied()} becomes true. An ordered score is not rotated. The same
   * prior still applies after eight states of that score. {@link JevDecision#orderFlipRate()} is
   * the fraction of later rotations whose winner disagrees with the first, and 0 when only one
   * rotation was read.
   */
  L0
}
