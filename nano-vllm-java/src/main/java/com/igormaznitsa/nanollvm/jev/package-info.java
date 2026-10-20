/**
 * Typed decisions read from a causal chat model: a choice, a yes/no, or a score, with a
 * probability over the options taken from the next-token distribution.
 *
 * <p>Build with {@link com.igormaznitsa.nanollvm.llm.LLM.Builder#toJev()}. The view owns the
 * engine it builds and closes that engine on {@link Jev#close()}. It does not close the shared
 * {@link com.igormaznitsa.nanollvm.models.LlmModel}. Embeddings, Whisper, Piper, and fastText
 * have no next-token vocabulary and are refused.
 *
 * <p>{@link JevLevel} chooses the readout. {@link JevLevel#L0} (the default) reads every cyclic
 * rotation of a choice, averages in log space, and divides out a label prior estimated from the
 * decisions themselves once eight states of the same question have been seen. {@link JevLevel#RAW}
 * is one prefill in the order given. Nothing is parsed and nothing is trained. The readout follows
 * the training-free contract in
 * <a href="https://github.com/nokia-applied-research/AnyJev">AnyJev</a>
 * (Zhang et al., <a href="https://arxiv.org/abs/2610.00831">arXiv:2610.00831</a>).
 *
 * @since 1.5.0
 */

package com.igormaznitsa.nanollvm.jev;
