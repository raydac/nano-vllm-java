/**
 * Application model surface: {@link LlmModel}, {@link LlmModelFactory}, {@link ModelSupport},
 * and stream-backed load helpers ({@link ModelFileId}, {@link ModelFileSource},
 * {@link ModelFileSources}).
 *
 * <p>{@link LlmModel} is safe to share across threads and across many
 * {@link com.igormaznitsa.nanollvm.llm.LLM} instances until {@link LlmModel#close()}. Scratchpad
 * markers are {@link LlmModel#thinkTags()} from load-time {@link LlmModel#options()}. Close each
 * {@code LLM} first, then the model, to release weight resources. Architecture graphs and weight
 * maps live in non-exported {@code models.internal}. {@link ModelSupport} is the catalog of
 * architectures this library can run; unsupported checkpoints fail at load with
 * {@link com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException}.
 */

package com.igormaznitsa.nanollvm.models;
