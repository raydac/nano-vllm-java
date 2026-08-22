/**
 * Application model surface: {@link LlmModel}, {@link LlmModelFactory}, {@link ModelSupport},
 * {@link LlmModality} / {@link LlmModalities}, and stream-backed load helpers ({@link ModelFileId},
 * {@link ModelFileSource}, {@link ModelFileSources}).
 *
 * <p>{@link LlmModel} is safe to share across threads and across many
 * {@link com.igormaznitsa.nanollvm.llm.LLM} instances until {@link LlmModel#close()}. Scratchpad
 * markers are {@link LlmModel#thinkTags()} and answer-search specials are
 * {@link LlmModel#chatSpecials()} from load-time {@link LlmModel#options()} (library defaults
 * when omitted). {@link LlmModel#modalities()} is the input/output {@link LlmModality} catalog
 * (chat graphs text→text, embedding encoders text→embedding; Gemma 4 also declares image/audio/video
 * on {@link LlmModel#modalities()} while {@link LlmModel#usableModalities()} stays text). Close each
 * {@code LLM} first, then the model, to release weight resources. Architecture graphs and weight
 * maps live in non-exported {@code models.internal} ({@link LlmModel} is sealed; the factory
 * returns that hidden implementation). Load layers live in non-exported
 * {@code models.llmcontainer} (weight files) and {@code models.llmarch} (architecture processors).
 * {@link ModelSupport} is the catalog of
 * architectures this library can run; unsupported checkpoints fail at load with
 * {@link com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException}.
 */

package com.igormaznitsa.nanollvm.models;
