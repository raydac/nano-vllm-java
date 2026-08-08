/**
 * Application model surface: {@link LlmModel} and {@link LlmModelFactory}.
 *
 * <p>{@link LlmModel} is safe to share across threads and across many
 * {@link com.igormaznitsa.nanollvm.llm.LLM} instances until {@link LlmModel#close()}. Close each
 * {@code LLM} first, then the model, to release weight resources. Architecture graphs and weight
 * maps live in non-exported {@code models.internal}.
 */

package com.igormaznitsa.nanollvm.models;
