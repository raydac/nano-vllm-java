/**
 * Application model surface: {@link LlmModel} and {@link LlmModelFactory}.
 *
 * <p>{@link LlmModel} is immutable and safe to share across threads and across many
 * {@link com.igormaznitsa.nanollvm.llm.LLM} instances. Architecture graphs and weight maps live in
 * non-exported {@code models.internal}.
 */

package com.igormaznitsa.nanollvm.models;
