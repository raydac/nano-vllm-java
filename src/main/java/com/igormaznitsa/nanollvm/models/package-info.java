/**
 * Loaded model surface ({@link LlmModel}, {@link LlmModelFactory}) and architecture graphs ({@link CausalLM}, …).
 * <p>
 * Application code should use {@link LlmModelFactory#make(java.nio.file.Path)} and treat graph types as
 * implementation details unless extending the engine in-module.
 */

package com.igormaznitsa.nanollvm.models;
