/**
 * Loaded model surface ({@link Model}, {@link ModelFactory}) and architecture graphs ({@link CausalLM}, …).
 * <p>
 * Application code should use {@link ModelFactory#make(java.nio.file.Path)} and treat graph types as
 * implementation details unless extending the engine in-module.
 */

package com.igormaznitsa.nanollvm.models;
