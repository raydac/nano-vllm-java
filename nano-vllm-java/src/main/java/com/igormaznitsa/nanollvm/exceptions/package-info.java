/**
 * Library runtime errors. All types extend {@link NanoLlvmException}.
 * <p>
 * Thrown by {@link com.igormaznitsa.nanollvm.llm.LLM},
 * {@link com.igormaznitsa.nanollvm.models.LlmModelFactory},
 * {@link com.igormaznitsa.nanollvm.tokenizer.Tokenizer} load paths,
 * {@link com.igormaznitsa.nanollvm.rag.RagFactory} corpus load, and related entry points.
 * Argument validation may still throw {@link IllegalArgumentException}.
 */

package com.igormaznitsa.nanollvm.exceptions;
