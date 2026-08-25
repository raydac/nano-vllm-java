/**
 * Architecture processors: bind a container catalog, fill a weight bag, and build the graph.
 * Chat families extend {@link CausalArchitecture} (Hugging Face bind by default, required causal
 * graph). Embedding families extend {@link EmbeddingArchitecture}. Whisper speech extends
 * {@link SpeechArchitecture}; Piper synthesis extends {@link SynthesisArchitecture}.
 * Not exported; application code loads through
 * {@link com.igormaznitsa.nanollvm.models.LlmModelFactory}.
 *
 * @since 1.1.0
 */

package com.igormaznitsa.nanollvm.models.llmarch;
