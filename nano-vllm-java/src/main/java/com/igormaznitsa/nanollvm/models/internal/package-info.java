/**
 * Non-exported architecture graphs, weight map, load schema, and {@code LlmModel} implementation.
 * Used by the engine and {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} only.
 *
 * <p>Causal chat is {@code *ForCausalLM}; embeddings {@link BertForEmbedding}; Whisper ASR
 * {@link SpeechToText} / {@link WhisperForAsr}; Piper TTS {@link TextToSpeech} /
 * {@link PiperForTts}. WAV and log-mel helpers live in {@code models.internal.audio}.
 */

package com.igormaznitsa.nanollvm.models.internal;
