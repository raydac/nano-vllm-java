/**
 * Tokenizers for loaded models ({@link Tokenizer}) and the GGUF metadata bridge
 * ({@link GgufTokenizerSource}). Encode/decode algorithms live in package-private codec
 * classes behind {@link Tokenizer}.
 *
 * <p>Application code normally obtains a tokenizer via
 * {@link com.igormaznitsa.nanollvm.models.LlmModel#tokenizer()} after
 * {@link com.igormaznitsa.nanollvm.models.LlmModelFactory#make}. Instances are immutable after
 * load and safe to share across threads.
 */

package com.igormaznitsa.nanollvm.tokenizer;
