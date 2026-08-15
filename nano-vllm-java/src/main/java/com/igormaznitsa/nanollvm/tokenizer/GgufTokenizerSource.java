package com.igormaznitsa.nanollvm.tokenizer;

import java.util.List;

/**
 * GGUF metadata needed by {@link Tokenizer#fromGguf(GgufTokenizerSource)}. Implemented inside the
 * module by the GGUF transport/reader; apps normally load via {@code LlmModelFactory}.
 */
public interface GgufTokenizerSource {

  /**
   * GGUF string-array metadata for {@code key}, or empty when absent.
   */
  List<String> metaStringArray(String key);

  /**
   * GGUF string metadata for {@code key}, or {@code defaultValue} when absent.
   */
  String metaString(String key, String defaultValue);

  /**
   * GGUF integer metadata for {@code key}, or {@code defaultValue} when absent.
   */
  int metaInt(String key, int defaultValue);
}
