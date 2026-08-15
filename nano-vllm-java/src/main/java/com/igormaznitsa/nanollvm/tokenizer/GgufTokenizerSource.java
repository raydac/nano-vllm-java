package com.igormaznitsa.nanollvm.tokenizer;

import java.util.List;

/**
 * GGUF metadata needed by {@link Tokenizer#fromGguf(GgufTokenizerSource)}. Implemented inside the
 * module by the GGUF transport/reader; apps normally load via {@code LlmModelFactory}.
 */
public interface GgufTokenizerSource {

  List<String> metaStringArray(String key);

  String metaString(String key, String defaultValue);

  int metaInt(String key, int defaultValue);
}
