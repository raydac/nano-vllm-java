package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import java.util.Optional;

/**
 * Text to phoneme ids for one tiny-voca language.
 */
public interface TinyVocaFront {

  Optional<TinyVocaUtterance> phonemize(String text);
}
