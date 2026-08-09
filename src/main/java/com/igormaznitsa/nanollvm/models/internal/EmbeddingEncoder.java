package com.igormaznitsa.nanollvm.models.internal;

import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;

/**
 * Encoder that maps token ids to a single dense embedding vector.
 *
 * @since 1.1.0
 */
public interface EmbeddingEncoder {

  String architectureName();

  int embeddingDim();

  float[] encode(int[] tokenIds, MatmulRuntime runtime);
}
