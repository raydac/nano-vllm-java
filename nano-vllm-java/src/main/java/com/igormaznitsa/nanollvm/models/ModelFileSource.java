package com.igormaznitsa.nanollvm.models;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import java.io.IOException;
import java.io.InputStream;

/**
 * Supplies model file bytes by logical {@link ModelFileId} (and named safetensors shards).
 *
 * <p>Return {@code null} from {@link #open(ModelFileId)} when an optional file is absent. Each call
 * must open a fresh stream; the factory closes streams it opens.
 *
 * @since 1.1.0
 */
@FunctionalInterface
public interface ModelFileSource {

  /**
   * Opens a stream for {@code id}, or {@code null} if that file is not available.
   *
   * @since 1.1.0
   */
  InputStream open(ModelFileId id) throws IOException;

  /**
   * Opens a weight shard named in {@link ModelFileId#SAFE_TENSORS_INDEX}
   * (e.g. {@code model-00001-of-00003.safetensors}).
   *
   * @since 1.1.0
   */
  default InputStream openWeightShard(final String fileName) throws IOException {
    throw new ModelLoadException("weight shard not supported by this source: " + fileName);
  }

  /**
   * Label for logs (e.g. {@code classpath:models/MyChatModel}).
   *
   * @since 1.1.0
   */
  default String displayName() {
    return "stream";
  }
}
