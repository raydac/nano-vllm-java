package com.igormaznitsa.nanollvm.models.llmcontainer;

import java.io.IOException;

/**
 * Weight-container transport: open a GGUF file, safetensors shard set, or ONNX protobuf, then
 * expose an architecture-agnostic {@link ContainerCatalog}. Does not bind graphs or fill
 * {@link com.igormaznitsa.nanollvm.models.internal.WeightBag}.
 *
 * @since 1.1.0
 */
public sealed interface ContainerTransport extends AutoCloseable
  permits GgufTransport, SafetensorsTransport, OnnxTransport {

  /**
   * Metadata and tensor names for architecture bind. Safe to call more than once.
   *
   * @since 1.1.0
   */
  ContainerCatalog catalog();

  /**
   * Drops payload buffers and leftover file handles. Idempotent.
   *
   * @since 1.1.0
   */
  @Override
  void close() throws IOException;
}
