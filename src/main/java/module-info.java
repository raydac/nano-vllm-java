/**
 * JPMS module for the nano-vllm-java library.
 *
 * <p>Public API packages are exported; engine / tensor / model internals stay module-private.
 * Consumers that use the Vector API path need {@code jdk.incubator.vector} on the module path
 * (optional at runtime — scalar kernels are used when it is absent).
 */
module io.nanovllm {
  requires com.google.gson;
  requires static jdk.incubator.vector;

  exports io.nanovllm;
  exports io.nanovllm.chat;
  exports io.nanovllm.tokenizer;
  exports io.nanovllm.prompts;
  exports io.nanovllm.utils;
}
