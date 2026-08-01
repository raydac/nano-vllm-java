/**
 * JPMS module for the nano-vllm-java library.
 *
 * <p>Public API packages are exported; engine / tensor / model internals stay module-private.
 * Consumers that use the Vector API path need {@code jdk.incubator.vector} on the module path
 * (optional at runtime — scalar kernels are used when it is absent).
 */
module com.igormaznitsa.nanollvm {
  requires com.google.gson;
  requires static jdk.incubator.vector;

  exports com.igormaznitsa.nanollvm;
  exports com.igormaznitsa.nanollvm.chat;
  exports com.igormaznitsa.nanollvm.tokenizer;
  exports com.igormaznitsa.nanollvm.prompts;
  exports com.igormaznitsa.nanollvm.utils;
}
