/**
 * JPMS module for the nano-vllm-java library.
 *
 * <p>Public API packages are exported ({@code models}, {@code llm}, {@code chat}, {@code rag},
 * {@code exceptions},
 * {@code tokenizer},
 * {@code prompts}, {@code utils} for Bundled* / Json / NanoVllmProps). Engine, tensor,
 * layers, and {@code internal} (loader / safetensors / inference Context) stay module-private.
 * Consumers that use the Vector API path need {@code jdk.incubator.vector} on the module path
 * (optional at runtime — scalar kernels are used when it is absent).
 */
module com.igormaznitsa.nanollvm {
  requires static jdk.incubator.vector;

  exports com.igormaznitsa.nanollvm;
  exports com.igormaznitsa.nanollvm.models;
  exports com.igormaznitsa.nanollvm.llm;
  exports com.igormaznitsa.nanollvm.exceptions;
  exports com.igormaznitsa.nanollvm.chat;
  exports com.igormaznitsa.nanollvm.rag;
  exports com.igormaznitsa.nanollvm.tokenizer;
  exports com.igormaznitsa.nanollvm.prompts;
  exports com.igormaznitsa.nanollvm.utils;
}
