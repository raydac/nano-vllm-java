/**
 * JPMS module for the nano-vllm-java library.
 *
 * <p><b>Exported (application API):</b> {@code models}, {@code llm}, {@code chat}, {@code rag},
 * {@code exceptions}, {@code tokenizer}, {@code utils}.
 *
 * <p><b>Not exported:</b> {@code prompts} (engine wording), {@code models.internal} (CausalLM graph /
 * weights; sealed {@code LlmModel} / hidden {@code LlmModelImpl}), {@code engine}, {@code layers}, {@code tensor},
 * {@code internal} (JSON / GGUF dequant / runtime helpers), {@code models.llmcontainer}
 * (weight-file transport), {@code models.llmarch} (architecture bind/fill/create). Application code should use
 * {@code LlmModelFactory} / {@code LLM} / {@code RagFactory}. Runnable demos live in the separate
 * {@code nano-vllm-java-samples} Maven module.
 *
 * <p>GGUF tokenizer build uses exported
 * {@link com.igormaznitsa.nanollvm.tokenizer.GgufTokenizerSource}.
 *
 * <p>Consumers that use the Vector API path need {@code jdk.incubator.vector} on the module path
 * (optional at runtime — scalar kernels are used when it is absent).
 */
module com.igormaznitsa.nanollvm {
  requires static jdk.incubator.vector;

  exports com.igormaznitsa.nanollvm.models;
  exports com.igormaznitsa.nanollvm.llm;
  exports com.igormaznitsa.nanollvm.exceptions;
  exports com.igormaznitsa.nanollvm.chat;
  exports com.igormaznitsa.nanollvm.rag;
  exports com.igormaznitsa.nanollvm.tokenizer;
  exports com.igormaznitsa.nanollvm.utils;
}
