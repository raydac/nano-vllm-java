/**
 * JPMS module for the nano-vllm-java library.
 *
 * <p><b>Exported (application API):</b> {@code models}, {@code llm}, {@code chat}, {@code rag},
 * {@code exceptions}, {@code tokenizer}, {@code prompts}, {@code utils}.
 *
 * <p><b>Not exported:</b> {@code samples} ({@code Example}, {@code Bench},
 * {@code LogTriageHelloWorld}, {@code samples.utils} — still runnable as module main classes),
 * {@code engine}, {@code layers}, {@code tensor} (and {@code tensor.scalar} / {@code tensor.vector}),
 * {@code internal} (safetensors / GGUF loaders, inference {@code Context}). Application code should
 * use {@code LlmModelFactory} / {@code LLM} / {@code RagFactory} rather than loaders or the network graph.
 *
 * <p>{@link com.igormaznitsa.nanollvm.models.CausalLM} and {@link com.igormaznitsa.nanollvm.models.WeightBag}
 * mention {@code tensor}/{@code layers} types in some signatures for in-module engine use;
 * {@link com.igormaznitsa.nanollvm.models.LlmModel#network()} is not a stable app surface.
 * GGUF tokenizer build uses exported {@link com.igormaznitsa.nanollvm.tokenizer.GgufTokenizerSource}
 * (implemented by the private GGUF reader).
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
  exports com.igormaznitsa.nanollvm.prompts;
  exports com.igormaznitsa.nanollvm.utils;
}
