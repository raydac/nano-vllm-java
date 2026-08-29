package com.igormaznitsa.nanollvm.utils;

/**
 * Shared system-property / environment-variable keys and model-directory filenames.
 *
 * <p>Keys use the {@code nanollvm.*} / {@code NANOLLVM_*} spelling only. Library load and engine
 * code read {@link #PROP_ARCH}, {@link #PROP_KERNELS}, and {@link #PROP_CPU_THREADS}. The model-path
 * and color keys are conveniences for applications and the samples module — {@code LlmModelFactory}
 * does not read them; you pass a {@link java.nio.file.Path} yourself.
 *
 * <table>
 *   <caption>Flags</caption>
 *   <tr><th>Key</th><th>Who reads it</th><th>Effect</th></tr>
 *   <tr><td>{@link #PROP_CPU_THREADS} ({@code -Dnanollvm.cpu.threads=N})</td>
 *       <td>{@link com.igormaznitsa.nanollvm.llm.LLM.Builder}</td>
 *       <td>Matmul worker count when the builder did not call {@code cpuThreads} /
 *           {@code allCpuThreads} / {@code disableMultiCpu}. Must be {@code >= 1}.</td></tr>
 *   <tr><td>{@link #PROP_ARCH} ({@code -Dnanollvm.arch=qwen3})</td>
 *       <td>{@link com.igormaznitsa.nanollvm.models.ModelSupport}</td>
 *       <td>Must match the checkpoint family; used to fail fast on a mismatch, not to pick a
 *           random architecture.</td></tr>
 *   <tr><td>{@link #PROP_KERNELS} ({@code auto}|{@code tornado}|{@code vector}|{@code scalar})</td>
 *       <td>CPU math kernels</td>
 *       <td>{@code auto} prefers TornadoVM when the optional add-on and a device are present,
 *           else Vector API, else scalar. {@code scalar} forces portable Java.</td></tr>
 *   <tr><td>{@link #PROP_MODEL} / {@link #ENV_MODEL}</td>
 *       <td>samples / your app</td>
 *       <td>Suggested single checkpoint path. Not read by {@code LlmModelFactory}.</td></tr>
 *   <tr><td>{@link #PROP_MODELS_DIR} / {@link #ENV_MODELS_DIR}</td>
 *       <td>samples / tests</td>
 *       <td>Folder of checkpoints. Not read by {@code LlmModelFactory}.</td></tr>
 *   <tr><td>{@link #PROP_RAG_DIR} / {@link #ENV_RAG_DIR}</td>
 *       <td>samples / tests</td>
 *       <td>Folder of RAG documents. Not read by {@code RagFactory}.</td></tr>
 *   <tr><td>{@link #PROP_COLOR}</td>
 *       <td>samples CLI</td>
 *       <td>{@code false} disables ANSI color on thinking streams.</td></tr>
 * </table>
 */
public final class NanoLlvmProps {

  /**
   * Application / samples: path to one checkpoint. Not read by {@code LlmModelFactory}.
   */
  public static final String PROP_MODEL = "nanollvm.model";
  /** Same as {@link #PROP_MODEL} via the environment. */
  public static final String ENV_MODEL = "NANOLLVM_MODEL";

  /** Application / samples / tests: folder of checkpoints. Not read by {@code LlmModelFactory}. */
  public static final String PROP_MODELS_DIR = "nanollvm.models.dir";
  /** Same as {@link #PROP_MODELS_DIR} via the environment. */
  public static final String ENV_MODELS_DIR = "NANOLLVM_MODELS_DIR";

  /** Application / samples / tests: folder of RAG documents. Not read by {@code RagFactory}. */
  public static final String PROP_RAG_DIR = "nanollvm.rag.dir";
  /** Same as {@link #PROP_RAG_DIR} via the environment. */
  public static final String ENV_RAG_DIR = "NANOLLVM_RAG_DIR";

  /** Library: must match the checkpoint family ({@code qwen3}, {@code gemma3}, {@code llama}, {@code lfm2}). */
  public static final String PROP_ARCH = "nanollvm.arch";
  /**
   * Library: kernel mode {@code auto}, {@code tornado}, {@code vector}, or {@code scalar}.
   */
  public static final String PROP_KERNELS = "nanollvm.kernels";
  /** Library: matmul workers when {@link com.igormaznitsa.nanollvm.llm.LLM.Builder} did not set a thread count. */
  public static final String PROP_CPU_THREADS = "nanollvm.cpu.threads";
  /** Samples CLI: {@code false} disables ANSI color on thinking streams. */
  public static final String PROP_COLOR = "nanollvm.color";

  /** Hugging Face sidecar filename expected in a model folder. */
  public static final String CONFIG_JSON = "config.json";

  private NanoLlvmProps() {
  }

  /**
   * Non-blank {@link System#getProperty(String)}, or {@code null} when unset/blank.
   */
  public static String systemProperty(final String key) {
    String value = System.getProperty(key);
    if (value != null && !value.isBlank()) {
      return value;
    }
    return null;
  }

  /**
   * Non-blank {@link System#getenv(String)}, or {@code null} when unset/blank.
   */
  public static String environment(final String key) {
    String value = System.getenv(key);
    if (value != null && !value.isBlank()) {
      return value;
    }
    return null;
  }
}
