package com.igormaznitsa.nanollvm.utils;

/**
 * Shared system-property / environment-variable keys and model-directory filenames.
 *
 * <p>Keys use the {@code nanollvm.*} / {@code NANOLLVM_*} spelling only. Library load and engine
 * code read {@link #PROP_ARCH}, {@link #PROP_KERNELS}, {@link #PROP_CPU_THREADS},
 * {@link #PROP_TORNADO_LAYERS}, and {@link #PROP_TORNADO_WEIGHT}. The model-path
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
 *   <tr><td>{@link #PROP_TORNADO_LAYERS} ({@code -Dnanollvm.tornado.layers=N})</td>
 *       <td>TornadoVM decoder graphs</td>
 *       <td>Layers in one device task graph. Default {@code 4}, range {@code 1}–{@code 16}.
 *           A larger group waits for the device less often and needs a larger
 *           {@code -Dtornado.tvm.maxbytecodesize} when the graph no longer fits.</td></tr>
 *   <tr><td>{@link #PROP_TORNADO_WEIGHT} ({@code -Dnanollvm.tornado.weight=512MB})</td>
 *       <td>TornadoVM GEMV</td>
 *       <td>Largest weight kept in one device buffer. A bigger matrix is streamed in tiles of
 *           this size. Unset, the size follows the TornadoVM heap: 32 MiB on the default 4 GiB
 *           heap, and the spare above 4 GiB when {@code -Dtornado.device.memory} is larger.</td></tr>
 * </table>
 */
public final class NanoLlvmProps {

  /**
   * Application / samples: path to one checkpoint. Not read by {@code LlmModelFactory}.
   */
  public static final String PROP_MODEL = "nanollvm.model";
  /**
   * Same as {@link #PROP_MODEL} via the environment.
   */
  public static final String ENV_MODEL = "NANOLLVM_MODEL";

  /**
   * Application / samples / tests: folder of checkpoints. Not read by {@code LlmModelFactory}.
   */
  public static final String PROP_MODELS_DIR = "nanollvm.models.dir";
  /**
   * Same as {@link #PROP_MODELS_DIR} via the environment.
   */
  public static final String ENV_MODELS_DIR = "NANOLLVM_MODELS_DIR";

  /**
   * Application / samples / tests: folder of RAG documents. Not read by {@code RagFactory}.
   */
  public static final String PROP_RAG_DIR = "nanollvm.rag.dir";
  /**
   * Same as {@link #PROP_RAG_DIR} via the environment.
   */
  public static final String ENV_RAG_DIR = "NANOLLVM_RAG_DIR";

  /**
   * Library: must match the checkpoint family ({@code qwen3}, {@code gemma3}, {@code llama}, {@code lfm2}).
   */
  public static final String PROP_ARCH = "nanollvm.arch";
  /**
   * Library: kernel mode {@code auto}, {@code tornado}, {@code vector}, or {@code scalar}.
   * Values {@code tornado} / {@code gpu} (and {@code auto} preferring TornadoVM) date from 1.4.0.
   */
  public static final String PROP_KERNELS = "nanollvm.kernels";
  /**
   * Library: matmul workers when {@link com.igormaznitsa.nanollvm.llm.LLM.Builder} did not set a thread count.
   */
  public static final String PROP_CPU_THREADS = "nanollvm.cpu.threads";
  /**
   * Samples CLI: {@code false} disables ANSI color on thinking streams.
   */
  public static final String PROP_COLOR = "nanollvm.color";
  /**
   * System property {@code nanollvm.tornado.layers}: how many decoder layers share one TornadoVM
   * task graph.
   *
   * <p>Read once, on the first TornadoVM decoder frame. A missing or blank value means {@code 4}.
   * The value must be an integer from {@code 1} to {@code 16}. Anything else throws
   * {@link IllegalArgumentException} when that frame starts.
   *
   * <p>Matmul, norm, RoPE, and attention for the grouped layers stay on the device. The activation
   * is copied back when the group ends, so a longer group waits for the device less often and
   * keeps more layer weights resident together. A shorter group compiles a smaller graph and
   * releases that resident set sooner. Prefill walks the same graph once per token.
   *
   * <p>TornadoVM encodes every graph into one bytecode buffer,
   * {@code -Dtornado.tvm.maxbytecodesize} (4096 bytes on TornadoVM 8 unless you change it). A group
   * that does not fit fails the launch. The failure names this property and that buffer: lower
   * the layer count, or raise the buffer, and run again. A graph of a whole layer stack also
   * measured slower than a short group, which is why the default stays at four.
   *
   * <p>Examples:
   * <ul>
   *   <li>{@code -Dnanollvm.tornado.layers=2} — shorter graphs on the default bytecode buffer</li>
   *   <li>{@code -Dnanollvm.tornado.layers=8 -Dtornado.tvm.maxbytecodesize=16384} — longer graphs
   *       when the device has room for the extra resident weights</li>
   * </ul>
   *
   * @see #PROP_TORNADO_WEIGHT
   * @since 1.5.0
   */
  public static final String PROP_TORNADO_LAYERS = "nanollvm.tornado.layers";
  /**
   * System property {@code nanollvm.tornado.weight}: the largest GEMV weight kept in one TornadoVM
   * device buffer.
   *
   * <p>Read once, on the first TornadoVM matrix-vector product. A missing or blank value selects
   * the budget from the device:
   * <ul>
   *   <li>TornadoVM's default heap is {@code -Dtornado.device.memory=4GB}. Decoder-layer graphs
   *       already fill that heap, so a matrix larger than 32 MiB, including a vocabulary
   *       projection, is copied through one reused tile of 32 MiB.</li>
   *   <li>A heap larger than 4 GiB may hold one extra matrix in the bytes above 4 GiB. The
   *       device's reported global memory and its maximum single allocation cap that figure. A
   *       heap setting larger than the memory the device reports does not raise the budget.</li>
   * </ul>
   *
   * <p>Set the property to replace that budget. The text is a byte count ({@code 33554432}) or a
   * 1024-based size: {@code 32MB}, {@code 32MiB}, {@code 512M}, {@code 1GB}. The accepted range is
   * 1 KiB through 64 GiB. A matrix that fits is uploaded once and reused for the current token
   * count. A larger matrix is streamed in tiles of this size. TornadoVM still allocates the
   * buffer from {@code -Dtornado.device.memory}, so this property cannot reserve memory the heap
   * does not have.
   *
   * <p>Example: {@code -Dtornado.device.memory=8GB -Dnanollvm.tornado.weight=1GB} keeps a
   * vocabulary projection resident when the heap has spare room above the layer graphs.
   *
   * @see #PROP_TORNADO_LAYERS
   * @since 1.5.0
   */
  public static final String PROP_TORNADO_WEIGHT = "nanollvm.tornado.weight";

  /**
   * Hugging Face sidecar filename expected in a model folder.
   */
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
