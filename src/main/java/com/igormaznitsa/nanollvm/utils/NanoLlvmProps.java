package com.igormaznitsa.nanollvm.utils;

/**
 * Shared system-property / environment-variable keys and model-directory filenames.
 *
 * <p>Preferred keys use the {@code nanollvm.*} / {@code NANOLLVM_*} spelling. Legacy
 * {@code nanovllm.*} / {@code NANOVLLM_*} keys are still read as fallbacks.
 */
public final class NanoLlvmProps {

  public static final String PROP_MODEL = "nanollvm.model";
  public static final String PROP_MODEL_LEGACY = "nanovllm.model";
  public static final String ENV_MODEL = "NANOLLVM_MODEL";
  public static final String ENV_MODEL_LEGACY = "NANOVLLM_MODEL";

  public static final String PROP_MODELS_DIR = "nanollvm.models.dir";
  public static final String PROP_MODELS_DIR_LEGACY = "nanovllm.models.dir";
  public static final String ENV_MODELS_DIR = "NANOLLVM_MODELS_DIR";
  public static final String ENV_MODELS_DIR_LEGACY = "NANOVLLM_MODELS_DIR";

  public static final String PROP_RAG_DIR = "nanollvm.rag.dir";
  public static final String PROP_RAG_DIR_LEGACY = "nanovllm.rag.dir";
  public static final String ENV_RAG_DIR = "NANOLLVM_RAG_DIR";
  public static final String ENV_RAG_DIR_LEGACY = "NANOVLLM_RAG_DIR";

  public static final String PROP_ARCH = "nanollvm.arch";
  public static final String PROP_ARCH_LEGACY = "nanovllm.arch";
  public static final String PROP_KERNELS = "nanollvm.kernels";
  public static final String PROP_KERNELS_LEGACY = "nanovllm.kernels";
  public static final String PROP_CPU_THREADS = "nanollvm.cpu.threads";
  public static final String PROP_CPU_THREADS_LEGACY = "nanovllm.cpu.threads";
  public static final String PROP_COLOR = "nanollvm.color";
  public static final String PROP_COLOR_LEGACY = "nanovllm.color";

  public static final String CONFIG_JSON = "config.json";

  private NanoLlvmProps() {
  }

  public static String systemProperty(final String preferred, final String legacy) {
    String value = System.getProperty(preferred);
    if (value != null && !value.isBlank()) {
      return value;
    }
    return System.getProperty(legacy);
  }

  public static String environment(final String preferred, final String legacy) {
    String value = System.getenv(preferred);
    if (value != null && !value.isBlank()) {
      return value;
    }
    return System.getenv(legacy);
  }
}
