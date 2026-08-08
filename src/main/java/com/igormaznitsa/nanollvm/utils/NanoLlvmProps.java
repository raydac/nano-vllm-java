package com.igormaznitsa.nanollvm.utils;

/**
 * Shared system-property / environment-variable keys and model-directory filenames.
 *
 * <p>Keys use the {@code nanollvm.*} / {@code NANOLLVM_*} spelling only.
 */
public final class NanoLlvmProps {

  public static final String PROP_MODEL = "nanollvm.model";
  public static final String ENV_MODEL = "NANOLLVM_MODEL";

  public static final String PROP_MODELS_DIR = "nanollvm.models.dir";
  public static final String ENV_MODELS_DIR = "NANOLLVM_MODELS_DIR";

  public static final String PROP_RAG_DIR = "nanollvm.rag.dir";
  public static final String ENV_RAG_DIR = "NANOLLVM_RAG_DIR";

  public static final String PROP_ARCH = "nanollvm.arch";
  public static final String PROP_KERNELS = "nanollvm.kernels";
  public static final String PROP_CPU_THREADS = "nanollvm.cpu.threads";
  public static final String PROP_COLOR = "nanollvm.color";

  public static final String CONFIG_JSON = "config.json";

  private NanoLlvmProps() {
  }

  public static String systemProperty(final String key) {
    String value = System.getProperty(key);
    if (value != null && !value.isBlank()) {
      return value;
    }
    return null;
  }

  public static String environment(final String key) {
    String value = System.getenv(key);
    if (value != null && !value.isBlank()) {
      return value;
    }
    return null;
  }
}
