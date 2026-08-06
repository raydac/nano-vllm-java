package com.igormaznitsa.nanollvm.utils;

/**
 * Shared system-property / environment-variable keys and model-directory filenames.
 */
public final class NanoVllmProps {

  public static final String PROP_MODEL = "nanovllm.model";
  public static final String ENV_MODEL = "NANOVLLM_MODEL";

  public static final String PROP_MODELS_DIR = "nanovllm.models.dir";
  public static final String ENV_MODELS_DIR = "NANOVLLM_MODELS_DIR";

  public static final String PROP_RAG_DIR = "nanovllm.rag.dir";
  public static final String ENV_RAG_DIR = "NANOVLLM_RAG_DIR";

  public static final String PROP_ARCH = "nanovllm.arch";
  public static final String PROP_KERNELS = "nanovllm.kernels";
  public static final String PROP_CPU_THREADS = "nanovllm.cpu.threads";
  public static final String PROP_COLOR = "nanovllm.color";

  public static final String CONFIG_JSON = "config.json";

  private NanoVllmProps() {
  }
}
