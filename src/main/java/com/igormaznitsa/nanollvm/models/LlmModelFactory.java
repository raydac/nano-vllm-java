package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.GgufModelLoader;
import com.igormaznitsa.nanollvm.internal.GgufModelLoader.LoadedGguf;
import com.igormaznitsa.nanollvm.internal.GgufReader;
import com.igormaznitsa.nanollvm.internal.ModelLoader;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.CausalLMFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Loads a {@link LlmModel} from a HuggingFace model directory or a {@code .gguf} file.
 *
 * <p>One {@link LlmModel} may be reused by any number of {@link LLM} instances until
 * {@link LlmModel#close()}. Load is blocking I/O on the calling thread; the returned model is safe
 * to share across threads while open.
 *
 * <p>GGUF stays packed by default. Pass {@code allowUnpackParameters=true} to dequantize to float32
 * during load (mmap → float tensors; no packed heap copy).
 */
public final class LlmModelFactory {

  private LlmModelFactory() {
  }

  /**
   * Loads weights with silent progress I/O.
   *
   * @throws ModelLoadException if the path is missing or the model cannot be loaded
   */
  public static LlmModel make(final Path modelDir) {
    return make(modelDir, LlmListeners.silent(), false);
  }

  /**
   * Loads weights from a path string with silent progress I/O.
   *
   * @throws ModelLoadException if the path is missing or the model cannot be loaded
   */
  public static LlmModel make(final String modelPath) {
    return make(Path.of(requireNonNull(modelPath, "modelPath")));
  }

  /**
   * Loads weights, reporting progress through {@code io} (null → silent). GGUF stays packed.
   *
   * @throws ModelLoadException if the path is missing or the model cannot be loaded
   */
  public static LlmModel make(final Path modelPath, final LlmListener io) {
    return make(modelPath, io, false);
  }

  /**
   * Loads weights, reporting progress through {@code io} (null → silent).
   *
   * @param allowUnpackParameters when {@code true}, GGUF tensors are expanded to float32 at load
   *                              (no packed heap residency); HF safetensors ignore this flag
   * @throws ModelLoadException if the path is missing or the model cannot be loaded
   */
  public static LlmModel make(
    final Path modelPath,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) {
    requireNonNull(modelPath, "modelPath");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Path path = modelPath.toAbsolutePath().normalize();
    try {
      if (Files.isRegularFile(path) && path.getFileName().toString().toLowerCase(Locale.ROOT)
        .endsWith(".gguf")) {
        return loadGguf(path, streams, allowUnpackParameters);
      }
      if (!Files.isDirectory(path)) {
        throw new ModelLoadException("model path is not a directory or .gguf file: " + path);
      }
      return loadHf(path, streams);
    } catch (ModelLoadException e) {
      throw e;
    } catch (RuntimeException | IOException e) {
      throw new ModelLoadException("failed to load model from " + path, e);
    }
  }

  private static LlmModel loadHf(final Path path, final LlmListener io) throws IOException {
    long t0 = System.nanoTime();
    LlmListeners.info(io, null, "CPU backend: " + MatmulRuntime.sequential().backendInfo());

    Config.HfConfig hfConfig = Config.HfConfig.load(path.resolve(CONFIG_JSON));
    String arch = CausalLMFactory.detect(hfConfig);
    WeightSchema schema = CausalLMFactory.schema(hfConfig);

    LlmListeners.info(io, null, "Loading " + arch + " weights…");
    WeightBag weights = ModelLoader.loadWeights(path, hfConfig, schema, io);

    LlmListeners.info(io, null, "Building " + arch + " model graph…");
    long tGraph = System.nanoTime();
    CausalLM network = CausalLMFactory.create(hfConfig, weights);
    LlmListeners.infof(io, null, "Model graph ready (%s) in %.1fs%n",
        network.architectureName(), (System.nanoTime() - tGraph) / 1e9);

    Tokenizer tokenizer = Tokenizer.fromPretrained(path);
    LlmListeners.infof(io, null, "Model loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
    return new LlmModel(path, hfConfig, weights, network, tokenizer);
  }

  private static LlmModel loadGguf(
    final Path path,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) throws IOException {
    long t0 = System.nanoTime();
    LlmListeners.info(io, null, "CPU backend: " + MatmulRuntime.sequential().backendInfo());
    if (allowUnpackParameters) {
      LlmListeners.info(io, null,
        "GGUF: unpacking to float32 during load (mmap → dense; no packed heap copy).");
    } else {
      LlmListeners.info(io, null,
        "GGUF weights stay packed; use LlmModelFactory.make(path, io, true) or "
          + "LLM.Builder.allowUnpackParameters() for float32 speed.");
    }

    LoadedGguf loaded = GgufModelLoader.load(path, io, allowUnpackParameters);
    try (GgufReader reader = loaded.reader()) {
      WeightSchema schema = CausalLMFactory.schema(loaded.config());
      for (String required : schema.expectedParameters()) {
        if (!loaded.weights().has(required)) {
          throw new IllegalStateException("missing required GGUF weight: " + required);
        }
      }

      LlmListeners.info(io, null,
        "Building " + CausalLMFactory.detect(loaded.config()) + " model graph…");
      long tGraph = System.nanoTime();
      CausalLM network = CausalLMFactory.create(loaded.config(), loaded.weights());
      LlmListeners.infof(io, null, "Model graph ready (%s) in %.1fs%n",
        network.architectureName(), (System.nanoTime() - tGraph) / 1e9);

      Tokenizer tokenizer = Tokenizer.fromGguf(reader);
      LlmListeners.infof(io, null, "Model loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
      return new LlmModel(path, loaded.config(), loaded.weights(), network, tokenizer);
    }
  }
}
