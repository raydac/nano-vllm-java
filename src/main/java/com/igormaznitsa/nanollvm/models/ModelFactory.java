package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.CONFIG_JSON;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.GgufModelLoader;
import com.igormaznitsa.nanollvm.internal.GgufModelLoader.LoadedGguf;
import com.igormaznitsa.nanollvm.internal.GgufReader;
import com.igormaznitsa.nanollvm.internal.ModelLoader;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.EngineIo;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Loads an immutable {@link Model} from a HuggingFace model directory or a {@code .gguf} file.
 *
 * <p>One {@link Model} may be reused by any number of {@link LLM} instances.
 */
public final class ModelFactory {

  private ModelFactory() {
  }

  public static Model make(final Path modelDir) {
    return make(modelDir, EngineIo.silent());
  }

  public static Model make(final String modelPath) {
    return make(Path.of(requireNonNull(modelPath, "modelPath")));
  }

  public static Model make(final Path modelPath, final EngineIo io) {
    requireNonNull(modelPath, "modelPath");
    EngineIo streams = io == null ? EngineIo.silent() : io;
    Path path = modelPath.toAbsolutePath().normalize();
    try {
      if (Files.isRegularFile(path) && path.getFileName().toString().toLowerCase(Locale.ROOT)
        .endsWith(".gguf")) {
        return loadGguf(path, streams);
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

  private static Model loadHf(final Path path, final EngineIo io) throws IOException {
    long t0 = System.nanoTime();
    io.info("CPU backend: " + VectorMath.backendInfo());

    Config.HfConfig hfConfig = Config.HfConfig.load(path.resolve(CONFIG_JSON));
    String arch = CausalLMFactory.detect(hfConfig);
    WeightSchema schema = CausalLMFactory.schema(hfConfig);

    io.info("Loading " + arch + " weights…");
    WeightBag weights = ModelLoader.loadWeights(path, hfConfig, schema, io);

    io.info("Building " + arch + " model graph…");
    long tGraph = System.nanoTime();
    CausalLM network = CausalLMFactory.create(hfConfig, weights);
    io.infof("Model graph ready (%s) in %.1fs%n",
        network.architectureName(), (System.nanoTime() - tGraph) / 1e9);

    Tokenizer tokenizer = Tokenizer.fromPretrained(path);
    io.infof("Model loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
    return new Model(path, hfConfig, network, tokenizer);
  }

  private static Model loadGguf(final Path path, final EngineIo io) throws IOException {
    long t0 = System.nanoTime();
    io.info("CPU backend: " + VectorMath.backendInfo());
    io.info(
      "GGUF weights dequantize to float32 — expect large heap (default -Xmx16g in .mvn/jvm.config).");

    LoadedGguf loaded = GgufModelLoader.load(path, io);
    try (GgufReader reader = loaded.reader()) {
      WeightSchema schema = CausalLMFactory.schema(loaded.config());
      for (String required : schema.expectedParameters()) {
        if (!loaded.weights().has(required)) {
          throw new IllegalStateException("missing required GGUF weight: " + required);
        }
      }

      io.info("Building " + CausalLMFactory.detect(loaded.config()) + " model graph…");
      long tGraph = System.nanoTime();
      CausalLM network = CausalLMFactory.create(loaded.config(), loaded.weights());
      io.infof("Model graph ready (%s) in %.1fs%n",
        network.architectureName(), (System.nanoTime() - tGraph) / 1e9);

      Tokenizer tokenizer = Tokenizer.fromGguf(reader);
      io.infof("Model loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
      return new Model(path, loaded.config(), network, tokenizer);
    }
  }
}
