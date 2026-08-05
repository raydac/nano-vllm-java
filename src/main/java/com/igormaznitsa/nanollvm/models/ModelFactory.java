package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.CONFIG_JSON;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.ModelLoader;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.EngineIo;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Loads an immutable {@link Model} from a HuggingFace model directory.
 *
 * <p>One {@link Model} may be reused by any number of {@link LLM} instances.
 */
public final class ModelFactory {

  private ModelFactory() {
  }

  public static Model make(Path modelDir) {
    return make(modelDir, EngineIo.silent());
  }

  public static Model make(String modelPath) {
    return make(Path.of(requireNonNull(modelPath, "modelPath")));
  }

  public static Model make(Path modelDir, EngineIo io) {
    requireNonNull(modelDir, "modelDir");
    EngineIo streams = io == null ? EngineIo.silent() : io;
    Path path = modelDir.toAbsolutePath().normalize();
    if (!Files.isDirectory(path)) {
      throw new ModelLoadException("model path is not a directory: " + path);
    }

    try {
      return load(path, streams);
    } catch (ModelLoadException e) {
      throw e;
    } catch (RuntimeException | IOException e) {
      throw new ModelLoadException("failed to load model from " + path, e);
    }
  }

  private static Model load(Path path, EngineIo io) throws IOException {
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
}
