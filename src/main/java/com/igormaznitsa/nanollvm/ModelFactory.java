package com.igormaznitsa.nanollvm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.models.CausalLM;
import com.igormaznitsa.nanollvm.models.CausalLMFactory;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import com.igormaznitsa.nanollvm.utils.ModelLoader;

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

    Config.HfConfig hfConfig = Config.HfConfig.load(path.resolve("config.json"));
    String arch = CausalLMFactory.detect(hfConfig);
    io.info("Building " + arch + " model graph…");
    CausalLM network = CausalLMFactory.create(hfConfig);
    io.infof("Model graph ready (%s) in %.1fs%n",
        network.architectureName(), (System.nanoTime() - t0) / 1e9);

    ModelLoader.loadModel(network, path, io);
    network.seal();

    Tokenizer tokenizer = Tokenizer.fromPretrained(path);
    io.infof("Model loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
    return new Model(path, hfConfig, network, tokenizer);
  }
}
