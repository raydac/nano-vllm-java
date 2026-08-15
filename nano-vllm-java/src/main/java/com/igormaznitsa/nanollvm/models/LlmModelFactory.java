package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatSpecials;
import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.ModelFileBundle;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoder;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.llmarch.GgufModelLoader;
import com.igormaznitsa.nanollvm.models.llmarch.GgufModelLoader.LoadedGguf;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding;
import com.igormaznitsa.nanollvm.models.llmarch.ModelFill;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsTransport;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tokenizer.GgufTokenizerSource;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Loads a {@link LlmModel} from a HuggingFace model directory (safetensors or ONNX weights), a
 * {@code .gguf} file, or a {@link ModelFileSource} (classpath / custom streams read into heap; no
 * filesystem cache).
 *
 * <p>One {@link LlmModel} may be reused by any number of {@link LLM} instances until
 * {@link LlmModel#close()}. Load is blocking I/O on the calling thread; the returned model is safe
 * to share across threads while open. BERT embedding GGUFs load through the same entry points and
 * expose {@link LlmModel#embed(CharSequence)}.
 *
 * <p>GGUF stays packed by default. Pass {@code allowUnpackParameters=true} to dequantize to float32
 * during load (mmap or in-memory buffer → float tensors; no packed heap residency).
 *
 * <p>ONNX folders (<strong>since 1.1.0</strong>) use Tier A initializer import only (no ORT). When
 * both {@code *.safetensors} and {@code *.onnx} are present, safetensors wins (BERT embeddings use
 * ONNX when present; Hugging Face BERT safetensors is not supported). Stream loads reject
 * ONNX {@code external_data} sidecars — use {@link #make(Path)} for those exports.
 *
 * <p>Architecture is checked by {@link ModelSupport} after the container catalog is opened
 * ({@code GgufTransport} / {@code SafetensorsTransport} / {@code OnnxTransport} →
 * {@code ArchitectureProcessor} bind/fill/create).
 * Unsupported families throw
 * {@link com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException} with a
 * catalog of what this library can run.
 *
 * <p>Prefer {@link #open(Path)} (or {@link #open(ModelFileSource)}) for named load knobs:
 * {@code LlmModelFactory.open(path).listen(io).unpackParameters().thinkTags(tags).chatSpecials(specials).make()}.
 * Existing {@link #make} / {@link #fromClasspath} overloads remain.
 *
 * <p>Optional load-time settings go in a {@code Map} (frozen as {@link java.util.Map#copyOf} on
 * the model). Known keys: {@link LlmModel#OPTION_THINK_TAGS} ({@link ThinkTags}) and
 * {@link LlmModel#OPTION_CHAT_SPECIALS} ({@link ChatSpecials}). Omitted keys receive library
 * defaults. Unknown keys and wrong value types fail before weights are read.
 * {@link Builder#thinkTags} / {@link Builder#chatSpecials} set those keys without a map.
 *
 * <p>Filesystem {@link #make(Path)} overloads load directly from disk and do not route through
 * {@link ModelFileSource}.
 */
public final class LlmModelFactory {

  private LlmModelFactory() {
  }

  /**
   * Starts a fluent load from a filesystem HF folder or {@code .gguf} file.
   *
   * @since 1.1.0
   */
  public static Builder open(final Path modelPath) {
    return new Builder(requireNonNull(modelPath, "modelPath"), null);
  }

  /**
   * Starts a fluent load from a filesystem path string.
   *
   * @since 1.1.0
   */
  public static Builder open(final String modelPath) {
    return open(Path.of(requireNonNull(modelPath, "modelPath")));
  }

  /**
   * Starts a fluent load from a stream source (classpath / custom bytes).
   *
   * @since 1.1.0
   */
  public static Builder open(final ModelFileSource source) {
    return new Builder(null, requireNonNull(source, "source"));
  }

  /**
   * Starts a fluent load of an HF-style folder on the classpath.
   *
   * @since 1.1.0
   */
  public static Builder openClasspath(final ClassLoader loader, final String resourceFolder) {
    return open(ModelFileSources.classpath(loader, resourceFolder));
  }

  /**
   * Starts a fluent load of a single GGUF resource on the classpath.
   *
   * @since 1.1.0
   */
  public static Builder openClasspathGguf(final ClassLoader loader, final String ggufResourceFile) {
    return open(ModelFileSources.classpathGguf(loader, ggufResourceFile));
  }

  /**
   * Loads a model from disk with silent progress output and packed GGUF weights (when applicable).
   *
   * @param modelPath filesystem path to either an HF model <em>folder</em>
   *                  ({@code config.json} + {@code *.safetensors} or supported {@code *.onnx}) or a
   *                  single {@code .gguf} <em>file</em>; must exist
   * @return loaded model; caller must {@link LlmModel#close()} after all bound {@link LLM} engines
   * @throws NullPointerException if {@code modelPath} is {@code null}
   * @throws ModelLoadException   if the path is missing, unsupported, or load fails
   */
  public static LlmModel make(final Path modelPath) {
    return open(modelPath).make();
  }

  /**
   * {@link #make(Path)} with load-time {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel make(final Path modelPath, final Map<String, ?> options) {
    return open(modelPath).options(options).make();
  }

  /**
   * Loads a model from a filesystem path string (same rules as {@link #make(Path)}).
   *
   * @param modelPath absolute or relative path to an HF model <em>folder</em> or a {@code .gguf}
   *                  <em>file</em>; non-blank
   * @return loaded model; caller must {@link LlmModel#close()} after all bound {@link LLM} engines
   * @throws NullPointerException if {@code modelPath} is {@code null}
   * @throws ModelLoadException   if the path is missing, unsupported, or load fails
   */
  public static LlmModel make(final String modelPath) {
    return open(modelPath).make();
  }

  /**
   * {@link #make(String)} with load-time {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel make(final String modelPath, final Map<String, ?> options) {
    return open(modelPath).options(options).make();
  }

  /**
   * Loads a model from disk with progress/status events and packed GGUF weights (when applicable).
   *
   * @param modelPath filesystem path to an HF model <em>folder</em> or a {@code .gguf} <em>file</em>;
   *                  must exist
   * @param io        receives load progress ({@link LlmListeners#info}); {@code null} is treated as
   *                  {@link LlmListeners#silent()}
   * @return loaded model; caller must {@link LlmModel#close()} after all bound {@link LLM} engines
   * @throws NullPointerException if {@code modelPath} is {@code null}
   * @throws ModelLoadException   if the path is missing, unsupported, or load fails
   */
  public static LlmModel make(final Path modelPath, final LlmListener io) {
    return open(modelPath).listen(io).make();
  }

  /**
   * {@link #make(Path, LlmListener)} with load-time {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel make(
    final Path modelPath,
    final LlmListener io,
    final Map<String, ?> options
  ) {
    return open(modelPath).listen(io).options(options).make();
  }

  /**
   * Loads a model from disk with progress/status events and optional GGUF unpack-at-load.
   *
   * <p>Hugging Face <em>folders</em> always materialize dense float weights from safetensors. For a
   * {@code .gguf} <em>file</em>, {@code allowUnpackParameters=false} keeps quantized tensors packed
   * (lower peak RAM, slower matmul unless later unpacked via
   * {@link LLM.Builder#allowUnpackParameters()}); {@code true} dequantizes to float32 during this
   * call.
   *
   * @param modelPath             filesystem path to an HF model <em>folder</em> or a {@code .gguf}
   *                              <em>file</em>; must exist
   * @param io                    receives load progress; {@code null} → silent
   * @param allowUnpackParameters {@code true} to dequantize GGUF weights to float32 at load;
   *                              ignored for HF safetensors folders
   * @return loaded model; caller must {@link LlmModel#close()} after all bound {@link LLM} engines
   * @throws NullPointerException if {@code modelPath} is {@code null}
   * @throws ModelLoadException   if the path is missing, unsupported, or load fails
   */
  public static LlmModel make(
    final Path modelPath,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) {
    return open(modelPath).listen(io).unpackParameters(allowUnpackParameters).make();
  }

  /**
   * {@link #make(Path, LlmListener, boolean)} with load-time {@link LlmModel#options() options}.
   *
   * @param options known keys only (see {@link LlmModel#OPTION_THINK_TAGS},
   *                {@link LlmModel#OPTION_CHAT_SPECIALS}); must not be {@code null}
   * @since 1.1.0
   */
  public static LlmModel make(
    final Path modelPath,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, ?> options
  ) {
    return open(modelPath)
      .listen(io)
      .unpackParameters(allowUnpackParameters)
      .options(options)
      .make();
  }

  /**
   * Loads a model from a {@link ModelFileSource} with silent progress and packed GGUF (when
   * applicable). Bytes are read into heap memory; nothing is written to a disk cache.
   *
   * @param source supplies HF sidecars / safetensors (or {@link ModelFileId#GGUF}) via streams;
   *               non-{@code null}
   * @return loaded model; {@link LlmModel#path()} is a virtual label under {@code /nanollvm-memory}
   * @throws NullPointerException if {@code source} is {@code null}
   * @throws ModelLoadException   if required files are missing or load fails
   * @since 1.1.0
   */
  public static LlmModel make(final ModelFileSource source) {
    return open(source).make();
  }

  /**
   * {@link #make(ModelFileSource)} with load-time {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel make(final ModelFileSource source, final Map<String, ?> options) {
    return open(source).options(options).make();
  }

  /**
   * Loads a model from a {@link ModelFileSource} with progress events and packed GGUF (when
   * applicable). Bytes stay in heap; no filesystem cache.
   *
   * @param source supplies model file bytes; non-{@code null}
   * @param io     receives load progress; {@code null} → silent
   * @return loaded model; {@link LlmModel#path()} is a virtual in-memory label
   * @throws NullPointerException if {@code source} is {@code null}
   * @throws ModelLoadException   if required files are missing or load fails
   * @since 1.1.0
   */
  public static LlmModel make(final ModelFileSource source, final LlmListener io) {
    return open(source).listen(io).make();
  }

  /**
   * {@link #make(ModelFileSource, LlmListener)} with load-time {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel make(
    final ModelFileSource source,
    final LlmListener io,
    final Map<String, ?> options
  ) {
    return open(source).listen(io).options(options).make();
  }

  /**
   * Loads a model from a {@link ModelFileSource} with progress events and optional GGUF unpack.
   *
   * <p>Use {@link ModelFileSources#classpath(ClassLoader, String)} /
   * {@link ModelFileSources#classpathGguf(ClassLoader, String)} for JAR resources, or a custom
   * source when bytes come from elsewhere. Entire weight files are buffered in RAM before the
   * graph is built.
   *
   * @param source                supplies model file bytes; non-{@code null}
   * @param io                    receives load progress; {@code null} → silent
   * @param allowUnpackParameters {@code true} to dequantize GGUF to float32 at load; ignored for
   *                              HF safetensors folders
   * @return loaded model; {@link LlmModel#path()} is a virtual in-memory label
   * @throws NullPointerException if {@code source} is {@code null}
   * @throws ModelLoadException   if required files are missing or load fails
   * @since 1.1.0
   */
  public static LlmModel make(
    final ModelFileSource source,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) {
    return open(source).listen(io).unpackParameters(allowUnpackParameters).make();
  }

  /**
   * {@link #make(ModelFileSource, LlmListener, boolean)} with load-time
   * {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel make(
    final ModelFileSource source,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, ?> options
  ) {
    return open(source)
      .listen(io)
      .unpackParameters(allowUnpackParameters)
      .options(options)
      .make();
  }

  /**
   * Loads an HF-style model <em>folder</em> from the classpath (silent).
   *
   * @since 1.1.0
   */
  public static LlmModel fromClasspath(final ClassLoader loader, final String resourceFolder) {
    return openClasspath(loader, resourceFolder).make();
  }

  /**
   * {@link #fromClasspath(ClassLoader, String)} with load-time {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel fromClasspath(
    final ClassLoader loader,
    final String resourceFolder,
    final Map<String, ?> options
  ) {
    return openClasspath(loader, resourceFolder).options(options).make();
  }

  /**
   * Loads an HF-style model <em>folder</em> from the classpath with progress events.
   *
   * @param loader                class loader that can see the resources; non-{@code null}
   * @param resourceFolder        classpath <em>folder</em> prefix without a leading slash; non-blank
   * @param io                    receives load progress; {@code null} → silent
   * @param allowUnpackParameters unused for standard HF classpath folders (kept for API symmetry
   *                              with GGUF loaders)
   * @return loaded model buffered from classpath streams
   * @throws NullPointerException     if {@code loader} or {@code resourceFolder} is {@code null}
   * @throws IllegalArgumentException if {@code resourceFolder} is blank
   * @throws ModelLoadException       if required resources are missing or load fails
   * @since 1.1.0
   */
  public static LlmModel fromClasspath(
    final ClassLoader loader,
    final String resourceFolder,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) {
    return openClasspath(loader, resourceFolder)
      .listen(io)
      .unpackParameters(allowUnpackParameters)
      .make();
  }

  /**
   * {@link #fromClasspath(ClassLoader, String, LlmListener, boolean)} with load-time
   * {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel fromClasspath(
    final ClassLoader loader,
    final String resourceFolder,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, ?> options
  ) {
    return openClasspath(loader, resourceFolder)
      .listen(io)
      .unpackParameters(allowUnpackParameters)
      .options(options)
      .make();
  }

  /**
   * Loads a single GGUF <em>file</em> from the classpath (silent, weights stay packed).
   *
   * @param loader           class loader that can see the resource; non-{@code null}
   * @param ggufResourceFile exact classpath path to one {@code .gguf} <em>file</em> (e.g.
   *                         {@code models/gte-small.Q2_K.gguf}); no leading slash; non-blank
   * @return loaded model (causal or embedding, depending on GGUF architecture)
   * @throws NullPointerException     if {@code loader} or {@code ggufResourceFile} is {@code null}
   * @throws IllegalArgumentException if {@code ggufResourceFile} is blank
   * @throws ModelLoadException       if the resource is missing or load fails
   * @since 1.1.0
   */
  public static LlmModel fromClasspathGguf(final ClassLoader loader,
                                           final String ggufResourceFile) {
    return openClasspathGguf(loader, ggufResourceFile).make();
  }

  /**
   * {@link #fromClasspathGguf(ClassLoader, String)} with load-time {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel fromClasspathGguf(
    final ClassLoader loader,
    final String ggufResourceFile,
    final Map<String, ?> options
  ) {
    return openClasspathGguf(loader, ggufResourceFile).options(options).make();
  }

  /**
   * Loads a single GGUF <em>file</em> from the classpath with progress and optional unpack-at-load.
   *
   * @param loader                class loader that can see the resource; non-{@code null}
   * @param ggufResourceFile      exact classpath path to one {@code .gguf} <em>file</em>; no
   *                              leading slash; non-blank
   * @param io                    receives load progress; {@code null} → silent
   * @param allowUnpackParameters {@code true} to dequantize GGUF weights to float32 during load
   * @return loaded model (causal or embedding, depending on GGUF architecture)
   * @throws NullPointerException     if {@code loader} or {@code ggufResourceFile} is {@code null}
   * @throws IllegalArgumentException if {@code ggufResourceFile} is blank
   * @throws ModelLoadException       if the resource is missing or load fails
   * @since 1.1.0
   */
  public static LlmModel fromClasspathGguf(
    final ClassLoader loader,
    final String ggufResourceFile,
    final LlmListener io,
    final boolean allowUnpackParameters
  ) {
    return openClasspathGguf(loader, ggufResourceFile)
      .listen(io)
      .unpackParameters(allowUnpackParameters)
      .make();
  }

  /**
   * {@link #fromClasspathGguf(ClassLoader, String, LlmListener, boolean)} with load-time
   * {@link LlmModel#options() options}.
   *
   * @since 1.1.0
   */
  public static LlmModel fromClasspathGguf(
    final ClassLoader loader,
    final String ggufResourceFile,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, ?> options
  ) {
    return openClasspathGguf(loader, ggufResourceFile)
      .listen(io)
      .unpackParameters(allowUnpackParameters)
      .options(options)
      .make();
  }

  private static LlmModel loadPath(
    final Path modelPath,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, ?> options
  ) {
    requireNonNull(modelPath, "modelPath");
    Map<String, Object> frozen = LlmModel.copyAndValidateOptions(options);
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    Path path = modelPath.toAbsolutePath().normalize();
    try {
      if (isGgufFile(path)) {
        return loadGgufFile(path, streams, allowUnpackParameters, frozen);
      }
      if (!Files.isDirectory(path)) {
        throw new ModelLoadException(
          "model path is not an HF model folder (safetensors/ONNX) or .gguf file: " + path);
      }
      return loadHfFolder(path, streams, frozen);
    } catch (ModelLoadException e) {
      throw e;
    } catch (RuntimeException | IOException e) {
      throw new ModelLoadException("failed to load model from " + path, e);
    }
  }

  private static LlmModel loadSource(
    final ModelFileSource source,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, ?> options
  ) {
    requireNonNull(source, "source");
    Map<String, Object> frozen = LlmModel.copyAndValidateOptions(options);
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    try {
      ModelFileBundle bundle = ModelFileBundle.load(source, streams);
      if (bundle.isGguf()) {
        return loadGguf(bundle, streams, allowUnpackParameters, frozen);
      }
      return loadHf(bundle, streams, frozen);
    } catch (ModelLoadException e) {
      throw e;
    } catch (RuntimeException | IOException e) {
      throw new ModelLoadException("failed to load model from " + source.displayName(), e);
    }
  }

  private static LlmModel loadGgufFile(
    final Path ggufFile,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, Object> options
  ) throws IOException {
    long t0 = System.nanoTime();
    LlmListeners.info(io, null, "CPU backend: " + MatmulRuntime.sequential().backendInfo());
    if (allowUnpackParameters) {
      LlmListeners.info(io, null,
        "GGUF: unpacking to float32 during load (mmap → dense; no packed heap copy).");
    } else {
      LlmListeners.info(io, null,
        "GGUF weights stay packed; use LlmModelFactory.open(path).unpackParameters() or "
          + "LLM.Builder.allowUnpackParameters() for float32 speed.");
    }

    LoadedGguf loaded = GgufModelLoader.load(ggufFile, io, allowUnpackParameters);
    try (GgufTransport transport = loaded.transport()) {
      if (loaded.processor().isEmbedding()) {
        return loadGgufEmbedding(ggufFile, loaded, transport, io, t0, options);
      }
      return loadGgufCausal(ggufFile, loaded, transport, io, t0, options);
    }
  }

  private static LlmModel loadHfFolder(
    final Path modelFolder,
    final LlmListener io,
    final Map<String, Object> options
  )
    throws IOException {
    long t0 = System.nanoTime();
    LlmListeners.info(io, null, "CPU backend: " + MatmulRuntime.sequential().backendInfo());

    String configJson = Files.readString(modelFolder.resolve(CONFIG_JSON), UTF_8);
    boolean hasSafetensors = SafetensorsTransport.present(modelFolder);
    boolean hasOnnx = OnnxTransport.present(modelFolder);
    ModelSupport.Source source = resolveHfSource(
      hasSafetensors, hasOnnx, Config.HfConfig.parse(configJson), modelFolder.toString());
    try (ContainerTransport transport = openHfTransport(source, modelFolder, configJson)) {
      ModelBinding.BoundModel bound = ModelBinding.bind(transport.catalog());
      LlmListeners.info(io, null, "Loading " + bound.selection().architectureId() + " weights…");
      WeightBag weights = ModelFill.fill(transport, bound, io, false);
      Tokenizer tokenizer = Tokenizer.fromPretrained(modelFolder);
      return finishLoadedModel(
        modelFolder, bound, weights, tokenizer, io, t0, options);
    }
  }

  private static LlmModel loadHf(
    final ModelFileBundle bundle,
    final LlmListener io,
    final Map<String, Object> options
  )
    throws IOException {
    long t0 = System.nanoTime();
    LlmListeners.info(io, null, "CPU backend: " + MatmulRuntime.sequential().backendInfo());

    String configJson = bundle.configJson();
    boolean hasSafetensors = !bundle.safetensors().isEmpty();
    boolean hasOnnx = !bundle.onnx().isEmpty();
    ModelSupport.Source source = resolveHfSource(
      hasSafetensors, hasOnnx, Config.HfConfig.parse(configJson), bundle.displayName());
    try (ContainerTransport transport = openHfTransport(source, bundle, configJson)) {
      ModelBinding.BoundModel bound = ModelBinding.bind(transport.catalog());
      LlmListeners.info(io, null, "Loading " + bound.selection().architectureId() + " weights…");
      WeightBag weights = ModelFill.fill(transport, bound, io, false);
      Tokenizer tokenizer = Tokenizer.fromJsonDocuments(
        bundle.textFile(ModelFileId.TOKENIZER).orElse(null),
        bundle.textFile(ModelFileId.TOKENIZER_CONFIG).orElse(null),
        bundle.textFile(ModelFileId.GENERATION_CONFIG).orElse(null),
        bundle.configJson());
      return finishLoadedModel(
        bundle.virtualPath(), bound, weights, tokenizer, io, t0, options);
    }
  }

  private static ContainerTransport openHfTransport(
    final ModelSupport.Source source,
    final Path modelFolder,
    final String configJson
  ) throws IOException {
    return switch (source) {
      case HF_SAFETENSORS -> SafetensorsTransport.open(modelFolder, configJson);
      case ONNX -> OnnxTransport.open(modelFolder, configJson);
      case GGUF -> throw new ModelLoadException("HF folder cannot be a GGUF container");
    };
  }

  private static ContainerTransport openHfTransport(
    final ModelSupport.Source source,
    final ModelFileBundle bundle,
    final String configJson
  ) throws IOException {
    return switch (source) {
      case HF_SAFETENSORS -> SafetensorsTransport.open(
        bundle.safetensors(), configJson, bundle.displayName());
      case ONNX -> {
        ModelFileBundle.NamedBytes primary = bundle.onnx().getFirst();
        yield OnnxTransport.open(primary.bytes(), primary.name(), configJson);
      }
      case GGUF -> throw new ModelLoadException("HF bundle cannot be a GGUF container");
    };
  }

  private static LlmModel finishLoadedModel(
    final Path modelPath,
    final ModelBinding.BoundModel bound,
    final WeightBag weights,
    final Tokenizer tokenizer,
    final LlmListener io,
    final long startedAtNanos,
    final Map<String, Object> options
  ) {
    String arch = bound.selection().architectureId();
    Config.HfConfig hfConfig = bound.config();
    if (bound.processor().isEmbedding()) {
      LlmListeners.info(io, null, "Building " + arch + " embedding graph…");
      long tGraph = System.nanoTime();
      EmbeddingEncoder encoder = bound.processor().createEmbedding(hfConfig, weights);
      LlmListeners.infof(io, null, "Embedding graph ready (%s) in %.1fs%n",
        encoder.architectureName(), (System.nanoTime() - tGraph) / 1e9);
      LlmListeners.infof(io, null, "Model loaded in %.1fs%n",
        (System.nanoTime() - startedAtNanos) / 1e9);
      return new LlmModel(modelPath, hfConfig, weights, encoder, tokenizer, options);
    }

    LlmListeners.info(io, null, "Building " + arch + " model graph…");
    long tGraph = System.nanoTime();
    CausalLM network = bound.processor().createCausal(hfConfig, weights);
    LlmListeners.infof(io, null, "Model graph ready (%s) in %.1fs%n",
      network.architectureName(), (System.nanoTime() - tGraph) / 1e9);
    LlmListeners.infof(io, null, "Model loaded in %.1fs%n",
      (System.nanoTime() - startedAtNanos) / 1e9);
    return new LlmModel(modelPath, hfConfig, weights, network, tokenizer, options);
  }

  private static ModelSupport.Source resolveHfSource(
    final boolean hasSafetensors,
    final boolean hasOnnx,
    final Config.HfConfig hfConfig,
    final String label
  ) {
    if (hasSafetensors && !ModelSupport.isEmbedding(hfConfig)) {
      return ModelSupport.Source.HF_SAFETENSORS;
    }
    if (hasOnnx) {
      return ModelSupport.Source.ONNX;
    }
    if (hasSafetensors) {
      return ModelSupport.Source.HF_SAFETENSORS;
    }
    throw new ModelLoadException(
      "model has no *.safetensors or supported *.onnx weights: " + label
        + System.lineSeparator() + System.lineSeparator() + ModelSupport.CATALOG);
  }

  private static LlmModel loadGguf(
    final ModelFileBundle bundle,
    final LlmListener io,
    final boolean allowUnpackParameters,
    final Map<String, Object> options
  ) throws IOException {
    long t0 = System.nanoTime();
    LlmListeners.info(io, null, "CPU backend: " + MatmulRuntime.sequential().backendInfo());
    if (allowUnpackParameters) {
      LlmListeners.info(io, null,
        "GGUF: unpacking to float32 during load (memory buffer → dense).");
    } else {
      LlmListeners.info(io, null,
        "GGUF weights stay packed; use LlmModelFactory.open(source).unpackParameters() or "
          + "LLM.Builder.allowUnpackParameters() for float32 speed.");
    }

    LoadedGguf loaded = GgufModelLoader.load(
      bundle.ggufBuffer(), bundle.virtualPath(), io, allowUnpackParameters);
    try (GgufTransport transport = loaded.transport()) {
      if (loaded.processor().isEmbedding()) {
        return loadGgufEmbedding(bundle.virtualPath(), loaded, transport, io, t0, options);
      }
      return loadGgufCausal(bundle.virtualPath(), loaded, transport, io, t0, options);
    }
  }

  private static boolean isGgufFile(final Path path) {
    Path name = path.getFileName();
    return Files.isRegularFile(path)
      && name != null
      && name.toString().toLowerCase(Locale.ROOT).endsWith(".gguf");
  }

  private static LlmModel loadGgufCausal(
    final Path modelPath,
    final LoadedGguf loaded,
    final GgufTokenizerSource tokenizerSource,
    final LlmListener io,
    final long startedAtNanos,
    final Map<String, Object> options
  ) {
    LlmListeners.info(io, null,
      "Building " + loaded.processor().architectureId() + " model graph…");
    long tGraph = System.nanoTime();
    CausalLM network = loaded.processor().createCausal(loaded.config(), loaded.weights());
    LlmListeners.infof(io, null, "Model graph ready (%s) in %.1fs%n",
      network.architectureName(), (System.nanoTime() - tGraph) / 1e9);

    Tokenizer tokenizer = Tokenizer.fromGguf(tokenizerSource);
    LlmListeners.infof(io, null, "Model loaded in %.1fs%n",
      (System.nanoTime() - startedAtNanos) / 1e9);
    return new LlmModel(modelPath, loaded.config(), loaded.weights(), network, tokenizer, options);
  }

  private static LlmModel loadGgufEmbedding(
    final Path modelPath,
    final LoadedGguf loaded,
    final GgufTokenizerSource tokenizerSource,
    final LlmListener io,
    final long startedAtNanos,
    final Map<String, Object> options
  ) {
    LlmListeners.info(io, null,
      "Building " + loaded.processor().architectureId() + " embedding graph…");
    long tGraph = System.nanoTime();
    EmbeddingEncoder encoder =
      loaded.processor().createEmbedding(loaded.config(), loaded.weights());
    LlmListeners.infof(io, null, "Embedding graph ready (%s) in %.1fs%n",
      encoder.architectureName(), (System.nanoTime() - tGraph) / 1e9);

    Tokenizer tokenizer = Tokenizer.fromGguf(tokenizerSource);
    LlmListeners.infof(io, null, "Model loaded in %.1fs%n",
      (System.nanoTime() - startedAtNanos) / 1e9);
    return new LlmModel(modelPath, loaded.config(), loaded.weights(), encoder, tokenizer, options);
  }

  /**
   * Fluent load configurator. Terminal {@link #make()} performs blocking I/O.
   *
   * @since 1.1.0
   */
  public static final class Builder {

    private final Path modelPath;
    private final ModelFileSource source;
    private LlmListener io = LlmListeners.silent();
    private boolean allowUnpackParameters;
    private Map<String, ?> options = Map.of();

    private Builder(final Path modelPath, final ModelFileSource source) {
      this.modelPath = modelPath;
      this.source = source;
    }

    /**
     * Replaces the load listener ({@code null} → silent).
     *
     * @since 1.1.0
     */
    public Builder listen(final LlmListener listener) {
      this.io = listener == null ? LlmListeners.silent() : listener;
      return this;
    }

    /**
     * Unpacks GGUF weights to float32 during load.
     *
     * @since 1.1.0
     */
    public Builder unpackParameters() {
      return this.unpackParameters(true);
    }

    /**
     * When {@code true}, unpacks GGUF weights to float32 during load.
     *
     * @since 1.1.0
     */
    public Builder unpackParameters(final boolean value) {
      this.allowUnpackParameters = value;
      return this;
    }

    /**
     * Sets {@link LlmModel#OPTION_THINK_TAGS} for this load.
     *
     * @since 1.1.0
     */
    public Builder thinkTags(final ThinkTags thinkTags) {
      requireNonNull(thinkTags, "thinkTags");
      return this.withOption(LlmModel.OPTION_THINK_TAGS, thinkTags);
    }

    /**
     * Sets {@link LlmModel#OPTION_CHAT_SPECIALS} for this load.
     *
     * @since 1.1.0
     */
    public Builder chatSpecials(final ChatSpecials chatSpecials) {
      requireNonNull(chatSpecials, "chatSpecials");
      return this.withOption(LlmModel.OPTION_CHAT_SPECIALS, chatSpecials);
    }

    /**
     * Replaces the load-time options map (omitted known keys receive library defaults).
     *
     * @since 1.1.0
     */
    public Builder options(final Map<String, ?> options) {
      this.options = requireNonNull(options, "options");
      return this;
    }

    private Builder withOption(final String key, final Object value) {
      Map<String, Object> merged = new LinkedHashMap<>(this.options);
      merged.put(key, value);
      this.options = merged;
      return this;
    }

    /**
     * Loads the checkpoint (blocking I/O).
     *
     * @since 1.1.0
     */
    public LlmModel make() {
      if (this.source != null) {
        return loadSource(this.source, this.io, this.allowUnpackParameters, this.options);
      }
      return loadPath(this.modelPath, this.io, this.allowUnpackParameters, this.options);
    }
  }
}
