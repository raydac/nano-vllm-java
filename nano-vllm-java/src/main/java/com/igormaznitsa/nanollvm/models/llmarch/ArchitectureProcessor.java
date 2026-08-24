package com.igormaznitsa.nanollvm.models.llmarch;

import static com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog.META_CONFIG_JSON;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.CausalLM;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoder;
import com.igormaznitsa.nanollvm.models.internal.SpeechToText;
import com.igormaznitsa.nanollvm.models.internal.TextToSpeech;
import com.igormaznitsa.nanollvm.models.internal.WeightBag;
import com.igormaznitsa.nanollvm.models.internal.WeightSchema;
import com.igormaznitsa.nanollvm.models.llmarch.ModelBinding.BoundModel;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerCatalog;
import com.igormaznitsa.nanollvm.models.llmcontainer.ContainerTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.SafetensorsTransport;
import java.io.IOException;
import java.util.function.Function;

/**
 * Architecture layer above {@link ContainerTransport}: decode a catalog into config + schema,
 * fill a {@link WeightBag} from transport payloads, and construct the target graph.
 *
 * <p>Chat families share {@link CausalArchitecture}; embedding families share
 * {@link EmbeddingArchitecture}; speech families share {@link SpeechArchitecture}; synthesis
 * families share {@link SynthesisArchitecture}. Application code loads through
 * {@link com.igormaznitsa.nanollvm.models.LlmModelFactory}.
 *
 * @since 1.1.0
 */
public sealed interface ArchitectureProcessor
  permits CausalArchitecture, EmbeddingArchitecture, SpeechArchitecture, SynthesisArchitecture {

  /**
   * Parses {@code config.json} stored on an HF / ONNX catalog.
   *
   * @param catalog container snapshot that must carry {@link ContainerCatalog#META_CONFIG_JSON}
   * @return parsed Hugging Face config
   * @throws IllegalStateException if the JSON sidecar is missing
   * @since 1.1.0
   */
  static Config.HfConfig hfConfig(final ContainerCatalog catalog) {
    String json = catalog.metaString(META_CONFIG_JSON, "");
    if (json.isBlank()) {
      throw new IllegalStateException(
        "missing config.json in container catalog: " + catalog.label());
    }
    return Config.HfConfig.parse(json);
  }

  /**
   * Canonical backend id ({@code qwen3}, {@code gemma3}, {@code gemma4}, {@code llama},
   * {@code lfm2}, {@code bert}, {@code whisper}).
   *
   * @since 1.1.0
   */
  String architectureId();

  /**
   * {@code true} for sentence-embedding encoders; {@code false} for chat and speech.
   *
   * @since 1.1.0
   */
  boolean isEmbedding();

  /**
   * {@code true} for speech-to-text graphs.
   *
   * @since 1.3.0
   */
  default boolean isSpeech() {
    return false;
  }

  /**
   * {@code true} for text-to-speech graphs.
   *
   * @since 1.3.0
   */
  default boolean isSynthesis() {
    return false;
  }

  /**
   * Decodes {@code catalog} into config + weight schema for {@code selected}.
   *
   * @param catalog  transport snapshot (GGUF, safetensors, or ONNX)
   * @param selected family already accepted by {@link ModelSupport}
   * @return bound config, schema, and this processor
   * @since 1.1.0
   */
  BoundModel bind(ContainerCatalog catalog, ModelSupport.Selection selected);

  /**
   * Reads payloads through {@code transport} into an immutable weight bag.
   *
   * <p>Default handles GGUF, safetensors, and ONNX. Families that accept only one container
   * override this (Gemma 4 QAT safetensors, LFM2 GGUF).
   *
   * @param transport       open container (must match the catalog used at {@link #bind})
   * @param bound           result of {@link #bind}
   * @param io              load progress; {@code null} is treated as silent
   * @param allowUnpackGguf when {@code true}, GGUF packed tensors are expanded to float32
   * @return filled bag for {@link #createCausal} / {@link #createEmbedding}
   * @throws IOException if a payload cannot be read
   * @since 1.1.0
   */
  default WeightBag fill(
    final ContainerTransport transport,
    final BoundModel bound,
    final LlmListener io,
    final boolean allowUnpackGguf
  ) throws IOException {
    requireNonNull(transport, "transport");
    requireNonNull(bound, "bound");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    return switch (transport) {
      case GgufTransport gguf -> ArchitectureFills.gguf(gguf, bound, streams, allowUnpackGguf);
      case SafetensorsTransport safetensors -> ModelLoader.fill(safetensors, bound, streams);
      case OnnxTransport onnx -> ArchitectureFills.onnx(onnx, bound, streams, this.isEmbedding());
    };
  }

  /**
   * Builds the causal chat graph. Embedding processors throw.
   *
   * @param config  bound Hugging Face / GGUF-derived config
   * @param weights filled parameter bag
   * @return immutable decoder
   * @throws IllegalStateException if this family is not chat
   * @since 1.1.0
   */
  default CausalLM createCausal(final Config.HfConfig config, final WeightBag weights) {
    throw new IllegalStateException(this.architectureId() + " is not a chat architecture");
  }

  /**
   * Builds the embedding encoder graph. Chat processors throw.
   *
   * @param config  bound Hugging Face / GGUF-derived config
   * @param weights filled parameter bag
   * @return immutable encoder
   * @throws IllegalStateException if this family is not an embedding encoder
   * @since 1.1.0
   */
  default EmbeddingEncoder createEmbedding(final Config.HfConfig config, final WeightBag weights) {
    throw new IllegalStateException(this.architectureId() + " is not an embedding architecture");
  }

  /**
   * Builds the speech-to-text graph. Chat and embedding processors throw.
   *
   * @param config  bound Hugging Face config
   * @param weights filled parameter bag
   * @return immutable ASR graph
   * @throws IllegalStateException if this family is not speech
   * @since 1.3.0
   */
  default SpeechToText createSpeech(final Config.HfConfig config, final WeightBag weights) {
    throw new IllegalStateException(this.architectureId() + " is not a speech architecture");
  }

  /**
   * Builds the text-to-speech graph. Chat, embedding, and speech processors throw.
   *
   * @param config  bound Hugging Face / Piper config
   * @param weights filled parameter bag
   * @return immutable TTS graph
   * @throws IllegalStateException if this family is not synthesis
   * @since 1.3.0
   */
  default TextToSpeech createSynthesis(final Config.HfConfig config, final WeightBag weights) {
    throw new IllegalStateException(this.architectureId() + " is not a synthesis architecture");
  }

  /**
   * GGUF catalog → {@link #bindGguf}; otherwise Hugging Face {@code config.json} → {@link #bindHf}.
   *
   * @param catalog    transport snapshot
   * @param selected   family already accepted by {@link ModelSupport}
   * @param ggufConfig maps a GGUF catalog onto {@link Config.HfConfig}
   * @return bound config, schema, and this processor
   * @since 1.1.0
   */
  default BoundModel bindDualSource(
    final ContainerCatalog catalog,
    final ModelSupport.Selection selected,
    final Function<ContainerCatalog, Config.HfConfig> ggufConfig
  ) {
    requireNonNull(catalog, "catalog");
    requireNonNull(selected, "selected");
    requireNonNull(ggufConfig, "ggufConfig");
    if (catalog.source() == ModelSupport.Source.GGUF) {
      return this.bindGguf(selected, catalog, ggufConfig.apply(catalog));
    }
    return this.bindHf(selected, ArchitectureProcessor.hfConfig(catalog), catalog.source());
  }

  /**
   * Binds an HF / ONNX catalog using {@link WeightSchema#forArchitecture}.
   *
   * @param selected family already accepted by {@link ModelSupport}
   * @param config   parsed {@code config.json}
   * @param source   {@link ModelSupport.Source#HF_SAFETENSORS} or {@link ModelSupport.Source#ONNX}
   * @return bound config, schema, and this processor
   * @since 1.1.0
   */
  default BoundModel bindHf(
    final ModelSupport.Selection selected,
    final Config.HfConfig config,
    final ModelSupport.Source source
  ) {
    requireNonNull(selected, "selected");
    requireNonNull(config, "config");
    requireNonNull(source, "source");
    return new BoundModel(
      selected, config, WeightSchema.forArchitecture(this.architectureId(), config), this);
  }

  /**
   * Binds a GGUF catalog using {@link WeightSchema#forGguf} and checks required tensor names.
   *
   * @param selected family already accepted by {@link ModelSupport}
   * @param catalog  GGUF snapshot
   * @param config   GGUF metadata mapped onto {@link Config.HfConfig}
   * @return bound config, schema, and this processor
   * @since 1.1.0
   */
  default BoundModel bindGguf(
    final ModelSupport.Selection selected,
    final ContainerCatalog catalog,
    final Config.HfConfig config
  ) {
    requireNonNull(selected, "selected");
    requireNonNull(catalog, "catalog");
    requireNonNull(config, "config");
    BoundModel bound = new BoundModel(
      selected, config, WeightSchema.forGguf(this.architectureId(), config), this);
    bound.requireCatalogTensors(catalog);
    return bound;
  }
}
