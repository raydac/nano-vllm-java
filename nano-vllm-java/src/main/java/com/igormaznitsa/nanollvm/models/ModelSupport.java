package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA4;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_WHISPER;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_ARCH;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.llmcontainer.GgufReader;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * Exact architecture detection and user-facing support catalog. Substring matching is
 * intentionally not used ({@code qwen3_5} is not {@code qwen3}).
 *
 * @since 1.1.0
 */
public final class ModelSupport {

  /**
   * User-facing list of architectures this library can run (and common look-alikes it rejects).
   *
   * @since 1.1.0
   */
  public static final String CATALOG = """
    Supported by this library:
      Chat from a Hugging Face folder (config.json + *.safetensors or *.onnx): qwen3, gemma3 / gemma3_text, \
    gemma4 (text / QAT mobile), llama
      Chat from a GGUF file: qwen3, lfm2
      Embeddings from GGUF or ONNX: bert encoder (bert / roberta / xlm-roberta)
      Speech from a Hugging Face folder (config.json + *.safetensors): whisper (openai/whisper-*)
    Not supported: Qwen2 / Qwen2.5, Qwen3.5 / Qwen3-Next / Fara, vision-language models, Gemma 1 / 2, \
    Gemma 4 vision/audio towers, Mistral / Mixtral, Phi, MoE, GGUF Llama / Gemma, Hugging Face BERT safetensors, \
    CTranslate2 / faster-whisper model.bin, Whisper GGUF / ONNX.""";

  private static final Pattern HF_CLASS_SUFFIX = Pattern.compile(
    "(ForCausalLM|ForConditionalGeneration|ForSequenceClassification|ForMaskedLM"
      + "|ForQuestionAnswering|ForTokenClassification|ForImageTextToText|ForVision2Seq|Model)$");

  private ModelSupport() {
  }

  /**
   * Resolves {@code config} and checks that {@code source} can carry that architecture
   * (e.g. Gemma3 / Llama chat are not loaded from GGUF).
   *
   * @return selected backend id and {@link Kind}
   * @throws com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException if the family is
   *                                                                        unknown or the source is incompatible
   * @since 1.1.0
   */
  public static Selection require(final Config.HfConfig config, final Source source) {
    Selection selected = resolve(requireNonNull(config, "config"));
    rejectIncompatibleSource(selected, requireNonNull(source, "source"), config);
    return selected;
  }

  /**
   * Resolves a GGUF {@code general.architecture} string ({@code qwen3} / {@code lfm2} chat or
   * BERT-encoder embeddings).
   *
   * @param generalArchitecture GGUF metadata architecture; {@code null} treated as missing
   * @return selected backend id and {@link Kind}
   * @throws com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException if unrecognized or
   *                                                                        unsupported
   * @since 1.1.0
   */
  public static Selection requireGguf(final String generalArchitecture) {
    String raw = generalArchitecture == null ? "" : generalArchitecture;
    Verdict verdict = classifyToken(normalize(raw));
    if (verdict == null) {
      throw unsupported(
        "GGUF architecture '%s' is not recognized.".formatted(blank(raw)),
        raw,
        List.of());
    }
    if (!verdict.supported()) {
      throw unsupported(verdict.detail(), raw, List.of());
    }
    Selection selected = applyForcedArchitecture(verdict.selection(), raw, List.of());
    rejectIncompatibleSource(selected, Source.GGUF, null);
    return selected;
  }

  /**
   * Classifies an HF {@code config.json} without checking the weight-file source.
   *
   * @return selected backend id and {@link Kind}
   * @throws com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException if the family is
   *                                                                        unknown or structurally unsupported
   * @since 1.1.0
   */
  public static Selection resolve(final Config.HfConfig config) {
    requireNonNull(config, "config");
    Verdict verdict = inspect(config);
    if (!verdict.supported()) {
      throw unsupported(verdict.detail(), config.modelType(), config.architectures());
    }
    return applyForcedArchitecture(verdict.selection(), config.modelType(), config.architectures());
  }

  /**
   * {@code true} when {@code config} is a BERT encoder this library can run ({@code bert},
   * {@code roberta}, {@code xlm-roberta}, and the same graph under those family names).
   * Unknown or chat families return {@code false} (they do not throw).
   *
   * @since 1.1.0
   */
  public static boolean isEmbedding(final Config.HfConfig config) {
    Verdict verdict = inspect(requireNonNull(config, "config"));
    return verdict.supported() && verdict.selection().isEmbedding();
  }

  /**
   * {@code true} when {@code path} is a BERT-encoder checkpoint (HF {@code config.json} or GGUF
   * metadata) without loading weights. Unknown, chat, or unreadable paths return {@code false}.
   *
   * @since 1.2.1
   */
  public static boolean isEmbeddingCheckpoint(final Path path) {
    Path file = requireNonNull(path, "path").toAbsolutePath().normalize();
    if (isGgufFile(file)) {
      return isEmbeddingGguf(file);
    }
    Path config = configJson(file);
    if (!Files.isRegularFile(config)) {
      return false;
    }
    try {
      return isEmbedding(Config.HfConfig.parse(Files.readString(config, UTF_8)));
    } catch (IOException | RuntimeException ignored) {
      return false;
    }
  }

  private static Path configJson(final Path file) {
    Path name = file.getFileName();
    if (Files.isRegularFile(file) && name != null && "config.json".equals(name.toString())) {
      return file;
    }
    return file.resolve("config.json");
  }

  private static boolean isGgufFile(final Path file) {
    Path name = file.getFileName();
    return Files.isRegularFile(file)
      && name != null
      && name.toString().toLowerCase(ROOT).endsWith(".gguf");
  }

  private static boolean isEmbeddingGguf(final Path file) {
    try {
      Verdict verdict = classifyToken(normalize(GgufReader.peekArchitecture(file)));
      return verdict != null && verdict.supported() && verdict.selection().isEmbedding();
    } catch (IOException | RuntimeException ignored) {
      return false;
    }
  }

  /**
   * {@code true} when {@code config} is OpenAI Whisper speech-to-text this library can run.
   * Unknown or chat/embedding families return {@code false} (they do not throw).
   *
   * @since 1.3.0
   */
  public static boolean isSpeech(final Config.HfConfig config) {
    Verdict verdict = inspect(requireNonNull(config, "config"));
    return verdict.supported() && verdict.selection().isSpeech();
  }

  /**
   * {@code true} when {@code path} is a Whisper safetensors folder, without loading weights.
   * GGUF, ONNX, CTranslate2, unknown, and chat/embedding paths return {@code false}.
   *
   * @since 1.3.0
   */
  public static boolean isSpeechCheckpoint(final Path path) {
    Path file = requireNonNull(path, "path").toAbsolutePath().normalize();
    if (isGgufFile(file)) {
      return false;
    }
    Path config = configJson(file);
    if (!Files.isRegularFile(config)) {
      return false;
    }
    try {
      return isSpeech(Config.HfConfig.parse(Files.readString(config, UTF_8)));
    } catch (IOException | RuntimeException ignored) {
      return false;
    }
  }

  /**
   * Message when {@link com.igormaznitsa.nanollvm.llm.LLM#builder} is used on an embedding checkpoint.
   *
   * @since 1.1.0
   */
  public static String chatMisuseMessage(final String architectureName) {
    return ("This checkpoint is a %s embedding encoder, not a chat model. Call LlmModel.embed(...) "
      + "instead of LLM.builder / generate.%n%n%s").formatted(blank(architectureName), CATALOG);
  }

  /**
   * Message when {@link LlmModel#embed} is used on a chat checkpoint.
   *
   * @since 1.1.0
   */
  public static String embedMisuseMessage(final String architectureName) {
    return ("This checkpoint is a %s chat model, not an embedding encoder. Use LLM.builder(model) "
      + "for generate / chat.%n%n%s").formatted(blank(architectureName), CATALOG);
  }

  /**
   * Message when {@link com.igormaznitsa.nanollvm.llm.LLM#builder} is used on a speech checkpoint.
   *
   * @since 1.3.0
   */
  public static String speechEngineMisuseMessage(final String architectureName) {
    return ("This checkpoint is a %s speech model, not a chat model. Call LlmModel.transcribe(...) "
      + "instead of LLM.builder / generate.%n%n%s").formatted(blank(architectureName), CATALOG);
  }

  /**
   * Message when {@link LlmModel#embed} is used on a speech checkpoint.
   *
   * @since 1.3.0
   */
  public static String speechEmbedMisuseMessage(final String architectureName) {
    return ("This checkpoint is a %s speech model, not an embedding encoder. Call "
      + "LlmModel.transcribe(...).%n%n%s").formatted(blank(architectureName), CATALOG);
  }

  /**
   * Message when {@link LlmModel#transcribe} is used on a chat or embedding checkpoint.
   *
   * @since 1.3.0
   */
  public static String transcribeMisuseMessage(final String architectureName) {
    return ("This checkpoint is a %s model, not Whisper speech-to-text. Chat uses LLM.builder; "
      + "embeddings use LlmModel.embed.%n%n%s").formatted(blank(architectureName), CATALOG);
  }

  private static Verdict inspect(final Config.HfConfig config) {
    Verdict fromType = classifyToken(normalize(config.modelType()));
    if (fromType != null && !fromType.supported()) {
      return fromType;
    }

    Verdict fromClasses = inspectArchitectureClasses(config.architectures());
    if (fromClasses != null && !fromClasses.supported()) {
      return fromClasses;
    }

    Verdict family = fromType != null ? fromType : fromClasses;
    if (family == null) {
      return Verdict.rejected(unknownReason(config));
    }
    if (family.selection().kind() == Kind.CHAT) {
      Optional<String> structural = structuralChatRejection(config);
      if (structural.isPresent()) {
        return Verdict.rejected(structural.get());
      }
    }
    return family;
  }

  private static Verdict inspectArchitectureClasses(final List<String> architectures) {
    Verdict firstSupported = null;
    for (String raw : architectures) {
      if (raw == null || raw.isBlank()) {
        continue;
      }
      Verdict fromName = classifyToken(fromArchitectureClass(raw));
      if (fromName != null && !fromName.supported()) {
        return fromName;
      }
      if (isMultimodalClass(raw.toLowerCase(ROOT))) {
        if (fromName != null && (ARCH_GEMMA4.equals(fromName.selection().architectureId())
          || ARCH_WHISPER.equals(fromName.selection().architectureId()))) {
          if (firstSupported == null) {
            firstSupported = fromName;
          }
          continue;
        }
        return Verdict.rejected(multimodalReason(raw));
      }
      if (fromName != null && firstSupported == null) {
        firstSupported = fromName;
      }
    }
    return firstSupported;
  }

  private static Optional<String> structuralChatRejection(final Config.HfConfig config) {
    if (config.isGemma4()) {
      if (config.gemma4().enableMoeBlock()) {
        return Optional.of(
          "Gemma 4 Mixture-of-Experts checkpoints are not implemented. This library loads the "
            + "dense E2B text decoder (including QAT mobile).");
      }
      return Optional.empty();
    }
    if (config.visionConfigPresent() || config.audioConfigPresent()) {
      return Optional.of(multimodalReason(config.modelType()));
    }
    if (config.nestedTextConfig()) {
      return Optional.of(
        "This checkpoint uses a nested text_config / composite layout (typical of Qwen3.5 and "
          + "vision-language models), which this library does not load.");
    }
    if (config.hasLinearAttentionLayers()) {
      return Optional.of(
        "This checkpoint declares linear_attention layers (Gated DeltaNet / Qwen3.5-style hybrid). "
          + "This library only implements dense GQA transformer blocks.");
    }
    return Optional.empty();
  }

  private static Selection applyForcedArchitecture(
    final Selection detected,
    final String modelType,
    final List<String> architectures
  ) {
    String forced = Optional.ofNullable(NanoLlvmProps.systemProperty(PROP_ARCH))
      .orElse("")
      .strip()
      .toLowerCase(ROOT);
    if (forced.isEmpty()) {
      return detected;
    }
    String forcedId = normalizeForcedAlias(forced);
    if (forcedId == null) {
      throw unsupported(
        "'%s' is not a valid -D%s value (use qwen3, gemma3, gemma4, llama, lfm2, bert, or whisper)."
          .formatted(forced, PROP_ARCH),
        modelType,
        architectures);
    }
    if (!forcedId.equals(detected.architectureId())) {
      throw unsupported(
        "Cannot apply -D%s=%s: this checkpoint is '%s', not '%s'."
          .formatted(PROP_ARCH, forced, detected.architectureId(), forcedId),
        modelType,
        architectures);
    }
    return detected;
  }

  private static void rejectIncompatibleSource(
    final Selection selected,
    final Source source,
    final Config.HfConfig config
  ) {
    String id = selected.architectureId();
    String modelType = config == null ? id : config.modelType();
    List<String> architectures = config == null ? List.of() : config.architectures();
    boolean hfFolderOnlyChat = ARCH_GEMMA3.equals(id) || ARCH_LLAMA.equals(id);
    if (ARCH_GEMMA4.equals(id) && source != Source.HF_SAFETENSORS) {
      throw unsupported(
        ("Architecture '%s' loads from a Hugging Face folder (config.json + *.safetensors QAT), "
          + "not from GGUF or ONNX. Vision and audio towers are skipped; text chat only.")
          .formatted(id),
        modelType,
        architectures);
    }
    if (hfFolderOnlyChat && source == Source.GGUF) {
      throw unsupported(
        ("Architecture '%s' loads from a Hugging Face folder (config.json + *.safetensors or "
          + "*.onnx), not from GGUF. GGUF chat is qwen3 and lfm2; embeddings are the BERT encoder.")
          .formatted(id),
        modelType,
        architectures);
    }
    if (ARCH_LFM2.equals(id) && source != Source.GGUF) {
      throw unsupported(
        "LFM2 loads from a .gguf file only, not from Hugging Face safetensors or ONNX.",
        modelType,
        architectures);
    }
    if (ARCH_BERT.equals(id) && source == Source.HF_SAFETENSORS) {
      throw unsupported(
        "BERT embeddings load from GGUF or ONNX, not from Hugging Face safetensors.",
        modelType,
        architectures);
    }
    if (ARCH_WHISPER.equals(id) && source != Source.HF_SAFETENSORS) {
      throw unsupported(
        "Whisper speech recognition loads from a Hugging Face folder (config.json + *.safetensors), "
          + "not from GGUF, ONNX, or CTranslate2 model.bin. Use openai/whisper-base (or tiny).",
        modelType,
        architectures);
    }
  }

  private static Verdict classifyToken(final String token) {
    if (token == null || token.isEmpty()) {
      return null;
    }
    if (isQwen3Text(token)) {
      return Verdict.ok(ARCH_QWEN3, Kind.CHAT);
    }
    if (token.startsWith("qwen3_5") || token.startsWith("qwen3_next") ||
      token.equals("qwen3next")) {
      return Verdict.rejected(
        "Qwen3.5 / Qwen3-Next / Fara use hybrid Gated DeltaNet + gated attention"
          + (token.contains("5") ? " (and often a vision tower)" : "")
          + ". This library implements Qwen3 text-only, not Qwen3.5.");
    }
    if (token.contains("qwen3_vl") || token.contains("qwen3vl") || token.startsWith("qwen3_moe")) {
      return Verdict.rejected(
        "This Qwen3 variant ('%s') is not implemented (vision or MoE)."
          .formatted(token));
    }
    if (token.startsWith("qwen3")) {
      return Verdict.rejected(
        ("Qwen3 variant '%s' is not implemented. Supported Qwen is exactly model_type qwen3 "
          + "(Qwen3ForCausalLM).").formatted(token));
    }
    if (isQwen2(token)) {
      return Verdict.rejected(
        "Qwen2 / Qwen2.5 is not Qwen3. This library implements Qwen3 (Q/K RMSNorm GQA), not the "
          + "Qwen2 decoder. Use a Qwen3 checkpoint.");
    }
    if (token.startsWith("qwen")) {
      return Verdict.rejected(
        "Qwen family '%s' is not implemented. Supported Qwen is qwen3 only.".formatted(token));
    }
    if (token.equals("gemma3") || token.equals("gemma3_text")) {
      return Verdict.ok(ARCH_GEMMA3, Kind.CHAT);
    }
    if (token.startsWith("gemma3")) {
      return Verdict.rejected(
        ("Gemma3 variant '%s' is not implemented. Supported Gemma is gemma3 / gemma3_text "
          + "(Gemma3ForCausalLM), not vision checkpoints.").formatted(token));
    }
    if (token.equals("gemma4") || token.equals("gemma4_text")) {
      return Verdict.ok(ARCH_GEMMA4, Kind.CHAT);
    }
    if (token.startsWith("gemma4")) {
      return Verdict.rejected(
        ("Gemma 4 variant '%s' is not implemented. Supported Gemma 4 is the text decoder "
          + "(model_type gemma4 / gemma4_text), loaded text-only from a QAT safetensors folder.")
          .formatted(token));
    }
    if (token.equals("gemma2") || token.startsWith("gemma2_")) {
      return Verdict.rejected(
        "Gemma 2 is not Gemma 3. This library implements Gemma3 text only.");
    }
    if (token.equals("gemma") || token.startsWith("gemma1") || token.startsWith("gemma_")) {
      return Verdict.rejected(
        "Gemma 1 is not implemented. This library implements Gemma3 text only.");
    }
    if (isLlama(token)) {
      return Verdict.ok(ARCH_LLAMA, Kind.CHAT);
    }
    if (token.startsWith("llama4")) {
      return Verdict.rejected(
        "Llama 4 is not implemented. Supported Llama is llama / llama2 / llama3.");
    }
    if (token.equals("lfm2") || token.startsWith("lfm2_5")) {
      return Verdict.ok(ARCH_LFM2, Kind.CHAT);
    }
    if (token.contains("lfm2moe") || token.endsWith("_moe") || token.contains("_moe_")) {
      return Verdict.rejected(
        "Mixture-of-experts checkpoints ('%s') are not implemented.".formatted(token));
    }
    if (isBertEncoder(token)) {
      return Verdict.ok(ARCH_BERT, Kind.EMBEDDING);
    }
    if (token.equals("whisper")) {
      return Verdict.ok(ARCH_WHISPER, Kind.SPEECH);
    }
    String known = knownUnsupportedReason(token);
    if (known != null) {
      return Verdict.rejected(known);
    }
    return null;
  }

  private static String knownUnsupportedReason(final String token) {
    if (token.startsWith("mistral") || token.startsWith("mixtral") ||
      token.startsWith("ministral")) {
      return "Mistral / Mixtral is not implemented. Closest supported chat graphs are llama "
        + "(HF/ONNX) and lfm2 (GGUF).";
    }
    if (token.startsWith("phi")) {
      return "Phi / Phi-3 is not implemented.";
    }
    if (token.startsWith("deepseek") || token.startsWith("minicpm") || token.startsWith("internlm")
      || token.startsWith("chatglm") || token.startsWith("glm") || token.startsWith("falcon")
      || token.startsWith("mpt") || token.startsWith("bloom") || token.startsWith("olmo")
      || token.startsWith("stablelm") || token.startsWith("cohere") ||
      token.startsWith("command")) {
      return "Architecture '%s' is not implemented.".formatted(token);
    }
    if (token.startsWith("gpt2") || token.startsWith("gptj") || token.startsWith("gpt_neox")
      || token.startsWith("gptneox") || token.equals("gpt_oss")) {
      return "GPT-2 / GPT-J / GPT-NeoX checkpoints are not implemented.";
    }
    if (token.startsWith("t5") || token.startsWith("bart") || token.startsWith("mamba")
      || token.startsWith("jamba") || token.startsWith("rwkv")) {
      return "Architecture '%s' is not a supported decoder-only dense Transformer.".formatted(
        token);
    }
    if (token.contains("llava") || token.contains("paligemma") || token.contains("internvl")
      || token.contains("idefics") || token.equals("fara")) {
      return multimodalReason(token);
    }
    if (isNonBertEmbeddingLookalike(token)) {
      return ("Embedding architecture '%s' is not implemented. Supported embeddings are the BERT "
        + "encoder (bert / roberta / xlm-roberta) from GGUF or ONNX.").formatted(token);
    }
    return null;
  }

  private static String unknownReason(final Config.HfConfig config) {
    String type = blank(config.modelType());
    String classes = config.architectures().isEmpty()
      ? "(none)"
      : String.join(", ", config.architectures());
    return "Cannot detect a supported architecture from model_type='%s' architectures=[%s]."
      .formatted(type, classes);
  }

  private static String multimodalReason(final String label) {
    return ("Vision-language / multimodal checkpoint '%s' is not supported. This library runs "
      + "text-only chat (qwen3, gemma3, gemma4, llama, lfm2) and BERT-encoder embeddings.")
      .formatted(blank(label));
  }

  private static boolean isQwen3Text(final String token) {
    return token.equals("qwen3") || token.equals("qwen3_text");
  }

  private static boolean isQwen2(final String token) {
    return token.equals("qwen2")
      || token.startsWith("qwen2_")
      || token.startsWith("qwen2vl");
  }

  private static boolean isLlama(final String token) {
    return token.equals("llama")
      || token.equals("llama2")
      || token.equals("llama3")
      || token.startsWith("llama3_");
  }

  private static boolean isBertEncoder(final String token) {
    if (isNonBertEmbeddingLookalike(token)) {
      return false;
    }
    return token.equals("bert")
      || token.startsWith("bert_")
      || token.endsWith("_bert")
      || token.contains("roberta");
  }

  private static boolean isNonBertEmbeddingLookalike(final String token) {
    return token.contains("deberta")
      || token.contains("distilbert")
      || token.contains("albert")
      || token.contains("electra")
      || token.contains("modernbert");
  }

  private static boolean isMultimodalClass(final String lower) {
    if (lower.contains("whisper")) {
      return false;
    }
    return lower.contains("conditionalgeneration")
      || lower.contains("vision")
      || lower.contains("imagetext")
      || lower.contains("image_text")
      || lower.contains("vlfor")
      || lower.contains("_vl")
      || lower.endsWith("vl");
  }

  private static String fromArchitectureClass(final String raw) {
    return normalize(HF_CLASS_SUFFIX.matcher(raw.strip()).replaceAll(""));
  }

  private static String normalize(final String raw) {
    if (raw == null || raw.isBlank()) {
      return "";
    }
    StringBuilder out = new StringBuilder(raw.length());
    for (int i = 0; i < raw.length(); i++) {
      char c = raw.charAt(i);
      if (c == '-' || c == '.' || c == ' ') {
        out.append('_');
      } else {
        out.append(Character.toLowerCase(c));
      }
    }
    return out.toString();
  }

  private static String normalizeForcedAlias(final String forced) {
    return switch (forced) {
      case ARCH_QWEN3 -> ARCH_QWEN3;
      case ARCH_GEMMA3, "gemma3_text" -> ARCH_GEMMA3;
      case ARCH_GEMMA4, "gemma4_text" -> ARCH_GEMMA4;
      case ARCH_LLAMA, "llama2", "llama3" -> ARCH_LLAMA;
      case ARCH_LFM2, "lfm2.5", "lfm2_5" -> ARCH_LFM2;
      case ARCH_BERT, "roberta", "xlm-roberta", "xlm_roberta" -> ARCH_BERT;
      case ARCH_WHISPER -> ARCH_WHISPER;
      default -> null;
    };
  }

  private static UnsupportedModelException unsupported(
    final String detail,
    final String modelType,
    final List<String> architectures
  ) {
    return new UnsupportedModelException(
      detail + System.lineSeparator() + System.lineSeparator() + CATALOG,
      modelType,
      architectures);
  }

  private static String blank(final String value) {
    return value == null || value.isBlank() ? "(missing)" : value;
  }

  /**
   * Load family: chat generator vs sentence-embedding encoder.
   *
   * @since 1.1.0
   */
  public enum Kind {
    /**
     * Causal chat / completion ({@code qwen3}, {@code gemma3}, {@code gemma4}, {@code llama}, {@code lfm2}).
     */
    CHAT,
    /** BERT-style encoder used with {@link LlmModel#embed}. */
    EMBEDDING,
    /**
     * Speech-to-text encoder-decoder used with {@link LlmModel#transcribe}.
     *
     * @since 1.3.0
     */
    SPEECH
  }

  /**
   * Weight container the checkpoint was loaded from.
   *
   * @since 1.1.0
   */
  public enum Source {
    /** Hugging Face folder with {@code *.safetensors}. */
    HF_SAFETENSORS,
    /** Hugging Face folder with {@code *.onnx} (Tier A, since 1.1.0). */
    ONNX,
    /** Single {@code .gguf} file. */
    GGUF
  }

  /**
   * Architecture chosen at load: backend id plus {@link Kind}.
   *
   * <p>Returned by {@link ModelSupport#require} / {@link ModelSupport#resolve} /
   * {@link ModelSupport#requireGguf}. Use {@link #isEmbedding()} to decide
   * {@link LlmModel#embed} vs {@link com.igormaznitsa.nanollvm.llm.LLM#builder}.
   * {@link #architectureId()} is the canonical key ({@code qwen3}, {@code gemma3}, {@code gemma4},
   * {@code llama}, {@code lfm2}, {@code bert}, {@code whisper}).
   *
   * @param architectureId canonical backend id; never {@code null}
   * @param kind           chat, embedding, or speech; never {@code null}
   * @since 1.1.0
   */
  public record Selection(String architectureId, Kind kind) {
    public Selection {
      requireNonNull(architectureId, "architectureId");
      requireNonNull(kind, "kind");
    }

    /**
     * {@code true} when this checkpoint is an embedding encoder, not a chat model.
     *
     * @since 1.1.0
     */
    public boolean isEmbedding() {
      return this.kind == Kind.EMBEDDING;
    }

    /**
     * {@code true} when this checkpoint is Whisper speech-to-text.
     *
     * @since 1.3.0
     */
    public boolean isSpeech() {
      return this.kind == Kind.SPEECH;
    }
  }

  private record Verdict(boolean supported, Selection selection, String detail) {
    static Verdict ok(final String architectureId, final Kind kind) {
      return new Verdict(true, new Selection(architectureId, kind), "");
    }

    static Verdict rejected(final String detail) {
      return new Verdict(false, null, detail);
    }
  }
}
