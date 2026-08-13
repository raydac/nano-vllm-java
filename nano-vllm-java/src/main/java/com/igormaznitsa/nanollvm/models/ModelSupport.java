package com.igormaznitsa.nanollvm.models;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_BERT;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_GEMMA3;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LFM2;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_LLAMA;
import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_QWEN3;
import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.PROP_ARCH;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;

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

  public static final String CATALOG = """
    Supported by this library:
      Chat from a Hugging Face folder (config.json + *.safetensors or *.onnx): qwen3, gemma3 / gemma3_text, llama
      Chat from a GGUF file: lfm2
      Embeddings from GGUF or ONNX: bert
    Not supported: Qwen2 / Qwen2.5, Qwen3.5 / Qwen3-Next / Fara, vision-language models, Gemma 1 / 2, \
    Mistral / Mixtral, Phi, MoE, GGUF Qwen / Llama / Gemma, Hugging Face BERT safetensors.""";

  private static final Pattern HF_CLASS_SUFFIX = Pattern.compile(
    "(ForCausalLM|ForConditionalGeneration|ForSequenceClassification|ForMaskedLM"
      + "|ForQuestionAnswering|ForTokenClassification|ForImageTextToText|ForVision2Seq|Model)$");

  private ModelSupport() {
  }

  /**
   * Resolves {@code config} and checks that {@code source} can carry that architecture
   * (e.g. Qwen3 chat is not loaded from GGUF).
   *
   * @return selected backend id and {@link Kind}
   * @throws com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException if the family is
   *                                                                        unknown or the source is incompatible
   */
  public static Selection require(final Config.HfConfig config, final Source source) {
    Selection selected = resolve(requireNonNull(config, "config"));
    rejectIncompatibleSource(selected, requireNonNull(source, "source"), config);
    return selected;
  }

  /**
   * Resolves a GGUF {@code general.architecture} string ({@code lfm2} chat or {@code bert}
   * embeddings).
   *
   * @param generalArchitecture GGUF metadata architecture; {@code null} treated as missing
   * @return selected backend id and {@link Kind}
   * @throws com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException if unrecognized or
   *                                                                        unsupported
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
   * {@code true} when {@code config} is a supported embedding encoder ({@code bert}).
   * Unknown or chat families return {@code false} (they do not throw).
   */
  public static boolean isEmbedding(final Config.HfConfig config) {
    Verdict verdict = inspect(requireNonNull(config, "config"));
    return verdict.supported() && verdict.selection().isEmbedding();
  }

  public static String chatMisuseMessage(final String architectureName) {
    return ("This checkpoint is a %s embedding encoder, not a chat model. Call LlmModel.embed(...) "
      + "instead of LLM.builder / generate.%n%n%s").formatted(blank(architectureName), CATALOG);
  }

  public static String embedMisuseMessage(final String architectureName) {
    return ("This checkpoint is a %s chat model, not an embedding encoder. Use LLM.builder(model) "
      + "for generate / chat.%n%n%s").formatted(blank(architectureName), CATALOG);
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
        return Verdict.rejected(multimodalReason(raw));
      }
      if (fromName != null && firstSupported == null) {
        firstSupported = fromName;
      }
    }
    return firstSupported;
  }

  private static Optional<String> structuralChatRejection(final Config.HfConfig config) {
    if (config.visionConfigPresent()) {
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
        "'%s' is not a valid -D%s value (use qwen3, gemma3, llama, lfm2, or bert)."
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
    boolean hfChat = ARCH_QWEN3.equals(id) || ARCH_GEMMA3.equals(id) || ARCH_LLAMA.equals(id);
    if (hfChat && source == Source.GGUF) {
      throw unsupported(
        ("Architecture '%s' loads from a Hugging Face folder (config.json + *.safetensors or "
          + "*.onnx), not from GGUF. GGUF is supported for lfm2 (chat) and bert (embeddings).")
          .formatted(id),
        config == null ? id : config.modelType(),
        config == null ? List.of() : config.architectures());
    }
    if (ARCH_LFM2.equals(id) && source != Source.GGUF) {
      throw unsupported(
        "LFM2 loads from a .gguf file only, not from Hugging Face safetensors or ONNX.",
        config == null ? id : config.modelType(),
        config == null ? List.of() : config.architectures());
    }
    if (ARCH_BERT.equals(id) && source == Source.HF_SAFETENSORS) {
      throw unsupported(
        "BERT embeddings load from GGUF or ONNX, not from Hugging Face safetensors.",
        config == null ? id : config.modelType(),
        config == null ? List.of() : config.architectures());
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
    if (isBert(token)) {
      return Verdict.ok(ARCH_BERT, Kind.EMBEDDING);
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
    if (token.contains("roberta") || token.contains("deberta") || token.contains("distilbert")
      || token.contains("albert") || token.contains("electra") || token.contains("modernbert")) {
      return ("Embedding architecture '%s' is not implemented. Supported embeddings are bert "
        + "(GGUF or ONNX).").formatted(token);
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
      + "text-only chat (qwen3, gemma3, llama, lfm2) and bert embeddings.").formatted(blank(label));
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

  private static boolean isBert(final String token) {
    if (token.contains("roberta") || token.contains("deberta") || token.contains("distilbert")
      || token.contains("albert") || token.contains("electra") || token.contains("modernbert")) {
      return false;
    }
    return token.equals("bert") || token.startsWith("bert_") || token.endsWith("_bert");
  }

  private static boolean isMultimodalClass(final String lower) {
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
      case ARCH_LLAMA, "llama2", "llama3" -> ARCH_LLAMA;
      case ARCH_LFM2, "lfm2.5", "lfm2_5" -> ARCH_LFM2;
      case ARCH_BERT -> ARCH_BERT;
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
     * Causal chat / completion ({@code qwen3}, {@code gemma3}, {@code llama}, {@code lfm2}).
     */
    CHAT,
    /** BERT-style encoder used with {@link LlmModel#embed}. */
    EMBEDDING
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
   * {@link #architectureId()} is the canonical key ({@code qwen3}, {@code gemma3}, {@code llama},
   * {@code lfm2}, {@code bert}).
   *
   * @param architectureId canonical backend id; never {@code null}
   * @param kind           chat vs embedding; never {@code null}
   * @since 1.1.0
   */
  public record Selection(String architectureId, Kind kind) {
    public Selection {
      requireNonNull(architectureId, "architectureId");
      requireNonNull(kind, "kind");
    }

    /**
     * {@code true} when this checkpoint is an embedding encoder, not a chat model.
     */
    public boolean isEmbedding() {
      return this.kind == Kind.EMBEDDING;
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
