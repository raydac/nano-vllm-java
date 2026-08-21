package com.igormaznitsa.nanollvm.tokenizer;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Text ↔ token-id bridge for this engine: encode prompts, decode completions, and format chat turns.
 *
 * <h2>What it loads</h2>
 * Hugging Face {@code tokenizer.json} (+ optional {@code tokenizer_config.json} /
 * {@code generation_config.json}) via {@link #fromPretrained(Path)}, SentencePiece
 * {@code tokenizer.model} when the JSON sidecar is absent, or GGUF
 * {@code tokenizer.ggml.*} metadata via {@link #fromGguf(GgufTokenizerSource)}.
 * Applications normally get an instance from {@link com.igormaznitsa.nanollvm.models.LlmModel#tokenizer()}.
 *
 * <h2>Encode styles</h2>
 * <ul>
 *   <li>GPT-2 byte-level BPE (common with ChatML exports).</li>
 *   <li>SentencePiece-style BPE with {@code ▁} word boundaries.</li>
 *   <li>BERT WordPiece (HF {@code ##} or GGUF metaspace).</li>
 *   <li>Unigram SentencePiece, including precompiled charsmap normalization.</li>
 *   <li>WordLevel and character models.</li>
 * </ul>
 * Chat prompt layout is {@link ChatFormat} detected from template/vocab markers, not product names.
 *
 * <p>Immutable after construction. Safe to share across threads and across many
 * {@link com.igormaznitsa.nanollvm.llm.LLM} instances.
 *
 * @see #encode(String)
 * @see #decode(List)
 * @see #applyChatTemplate(List, boolean, boolean)
 */
public final class Tokenizer {

  private final TokenVocab vocab;
  private final TokenCodec codec;
  private final TextNormalizer normalizer;
  private final ChatFormat chatFormat;
  private final int eosTokenId;
  private final List<Integer> stopTokenIds;
  private final int padTokenId;
  private final boolean inviteThinking;

  Tokenizer(
    final TokenVocab vocab,
    final TokenCodec codec,
    final TextNormalizer normalizer,
    final ChatFormat chatFormat,
    final int eosTokenId,
    final List<Integer> stopTokenIds,
    final int padTokenId,
    final boolean inviteThinking
  ) {
    this.vocab = requireNonNull(vocab, "vocab");
    this.codec = requireNonNull(codec, "codec");
    this.normalizer = requireNonNull(normalizer, "normalizer");
    this.chatFormat = requireNonNull(chatFormat, "chatFormat");
    this.eosTokenId = eosTokenId;
    this.stopTokenIds = List.copyOf(stopTokenIds);
    this.padTokenId = padTokenId;
    this.inviteThinking = inviteThinking;
  }

  /**
   * Loads a tokenizer from an HF model directory ({@code tokenizer.json}, else
   * {@code tokenizer.model}, else a tiny {@linkplain #fromJsonDocuments bare} fallback from
   * {@code config.json}).
   *
   * @param modelDir directory containing tokenizer files; non-{@code null}
   * @return immutable tokenizer
   * @throws com.igormaznitsa.nanollvm.exceptions.ModelLoadException if I/O or parse fails
   */
  public static Tokenizer fromPretrained(final Path modelDir) {
    return TokenizerLoader.fromPretrained(requireNonNull(modelDir, "modelDir"));
  }

  /**
   * Builds a tokenizer from in-memory HF JSON sidecars (no filesystem access).
   *
   * @since 1.1.0
   */
  public static Tokenizer fromJsonDocuments(
    final String tokenizerJson,
    final String tokenizerConfigJson,
    final String generationConfigJson,
    final String modelConfigJson
  ) {
    return TokenizerLoader.fromJsonDocuments(
      tokenizerJson, tokenizerConfigJson, generationConfigJson, modelConfigJson);
  }

  /**
   * Builds a tokenizer from a SentencePiece {@code tokenizer.model} protobuf.
   *
   * @param modelBytes           SentencePiece ModelProto bytes; non-{@code null}
   * @param tokenizerConfigJson  optional {@code tokenizer_config.json}
   * @param generationConfigJson optional {@code generation_config.json}
   * @return immutable tokenizer
   * @since 1.2.0
   */
  public static Tokenizer fromSentencePiece(
    final byte[] modelBytes,
    final String tokenizerConfigJson,
    final String generationConfigJson
  ) {
    return TokenizerLoader.fromSentencePiece(
      modelBytes, tokenizerConfigJson, generationConfigJson);
  }

  /**
   * Builds a tokenizer from GGUF {@code tokenizer.ggml.*} metadata.
   * Prefer {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} for application load paths.
   *
   * @param source GGUF metadata reader; non-{@code null}
   * @return immutable tokenizer ({@link #invitesThinking()} follows vocab {@code <think>}/{@code </think>} markers)
   * @throws com.igormaznitsa.nanollvm.exceptions.ModelLoadException if {@code tokenizer.ggml.tokens} is missing/empty
   * @throws NullPointerException if {@code source} is {@code null}
   */
  public static Tokenizer fromGguf(final GgufTokenizerSource source) {
    return TokenizerLoader.fromGguf(source);
  }

  /**
   * Decode UTF-8 but drop a trailing incomplete multi-byte sequence.
   * Needed for streamed token decode so Cyrillic (e.g. щ = {@code D1 89}) does not show as {@code �}.
   *
   * @param bytes raw UTF-8 bytes accumulated so far; {@code null}/empty → {@code ""}
   * @return decoded prefix of complete code points only
   */
  public static String decodeUtf8Complete(final byte[] bytes) {
    return Utf8Complete.decode(bytes);
  }

  /**
   * Primary end-of-sequence token id for this vocabulary.
   *
   * @return EOS id (also the first entry of {@link #stopTokenIds()} unless config adds more)
   */
  public int eosTokenId() {
    return this.eosTokenId;
  }

  /**
   * Token ids that finish a generation sequence (EOS plus model-specific end markers).
   *
   * @return immutable list; never empty (at least {@link #eosTokenId()})
   */
  public List<Integer> stopTokenIds() {
    return this.stopTokenIds;
  }

  /**
   * Padding token id (may equal {@link #eosTokenId()} when the model has no dedicated pad).
   *
   * @return pad id
   */
  public int padTokenId() {
    return this.padTokenId;
  }

  /**
   * Vocabulary id for {@code token}, or empty when absent.
   *
   * @since 1.1.0
   */
  public Optional<Integer> tokenId(final String token) {
    return Optional.ofNullable(this.vocab.id(requireNonNull(token, "token")));
  }

  /**
   * Detected chat prompt layout for {@link #applyChatTemplate(List, boolean, boolean)}.
   *
   * @since 1.1.0
   */
  public ChatFormat chatFormat() {
    return this.chatFormat;
  }

  /**
   * {@code true} when turns use {@code <start_of_turn>} / {@code <end_of_turn>} markers.
   *
   * @since 1.1.0
   */
  public boolean isTurnBasedChat() {
    return this.chatFormat == ChatFormat.TURN_BASED;
  }

  /**
   * Whether this chat path invites a {@code <think>} scratchpad.
   * True only when the vocab contains both {@code <think>} and {@code </think>}.
   *
   * @return {@code true} when {@link #applyChatTemplate(List, boolean, boolean)} may emit an empty
   * think block when thinking is disabled
   */
  public boolean invitesThinking() {
    return this.inviteThinking;
  }

  /**
   * Whether both scratchpad markers exist as whole vocabulary tokens (the ChatML skip-seed gate).
   *
   * @param thinkOpen  start marker; must not be {@code null}
   * @param thinkClose end marker; must not be {@code null}
   * @return {@code true} when both strings are non-empty and present in vocab
   * @since 1.1.0
   */
  public boolean invitesThinking(final String thinkOpen, final String thinkClose) {
    requireNonNull(thinkOpen, "thinkOpen");
    requireNonNull(thinkClose, "thinkClose");
    return !thinkOpen.isEmpty()
      && !thinkClose.isEmpty()
      && this.vocab.contains(thinkOpen)
      && this.vocab.contains(thinkClose);
  }

  /**
   * Whether chat decode should omit special/control tokens (turn-based markup models).
   *
   * @since 1.1.0
   */
  public boolean skipSpecialTokensOnChatDecode() {
    return this.chatFormat == ChatFormat.TURN_BASED;
  }

  /**
   * Encodes plain text to vocabulary ids (detected BPE, WordPiece, Unigram, WordLevel, or Char).
   *
   * @param text input text; non-{@code null} (may be empty)
   * @return immutable token-id list
   */
  public List<Integer> encode(final String text) {
    return this.codec.encode(this.normalizer.normalize(requireNonNull(text, "text")));
  }

  /**
   * Decodes token ids to text, keeping special tokens in the string.
   *
   * @param tokenIds vocabulary ids; non-{@code null}
   * @return decoded text
   * @see #decode(List, boolean)
   */
  public String decode(final List<Integer> tokenIds) {
    return this.decode(tokenIds, false);
  }

  /**
   * Decodes token ids to text.
   *
   * @param tokenIds          vocabulary ids; non-{@code null}
   * @param skipSpecialTokens when {@code true}, omit ids in the special/skip set (e.g. think tags)
   * @return decoded UTF-8 text (Metaspace {@code ▁} restored to spaces)
   */
  public String decode(final List<Integer> tokenIds, final boolean skipSpecialTokens) {
    requireNonNull(tokenIds, "tokenIds");
    List<String> pieces = new ArrayList<>(tokenIds.size());
    for (int id : tokenIds) {
      if (skipSpecialTokens && this.vocab.skip(id)) {
        continue;
      }
      String tok = this.vocab.token(id);
      if (tok != null) {
        pieces.add(tok);
      }
    }
    return this.codec.decode(pieces);
  }

  /**
   * Formats chat messages with the model’s template and appends an assistant generation prompt.
   *
   * @param messages            role/content maps ({@code "role"}, {@code "content"}); non-{@code null}
   * @param addGenerationPrompt when {@code true}, append the assistant turn opener
   * @return prompt string ready to {@link #encode(String)}
   */
  public String applyChatTemplate(final List<Map<String, String>> messages,
                                  final boolean addGenerationPrompt) {
    return this.applyChatTemplate(messages, addGenerationPrompt, true);
  }

  /**
   * Formats chat messages for the model ({@link ChatFormat}: turn-based, ChatML, or plain).
   *
   * @param messages            role/content maps; non-{@code null}
   * @param addGenerationPrompt when {@code true}, append the assistant / model turn opener
   * @param enableThinking      ChatML only: when {@code false} and {@link #invitesThinking()}, insert an
   *                            empty {@code <think>}…{@code </think>} so the model skips the scratchpad
   * @return prompt string ready to {@link #encode(String)}
   */
  public String applyChatTemplate(final List<Map<String, String>> messages,
                                  final boolean addGenerationPrompt,
                                  final boolean enableThinking) {
    return this.applyChatTemplate(messages, addGenerationPrompt, enableThinking, "<think>",
      "</think>");
  }

  /**
   * {@link #applyChatTemplate(List, boolean, boolean)} with custom scratchpad markers for the ChatML
   * skip-seed (emitted only when thinking is disabled and both strings are in vocab).
   *
   * @param thinkOpen  start marker; must not be {@code null}
   * @param thinkClose end marker; must not be {@code null}
   * @return prompt string ready to {@link #encode(String)}
   * @since 1.1.0
   */
  public String applyChatTemplate(final List<Map<String, String>> messages,
                                  final boolean addGenerationPrompt,
                                  final boolean enableThinking,
                                  final String thinkOpen,
                                  final String thinkClose) {
    requireNonNull(thinkOpen, "thinkOpen");
    requireNonNull(thinkClose, "thinkClose");
    if (this.chatFormat == ChatFormat.TURN_BASED) {
      return this.applyTurnBasedChat(messages, addGenerationPrompt, enableThinking);
    }
    if (this.chatFormat == ChatFormat.CHATML) {
      StringBuilder sb = new StringBuilder();
      for (Map<String, String> msg : messages) {
        String role = msg.getOrDefault("role", "user");
        String content = msg.getOrDefault("content", "");
        sb.append("<|im_start|>").append(role).append('\n').append(content).append("<|im_end|>\n");
      }
      if (addGenerationPrompt) {
        sb.append("<|im_start|>assistant\n");
        if (!enableThinking && this.invitesThinking(thinkOpen, thinkClose)) {
          sb.append(thinkOpen).append("\n\n").append(thinkClose).append("\n\n");
        }
      }
      return sb.toString();
    }
    StringBuilder sb = new StringBuilder();
    for (Map<String, String> msg : messages) {
      sb.append(msg.getOrDefault("role", "user")).append(": ")
        .append(msg.getOrDefault("content", "")).append('\n');
    }
    if (addGenerationPrompt) {
      sb.append("assistant: ");
    }
    return sb.toString();
  }

  private String applyTurnBasedChat(final List<Map<String, String>> messages,
                                    final boolean addGenerationPrompt,
                                    final boolean enableThinking) {
    if (this.vocab.contains("<|turn>")) {
      return this.applyAngleTurnChat(messages, addGenerationPrompt, enableThinking);
    }
    String system = null;
    List<Map<String, String>> turns = new ArrayList<>();
    for (Map<String, String> msg : messages) {
      String role = msg.getOrDefault("role", "user");
      if ("system".equals(role)) {
        system = msg.getOrDefault("content", "");
        continue;
      }
      turns.add(msg);
    }

    StringBuilder sb = new StringBuilder();
    if (this.vocab.contains("<bos>")) {
      sb.append("<bos>");
    }
    boolean firstUser = true;
    for (Map<String, String> msg : turns) {
      String role = msg.getOrDefault("role", "user");
      if ("assistant".equals(role)) {
        role = "model";
      }
      String content = msg.getOrDefault("content", "");
      if ("user".equals(role)) {
        content = ChatPrompts.foldSystemIntoFirstUser(system, content, firstUser);
        firstUser = false;
      }
      sb.append("<start_of_turn>").append(role).append('\n')
        .append(content).append("<end_of_turn>\n");
    }
    if (addGenerationPrompt) {
      sb.append("<start_of_turn>model\n");
    }
    return sb.toString();
  }

  private String applyAngleTurnChat(final List<Map<String, String>> messages,
                                    final boolean addGenerationPrompt,
                                    final boolean enableThinking) {
    String system = null;
    List<Map<String, String>> turns = new ArrayList<>();
    for (Map<String, String> msg : messages) {
      String role = msg.getOrDefault("role", "user");
      if ("system".equals(role) || "developer".equals(role)) {
        system = msg.getOrDefault("content", "");
        continue;
      }
      turns.add(msg);
    }

    StringBuilder sb = new StringBuilder();
    if (this.vocab.contains("<bos>")) {
      sb.append("<bos>");
    }
    if (enableThinking || (system != null && !system.isBlank())) {
      sb.append("<|turn>system\n");
      if (enableThinking) {
        sb.append("<|think|>\n");
      }
      if (system != null && !system.isBlank()) {
        sb.append(system.strip());
      }
      sb.append("<turn|>\n");
    }
    for (Map<String, String> msg : turns) {
      String role = msg.getOrDefault("role", "user");
      if ("assistant".equals(role)) {
        role = "model";
      }
      sb.append("<|turn>").append(role).append('\n')
        .append(msg.getOrDefault("content", "").strip())
        .append("<turn|>\n");
    }
    if (addGenerationPrompt) {
      sb.append("<|turn>model\n");
    }
    return sb.toString();
  }

  /**
   * Chat prompt layout detected from template / vocabulary markers.
   *
   * @since 1.1.0
   */
  public enum ChatFormat {
    /**
     * {@code <|im_start|>} / {@code <|im_end|>} turns.
     */
    CHATML,
    /**
     * {@code <start_of_turn>} / {@code <end_of_turn>} or Gemma 4 {@code <|turn>} / {@code <turn|>}
     * turns. Gemma 3 folds system into the first user turn; Gemma 4 emits a native system turn.
     */
    TURN_BASED,
    /**
     * Plain {@code role: text} fallback when no chat markers are present.
     */
    PLAIN
  }
}
