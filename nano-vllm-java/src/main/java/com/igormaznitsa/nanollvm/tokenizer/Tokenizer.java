package com.igormaznitsa.nanollvm.tokenizer;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Text ↔ token-id bridge for this engine: encode prompts, decode completions, and format chat turns.
 *
 * <h2>What it loads</h2>
 * Hugging Face {@code tokenizer.json} (+ optional {@code tokenizer_config.json} /
 * {@code generation_config.json}) via {@link #fromPretrained(Path)}, or GGUF
 * {@code tokenizer.ggml.*} metadata via {@link #fromGguf(GgufTokenizerSource)}.
 * Applications normally get an instance from {@link com.igormaznitsa.nanollvm.models.LlmModel#tokenizer()}.
 *
 * <h2>BPE / WordPiece styles</h2>
 * <ul>
 *   <li>{@link Style#GPT2_BYTE_BPE} — GPT-2 byte-level BPE (common with ChatML exports): UTF-8 bytes
 *       map through a printable-char encoder, then merge ranks from {@code merges}.</li>
 *   <li>{@link Style#METASPACE_BPE} — SentencePiece-style with {@code ▁} word boundaries.</li>
 *   <li>{@link Style#WORDPIECE} — BERT WordPiece (embedding models).</li>
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

  // GPT-2 pretokenize regex: contractions, letters, numbers, punctuation, whitespace runs
  private static final Pattern GPT2_PATTERN = Pattern.compile(
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  );
  // SentencePiece / Metaspace word-boundary marker (U+2581)
  private static final String META_SPACE = "▁";
  // GPT-2: each raw byte 0..255 → one Unicode code unit used as a BPE alphabet symbol
  private static final String[] BYTE_ENCODER = new String[256];
  // Inverse of BYTE_ENCODER for decode
  private static final Map<String, Integer> BYTE_DECODER = new HashMap<>();

  static {
    // Build the GPT-2 printable-byte remapping (bytes that are not already printable get
    // assigned code points starting at 256 so every byte has a unique BPE symbol).
    List<Integer> bs = new ArrayList<>();
    for (int i = '!'; i <= '~'; i++) {
      bs.add(i);
    }
    for (int i = '¡'; i <= '¬'; i++) {
      bs.add(i);
    }
    for (int i = '®'; i <= 'ÿ'; i++) {
      bs.add(i);
    }
    List<Integer> cs = new ArrayList<>(bs);
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n++;
      }
    }
    for (int i = 0; i < bs.size(); i++) {
      String ch = new String(Character.toChars(cs.get(i)));
      BYTE_ENCODER[bs.get(i)] = ch;
      BYTE_DECODER.put(ch, bs.get(i));
    }
  }

  /**
   * Token string → vocabulary id (includes added / special tokens).
   */
  private final Map<String, Integer> vocab;
  /**
   * Vocabulary id → token string (inverse of {@link #vocab}).
   */
  private final Map<Integer, String> idToToken;
  /**
   * BPE merge key {@code "left right"} → rank (lower rank merges first).
   */
  private final Map<String, Integer> merges;
  /**
   * Added/special token strings sorted longest-first for greedy matching during encode/decode.
   */
  private final List<String> addedTokensByLength;
  /**
   * Vocab ids omitted from {@link #decode(List, boolean)} when {@code skipSpecialTokens} is true.
   */
  private final Set<Integer> skipTokenIds;
  /**
   * Primary end-of-sequence id used by the engine when stop lists are empty.
   */
  private final int eosTokenId;
  /**
   * All stop ids (EOS plus extras from config / known marker strings).
   */
  private final List<Integer> stopTokenIds;
  /**
   * Padding id (often same as EOS when the model has no dedicated pad).
   */
  private final int padTokenId;
  /**
   * Raw HF / GGUF chat template string when present; used to detect {@link ChatFormat}.
   */
  private final String chatTemplate;
  /**
   * When a BPE piece is missing from vocab, fall back to byte / {@code <0x..>} pieces.
   */
  private final boolean byteFallback;
  /**
   * Active encode/decode algorithm.
   */
  private final Style style;
  /**
   * Chat prompt layout detected from template / vocab markers.
   */
  private final ChatFormat chatFormat;
  /**
   * Metaspace only: prepend {@code ▁} before a chunk (HF Metaspace pretok); replace-only normalizers stay false.
   */
  private final boolean prependMetaSpace;
  /**
   * WordPiece / BERT: lowercase before encode (uncased models).
   */
  private final boolean lowercase;
  /**
   * When {@code true}, ChatML generation prompt may open an empty {@code <think>} block unless disabled.
   */
  private final boolean inviteThinking;

  private Tokenizer(
    final Map<String, Integer> vocab,
    final Map<String, Integer> merges,
    final Set<String> addedTokenTexts,
    final Set<String> skipTokenTexts,
    final int eosTokenId,
    final List<Integer> stopTokenIds,
    final int padTokenId,
    final String chatTemplate,
    final boolean byteFallback,
    final Style style,
    final ChatFormat chatFormat,
    final boolean prependMetaSpace,
    final boolean lowercase,
    final boolean inviteThinking
  ) {
    this.vocab = vocab;
    this.idToToken = new HashMap<>();
    for (var e : vocab.entrySet()) {
      this.idToToken.put(e.getValue(), e.getKey());
    }
    this.merges = merges;
    this.addedTokensByLength = addedTokenTexts.stream()
      .sorted(Comparator.comparingInt(String::length).reversed())
      .toList();
    this.skipTokenIds = new HashSet<>();
    for (String text : skipTokenTexts) {
      Integer id = vocab.get(text);
      if (id != null) {
        this.skipTokenIds.add(id);
      }
    }
    this.eosTokenId = eosTokenId;
    this.stopTokenIds = List.copyOf(stopTokenIds);
    this.padTokenId = padTokenId;
    this.chatTemplate = chatTemplate;
    this.byteFallback = byteFallback;
    this.style = style;
    this.chatFormat = requireNonNull(chatFormat, "chatFormat");
    this.prependMetaSpace = prependMetaSpace;
    this.lowercase = lowercase;
    this.inviteThinking = inviteThinking;
  }

  /**
   * Loads a tokenizer from an HF model directory ({@code tokenizer.json} required for a full
   * vocab; otherwise a tiny {@linkplain #bare(String) bare} fallback is built from {@code config.json}).
   *
   * @param modelDir directory containing tokenizer files; non-{@code null}
   * @return immutable tokenizer
   * @throws ModelLoadException if JSON I/O fails
   */
  public static Tokenizer fromPretrained(final Path modelDir) {
    try {
      Path tokenizerJson = modelDir.resolve("tokenizer.json");
      if (!Files.isRegularFile(tokenizerJson)) {
        return bare(Files.isRegularFile(modelDir.resolve(CONFIG_JSON))
          ? Files.readString(modelDir.resolve(CONFIG_JSON))
          : null);
      }
      String tokenizerConfig = null;
      Path config = modelDir.resolve("tokenizer_config.json");
      if (Files.isRegularFile(config)) {
        tokenizerConfig = Files.readString(config);
      }
      String generationConfig = null;
      Path genConfig = modelDir.resolve("generation_config.json");
      if (Files.isRegularFile(genConfig)) {
        generationConfig = Files.readString(genConfig);
      }
      String modelConfig = null;
      Path cfg = modelDir.resolve(CONFIG_JSON);
      if (Files.isRegularFile(cfg)) {
        modelConfig = Files.readString(cfg);
      }
      return fromJsonDocuments(
        Files.readString(tokenizerJson),
        tokenizerConfig,
        generationConfig,
        modelConfig);
    } catch (IOException e) {
      throw new ModelLoadException("failed to load tokenizer from " + modelDir, e);
    }
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
    if (tokenizerJson == null || tokenizerJson.isBlank()) {
      return bare(modelConfigJson);
    }
    try {
      Map<String, Object> root = Json.parseObject(tokenizerJson);
      Map<String, Object> model = Json.asObject(root.get("model"));
      Map<String, Object> vocabObj = Json.asObject(model.get("vocab"));
      Map<String, Integer> vocab = new LinkedHashMap<>();
      for (var e : vocabObj.entrySet()) {
        vocab.put(e.getKey(), Json.asInt(e.getValue(), 0));
      }

      Set<String> addedTexts = new HashSet<>();
      Set<String> specialTexts = new HashSet<>();
      List<Object> added = Json.asArray(root.get("added_tokens"));
      if (added != null) {
        for (Object item : added) {
          Map<String, Object> tok = Json.asObject(item);
          if (tok == null) {
            continue;
          }
          String content = Json.asString(tok.get("content"));
          int id = Json.asInt(tok.get("id"), -1);
          if (content != null && id >= 0) {
            vocab.put(content, id);
            addedTexts.add(content);
            if (Json.asBoolean(tok.get("special"), false)
              || content.equals("<think>")
              || content.equals("</think>")
              || content.equals("<bos>")
              || content.equals("<eos>")
              || content.equals("<start_of_turn>")
              || content.equals("<end_of_turn>")) {
              specialTexts.add(content);
            }
          }
        }
      }

      Map<String, Integer> merges = new HashMap<>();
      List<Object> mergeList = Json.asArray(model.get("merges"));
      if (mergeList != null) {
        for (int i = 0; i < mergeList.size(); i++) {
          Object m = mergeList.get(i);
          String merge;
          if (m instanceof String s) {
            merge = s;
          } else if (m instanceof List<?> pair) {
            merge = pair.get(0) + " " + pair.get(1);
          } else {
            continue;
          }
          merges.put(merge, i);
        }
      }

      String chatTemplate = null;
      String eosToken = null;
      String padToken = null;
      if (tokenizerConfigJson != null && !tokenizerConfigJson.isBlank()) {
        Map<String, Object> tc = Json.parseObject(tokenizerConfigJson);
        chatTemplate = Json.asString(tc.get("chat_template"));
        eosToken = tokenString(tc.get("eos_token"));
        padToken = tokenString(tc.get("pad_token"));
      }

      ChatFormat chatFormat = detectChatFormat(vocab, chatTemplate, root);
      boolean turnBased = chatFormat == ChatFormat.TURN_BASED;
      Style style =
          turnBased || usesMetaspace(root, vocab) ? Style.METASPACE_BPE : Style.GPT2_BYTE_BPE;
      boolean prependMetaSpace = shouldPrependMetaSpace(root);

      int eos = resolveEos(vocab, eosToken);
      int pad = resolvePad(vocab, padToken, eos);
      List<Integer> stopIds = new ArrayList<>();
      stopIds.add(eos);
      addStopIfPresent(vocab, stopIds, "<|endoftext|>");
      addStopIfPresent(vocab, stopIds, "<eos>");
      addStopIfPresent(vocab, stopIds, "<end_of_turn>");

      if (generationConfigJson != null && !generationConfigJson.isBlank()) {
        Map<String, Object> gc = Json.parseObject(generationConfigJson);
        Object eosField = gc.get("eos_token_id");
        if (eosField instanceof List<?> list) {
          for (Object o : list) {
            int id = Json.asInt(o, -1);
            if (id >= 0 && !stopIds.contains(id)) {
              stopIds.add(id);
            }
          }
        } else if (eosField != null) {
          int id = Json.asInt(eosField, -1);
          if (id >= 0 && !stopIds.contains(id)) {
            stopIds.add(id);
          }
        }
      }

      boolean byteFallback =
        Json.asBoolean(model.get("byte_fallback"), style == Style.GPT2_BYTE_BPE);
      addStopIfPresent(vocab, stopIds, "<|im_end|>");
      boolean inviteThinking = vocab.containsKey("<think>") && vocab.containsKey("</think>");
      return new Tokenizer(
        vocab, merges, addedTexts, specialTexts, eos, stopIds, pad,
          chatTemplate, byteFallback, style, chatFormat, prependMetaSpace, false, inviteThinking
      );
    } catch (RuntimeException e) {
      throw new ModelLoadException("failed to load tokenizer from JSON documents", e);
    }
  }

  /**
   * Builds a tokenizer from GGUF {@code tokenizer.ggml.*} metadata.
   * Prefer {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} for application load paths.
   *
   * @param source GGUF metadata reader; non-{@code null}
   * @return immutable tokenizer ({@link #invitesThinking()} follows vocab {@code <think>}/{@code </think>} markers)
   * @throws ModelLoadException   if {@code tokenizer.ggml.tokens} is missing/empty
   * @throws NullPointerException if {@code source} is {@code null}
   */
  public static Tokenizer fromGguf(final GgufTokenizerSource source) {
    requireNonNull(source, "source");
    List<String> tokens = source.metaStringArray("tokenizer.ggml.tokens");
    if (tokens.isEmpty()) {
      throw new ModelLoadException("GGUF missing tokenizer.ggml.tokens");
    }
    Map<String, Integer> vocab = new LinkedHashMap<>(tokens.size() * 2);
    Set<String> addedTexts = new HashSet<>();
    Set<String> specialTexts = new HashSet<>();
    for (int i = 0; i < tokens.size(); i++) {
      String tok = tokens.get(i);
      vocab.put(tok, i);
      if (tok.startsWith("<") && tok.endsWith(">")) {
        addedTexts.add(tok);
        specialTexts.add(tok);
      }
    }

    Map<String, Integer> merges = new HashMap<>();
    List<String> mergeList = source.metaStringArray("tokenizer.ggml.merges");
    for (int i = 0; i < mergeList.size(); i++) {
      merges.put(mergeList.get(i), i);
    }

    String ggmlModel = source.metaString("tokenizer.ggml.model", "gpt2").toLowerCase(Locale.ROOT);
    boolean bertWordPiece = ggmlModel.contains("bert");
    for (String marker : List.of("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")) {
      if (vocab.containsKey(marker)) {
        addedTexts.add(marker);
        specialTexts.add(marker);
      }
    }
    // tokenizer.ggml.model values "llama" / "spm" mark SentencePiece-style metaspace alphabets
    boolean metaspace =
      tokens.stream().limit(4000).filter(t -> t.startsWith(META_SPACE)).count() > 200
        || ggmlModel.contains("llama")
        || ggmlModel.contains("spm");
    Style style = bertWordPiece
      ? Style.WORDPIECE
      : (metaspace ? Style.METASPACE_BPE : Style.GPT2_BYTE_BPE);
    // GGUF metaspace alphabets usually do not prepend ▁ the way HF Metaspace pretok does
    boolean prependMetaSpace = bertWordPiece;
    boolean inviteThinking = vocab.containsKey("<think>") && vocab.containsKey("</think>");

    int eos = source.metaInt("tokenizer.ggml.eos_token_id", -1);
    if (eos < 0) {
      eos = resolveEos(vocab, null);
    }
    int pad = source.metaInt("tokenizer.ggml.padding_token_id", -1);
    if (pad < 0) {
      pad = resolvePad(vocab, null, eos);
    }
    List<Integer> stopIds = new ArrayList<>();
    stopIds.add(eos);
    int eot = source.metaInt("tokenizer.ggml.eot_token_id", -1);
    if (eot >= 0 && !stopIds.contains(eot)) {
      stopIds.add(eot);
    }
    addStopIfPresent(vocab, stopIds, "<|im_end|>");
    addStopIfPresent(vocab, stopIds, "<|endoftext|>");

    String chatTemplate = source.metaString("tokenizer.chat_template", null);
    if (chatTemplate == null) {
      chatTemplate = source.metaString("tokenizer.ggml.chat_template", null);
    }
    if (chatTemplate == null && vocab.containsKey("<|im_start|>")) {
      chatTemplate = "<|im_start|>";
    }
    ChatFormat chatFormat = detectChatFormat(vocab, chatTemplate, null);
    boolean byteFallback = style == Style.GPT2_BYTE_BPE;

    return new Tokenizer(
      vocab, merges, addedTexts, specialTexts, eos, stopIds, pad,
        chatTemplate, byteFallback, style, chatFormat, prependMetaSpace, bertWordPiece,
        inviteThinking
    );
  }

  // HF eos_token / pad_token may be a plain string or { "content": "..." }
  private static String tokenString(final Object value) {
    if (value instanceof String s) {
      return s;
    }
    if (value instanceof Map<?, ?> map) {
      Object content = map.get("content");
      return content instanceof String s ? s : null;
    }
    return null;
  }

  // Detect chat layout from template / vocab markers (not product names)
  private static ChatFormat detectChatFormat(
    final Map<String, Integer> vocab,
    final String chatTemplate,
    final Map<String, Object> tokenizerRoot
  ) {
    if (chatTemplate != null && chatTemplate.contains("start_of_turn")) {
      return ChatFormat.TURN_BASED;
    }
    if (vocab.containsKey("<start_of_turn>")) {
      return ChatFormat.TURN_BASED;
    }
    if (usesChatMlMarkers(chatTemplate, vocab)) {
      return ChatFormat.CHATML;
    }
    if (tokenizerRoot != null && usesMetaspace(tokenizerRoot, vocab) &&
        vocab.containsKey("<bos>")) {
      return ChatFormat.TURN_BASED;
    }
    return ChatFormat.PLAIN;
  }

  private static boolean usesChatMlMarkers(final String chatTemplate,
                                           final Map<String, Integer> vocab) {
    return (chatTemplate != null && chatTemplate.contains("<|im_start|>"))
        || vocab.containsKey("<|im_start|>");
  }

  // True when tokenizer.json pretok/decoder/normalizer (or vocab sample) uses Metaspace ▁
  private static boolean usesMetaspace(final Map<String, Object> root,
                                       final Map<String, Integer> vocab) {
    if (containsMetaspaceNode(root.get("pre_tokenizer")) ||
      containsMetaspaceNode(root.get("decoder"))) {
      return true;
    }
    if (normalizerReplacesSpaceWithMeta(root.get("normalizer"))) {
      return true;
    }
    int sample = 0;
    int withMeta = 0;
    for (String t : vocab.keySet()) {
      if (sample++ > 4000) {
        break;
      }
      if (t.startsWith(META_SPACE)) {
        withMeta++;
      }
    }
    return withMeta > 200;
  }

  // Metaspace pretok usually prepends ▁; Replace-only normalizers only swap spaces → ▁
  private static boolean shouldPrependMetaSpace(final Map<String, Object> root) {
    return containsMetaspaceNode(root.get("pre_tokenizer"));
  }

  // Walk HF normalizer tree for Replace(" " → ▁)
  private static boolean normalizerReplacesSpaceWithMeta(final Object node) {
    if (!(node instanceof Map<?, ?> map)) {
      return false;
    }
    Object type = map.get("type");
    if (type instanceof String s && s.equalsIgnoreCase("Replace")) {
      Object pattern = map.get("pattern");
      Object content = map.get("content");
      String pat = pattern instanceof Map<?, ?> pm ? String.valueOf(pm.get("String")) : null;
      return " ".equals(pat) && META_SPACE.equals(content);
    }
    if (type instanceof String s && s.equalsIgnoreCase("Sequence")) {
      Object norms = map.get("normalizers");
      if (norms instanceof List<?> list) {
        for (Object o : list) {
          if (normalizerReplacesSpaceWithMeta(o)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  // Recursively find type=Metaspace in pretok / decoder JSON nodes
  private static boolean containsMetaspaceNode(final Object node) {
    if (!(node instanceof Map<?, ?> map)) {
      return false;
    }
    Object type = map.get("type");
    if (type instanceof String s && s.equalsIgnoreCase("Metaspace")) {
      return true;
    }
    Object pretenders = map.get("pretokenizers");
    if (pretenders instanceof List<?> list) {
      for (Object o : list) {
        if (containsMetaspaceNode(o)) {
          return true;
        }
      }
    }
    Object decoders = map.get("decoders");
    if (decoders instanceof List<?> list) {
      for (Object o : list) {
        if (containsMetaspaceNode(o)) {
          return true;
        }
      }
    }
    return false;
  }

  // Pick EOS id from config string, then common stop markers present in vocab
  private static int resolveEos(final Map<String, Integer> vocab, final String eosToken) {
    if (eosToken != null && vocab.containsKey(eosToken)) {
      return vocab.get(eosToken);
    }
    for (String marker : List.of(
        "<|im_end|>", "<eos>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eot_id|>"
    )) {
      if (vocab.containsKey(marker)) {
        return vocab.get(marker);
      }
    }
    throw new ModelLoadException(
        "cannot resolve EOS token id (set eos_token in tokenizer_config.json or GGUF eos_token_id)");
  }

  // Pad id from config / <pad> / <|endoftext|>, else reuse EOS
  private static int resolvePad(final Map<String, Integer> vocab, final String padToken,
                                final int eos) {
    if (padToken != null && vocab.containsKey(padToken)) {
      return vocab.get(padToken);
    }
    if (vocab.containsKey("<pad>")) {
      return vocab.get("<pad>");
    }
    if (vocab.containsKey("<|endoftext|>")) {
      return vocab.get("<|endoftext|>");
    }
    return eos;
  }

  // Append token id to stop list when the string exists in vocab
  private static void addStopIfPresent(final Map<String, Integer> vocab,
                                       final List<Integer> stopIds,
                                       final String token) {
    Integer id = vocab.get(token);
    if (id != null && !stopIds.contains(id)) {
      stopIds.add(id);
    }
  }

  // Last-resort tokenizer when tokenizer.json is missing (tiny identity map over first 256 chars)
  private static Tokenizer bare(final String modelConfigJson) {
    if (modelConfigJson == null || modelConfigJson.isBlank()) {
      throw new ModelLoadException("bare tokenizer requires config.json with vocab_size");
    }
    int vocabSize = Json.asInt(Json.parseObject(modelConfigJson).get("vocab_size"), -1);
    if (vocabSize <= 0) {
      throw new ModelLoadException("bare tokenizer requires positive vocab_size in config.json");
    }
    Map<String, Integer> vocab = new LinkedHashMap<>();
    for (int i = 0; i < Math.min(256, vocabSize); i++) {
      vocab.put(String.valueOf((char) i), i);
    }
    int eos = vocabSize - 1;
    return new Tokenizer(
      vocab, Map.of(), Set.of(), Set.of(),
        eos, List.of(eos), eos,
        null, true, Style.GPT2_BYTE_BPE, ChatFormat.PLAIN, false, false, false
    );
  }

  // Split text into Unicode code-point strings (Metaspace BPE input units)
  private static List<String> codepoints(final String text) {
    List<String> out = new ArrayList<>();
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      out.add(new String(Character.toChars(cp)));
      i += Character.charCount(cp);
    }
    return out;
  }

  /**
   * Decode UTF-8 but drop a trailing incomplete multi-byte sequence.
   * Needed for streamed token decode so Cyrillic (e.g. щ = {@code D1 89}) does not show as {@code �}.
   *
   * @param bytes raw UTF-8 bytes accumulated so far; {@code null}/empty → {@code ""}
   * @return decoded prefix of complete code points only
   */
  public static String decodeUtf8Complete(final byte[] bytes) {
    if (bytes == null || bytes.length == 0) {
      return "";
    }
    int complete = 0;
    int i = 0;
    while (i < bytes.length) {
      int need = utf8SequenceLength(bytes[i]);
      if (need < 1 || i + need > bytes.length) {
        break;
      }
      boolean ok = true;
      for (int j = 1; j < need; j++) {
        if ((bytes[i + j] & 0xC0) != 0x80) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        break;
      }
      i += need;
      complete = i;
    }
    return complete == 0 ? "" : new String(bytes, 0, complete, UTF_8);
  }

  // Expected byte count of a UTF-8 sequence from its lead byte, or -1 if invalid
  private static int utf8SequenceLength(final byte lead) {
    int v = lead & 0xFF;
    if (v < 0x80) {
      return 1;
    }
    if (v < 0xC2) {
      return -1;
    }
    if (v < 0xE0) {
      return 2;
    }
    if (v < 0xF0) {
      return 3;
    }
    if (v < 0xF5) {
      return 4;
    }
    return -1;
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
    return Optional.ofNullable(this.vocab.get(requireNonNull(token, "token")));
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
      && this.vocab.containsKey(thinkOpen)
      && this.vocab.containsKey(thinkClose);
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
   * Encodes plain text to vocabulary ids (BPE of the detected {@link Style}).
   *
   * @param text input text; non-{@code null} (may be empty)
   * @return immutable token-id list
   */
  public List<Integer> encode(final String text) {
    requireNonNull(text, "text");
    String prepared = this.lowercase ? text.toLowerCase(Locale.ROOT) : text;
    return switch (this.style) {
      case WORDPIECE -> this.encodeWordPiece(prepared);
      case METASPACE_BPE -> this.encodeMetaspace(prepared);
      case GPT2_BYTE_BPE -> this.encodeGpt2(prepared);
    };
  }

  private List<Integer> encodeWordPiece(final String text) {
    List<Integer> ids = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      String special = this.matchAdded(text, i);
      if (special != null) {
        ids.add(this.vocab.get(special));
        i += special.length();
        continue;
      }
      int nextSpecial = this.findNextAdded(text, i);
      String chunk = text.substring(i, nextSpecial);
      if (!chunk.isEmpty()) {
        String prepared = chunk.replace(" ", META_SPACE);
        if (this.prependMetaSpace) {
          prepared = META_SPACE + prepared;
        }
        ids.addAll(this.greedyWordPiece(prepared));
      }
      i = nextSpecial;
    }
    return ids;
  }

  private List<Integer> greedyWordPiece(final String text) {
    List<Integer> ids = new ArrayList<>();
    int i = 0;
    Integer unk = this.vocab.get("[UNK]");
    while (i < text.length()) {
      int matchedEnd = -1;
      Integer matchedId = null;
      for (int end = text.length(); end > i; end--) {
        Integer id = this.vocab.get(text.substring(i, end));
        if (id != null) {
          matchedEnd = end;
          matchedId = id;
          break;
        }
      }
      if (matchedId == null) {
        if (unk == null) {
          throw new IllegalStateException("missing vocab piece at index " + i + " and no [UNK]");
        }
        ids.add(unk);
        i += Character.charCount(text.codePointAt(i));
      } else {
        ids.add(matchedId);
        i = matchedEnd;
      }
    }
    return ids;
  }

  // GPT-2 path: greedy added-token match, else regex pieces → byte symbols → BPE
  private List<Integer> encodeGpt2(final String text) {
    List<Integer> ids = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      String special = this.matchAdded(text, i);
      if (special != null) {
        ids.add(this.vocab.get(special));
        i += special.length();
        continue;
      }
      Matcher matcher = GPT2_PATTERN.matcher(text);
      if (!matcher.find(i) || matcher.start() != i) {
        int cp = text.codePointAt(i);
        String ch = new String(Character.toChars(cp));
        Integer id = this.vocab.get(ch);
        if (id != null) {
          ids.add(id);
        }
        i += Character.charCount(cp);
        continue;
      }
      String piece = matcher.group();
      ids.addAll(this.bpe(this.utf8ToTokens(piece)));
      i = matcher.end();
    }
    return ids;
  }

  // Metaspace path: segments between specials; spaces → ▁; optional leading ▁; then BPE
  private List<Integer> encodeMetaspace(final String text) {
    List<Integer> ids = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      String special = this.matchAdded(text, i);
      if (special != null) {
        ids.add(this.vocab.get(special));
        i += special.length();
        continue;
      }
      int nextSpecial = this.findNextAdded(text, i);
      String chunk = text.substring(i, nextSpecial);
      if (!chunk.isEmpty()) {
        String prepared = chunk.replace(" ", META_SPACE);
        if (this.prependMetaSpace) {
          prepared = META_SPACE + prepared;
        }
        ids.addAll(this.bpe(codepoints(prepared)));
      }
      i = nextSpecial;
    }
    return ids;
  }

  // Index of the next added/special token after {@code from}, or text.length()
  private int findNextAdded(final String text, final int from) {
    int best = text.length();
    for (String added : this.addedTokensByLength) {
      int at = text.indexOf(added, from);
      if (at >= from && at < best) {
        best = at;
      }
    }
    return best;
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
    StringBuilder sb = new StringBuilder();
    for (int id : tokenIds) {
      if (skipSpecialTokens && this.skipTokenIds.contains(id)) {
        continue;
      }
      String tok = this.idToToken.get(id);
      if (tok == null) {
        continue;
      }
      sb.append(tok);
    }
    if (this.style == Style.METASPACE_BPE || this.style == Style.WORDPIECE) {
      return this.decodeMetaspace(sb.toString());
    }
    return this.tokensToUtf8(sb.toString());
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

  // Turn-based layout: optional system folded into first user; roles user|model with start/end markers
  private String applyTurnBasedChat(final List<Map<String, String>> messages,
                                    final boolean addGenerationPrompt) {
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
    if (this.vocab.containsKey("<bos>")) {
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

  // Longest added/special token that matches at {@code index}, or null
  private String matchAdded(final String text, final int index) {
    for (String added : this.addedTokensByLength) {
      if (text.startsWith(added, index)) {
        return added;
      }
    }
    return null;
  }

  // Map a UTF-8 string to GPT-2 BPE alphabet symbols (one symbol per byte)
  private List<String> utf8ToTokens(final String text) {
    byte[] bytes = text.getBytes(UTF_8);
    List<String> tokens = new ArrayList<>(bytes.length);
    for (byte b : bytes) {
      tokens.add(BYTE_ENCODER[b & 0xFF]);
    }
    return tokens;
  }

  // Metaspace decode: expand <0xHH> byte runs, then ▁ → space
  private String decodeMetaspace(final String tokenString) {
    StringBuilder out = new StringBuilder();
    int i = 0;
    List<Byte> bytes = new ArrayList<>();
    while (i < tokenString.length()) {
      if (tokenString.startsWith("<0x", i) && i + 5 < tokenString.length() &&
        tokenString.charAt(i + 5) == '>') {
        try {
          int b = Integer.parseInt(tokenString.substring(i + 3, i + 5), 16);
          bytes.add((byte) b);
          i += 6;
          continue;
        } catch (NumberFormatException ignored) {
          // fall through
        }
      }
      if (!bytes.isEmpty()) {
        byte[] arr = new byte[bytes.size()];
        for (int j = 0; j < bytes.size(); j++) {
          arr[j] = bytes.get(j);
        }
        out.append(decodeUtf8Complete(arr));
        bytes.clear();
      }
      out.append(tokenString.charAt(i));
      i++;
    }
    if (!bytes.isEmpty()) {
      byte[] arr = new byte[bytes.size()];
      for (int j = 0; j < bytes.size(); j++) {
        arr[j] = bytes.get(j);
      }
      out.append(decodeUtf8Complete(arr));
    }
    return out.toString().replace(META_SPACE, " ");
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
      return this.applyTurnBasedChat(messages, addGenerationPrompt);
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

  // GPT-2 decode: map BPE alphabet symbols back to UTF-8 bytes (preserve added tokens as text)
  private String tokensToUtf8(final String tokenString) {
    StringBuilder out = new StringBuilder();
    int i = 0;
    while (i < tokenString.length()) {
      String special = this.matchAdded(tokenString, i);
      if (special != null) {
        out.append(special);
        i += special.length();
        continue;
      }
      String ch = String.valueOf(tokenString.charAt(i));
      Integer b = BYTE_DECODER.get(ch);
      if (b != null) {
        List<Byte> bytes = new ArrayList<>();
        while (i < tokenString.length()) {
          if (this.matchAdded(tokenString, i) != null) {
            break;
          }
          String c = String.valueOf(tokenString.charAt(i));
          Integer bb = BYTE_DECODER.get(c);
          if (bb == null) {
            break;
          }
          bytes.add(bb.byteValue());
          i++;
        }
        byte[] arr = new byte[bytes.size()];
        for (int j = 0; j < bytes.size(); j++) {
          arr[j] = bytes.get(j);
        }
        out.append(decodeUtf8Complete(arr));
      } else {
        out.append(tokenString.charAt(i));
        i++;
      }
    }
    return out.toString();
  }

  // Greedy lowest-rank pair merges until no merge applies; missing pieces use byteFallback
  private List<Integer> bpe(final List<String> tokens) {
    if (tokens.isEmpty()) {
      return List.of();
    }
    List<String> word = new ArrayList<>(tokens);
    while (word.size() > 1) {
      int bestRank = Integer.MAX_VALUE;
      int bestIndex = -1;
      for (int i = 0; i < word.size() - 1; i++) {
        Integer rank = this.merges.get(word.get(i) + " " + word.get(i + 1));
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) {
        break;
      }
      List<String> next = new ArrayList<>();
      int i = 0;
      while (i < word.size()) {
        if (i == bestIndex) {
          next.add(word.get(i) + word.get(i + 1));
          i += 2;
        } else {
          next.add(word.get(i));
          i++;
        }
      }
      word = next;
    }
    List<Integer> ids = new ArrayList<>(word.size());
    for (String w : word) {
      Integer id = this.vocab.get(w);
      if (id != null) {
        ids.add(id);
      } else if (this.byteFallback) {
        if (this.style == Style.METASPACE_BPE) {
          for (byte b : w.getBytes(UTF_8)) {
            String piece = "<0x%02X>".formatted(b & 0xFF);
            Integer bid = this.vocab.get(piece);
            if (bid != null) {
              ids.add(bid);
            }
          }
        } else {
          for (int i = 0; i < w.length(); ) {
            int cp = w.codePointAt(i);
            String piece = new String(Character.toChars(cp));
            Integer bid = this.vocab.get(piece);
            if (bid == null) {
              for (byte b : piece.getBytes(UTF_8)) {
                Integer bb = this.vocab.get(BYTE_ENCODER[b & 0xFF]);
                if (bb != null) {
                  ids.add(bb);
                }
              }
            } else {
              ids.add(bid);
            }
            i += Character.charCount(cp);
          }
        }
      }
    }
    return ids;
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
     * {@code <start_of_turn>} / {@code <end_of_turn>} turns (system folded into first user).
     */
    TURN_BASED,
    /**
     * Plain {@code role: text} fallback when no chat markers are present.
     */
    PLAIN
  }

  /**
   * Encode/decode algorithm selected at load from HF JSON or GGUF heuristics.
   */
  private enum Style {
    /** GPT-2 byte-level BPE (common with ChatML exports). */
    GPT2_BYTE_BPE,
    /** SentencePiece-style BPE with {@code ▁} word boundaries. */
    METASPACE_BPE,
    /** BERT WordPiece / greedy longest-match. */
    WORDPIECE
  }
}
