package com.igormaznitsa.nanollvm.tokenizer;

import static com.igormaznitsa.nanollvm.utils.NanoVllmProps.CONFIG_JSON;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.utils.Json;

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
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HF {@code tokenizer.json} loader: GPT-2 byte-level BPE (Qwen) or Metaspace/{@code ▁} BPE (Gemma).
 */
public final class Tokenizer {

  private static final Pattern GPT2_PATTERN = Pattern.compile(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  );
  private static final String META_SPACE = "\u2581"; // ▁
  private static final String[] BYTE_ENCODER = new String[256];
  private static final Map<String, Integer> BYTE_DECODER = new HashMap<>();

  static {
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

  private final Map<String, Integer> vocab;
  private final Map<Integer, String> idToToken;
  private final Map<String, Integer> merges;
  private final List<String> addedTokensByLength;
  private final Set<Integer> skipTokenIds;
  private final int eosTokenId;
  private final List<Integer> stopTokenIds;
  private final int padTokenId;
  private final String chatTemplate;
  private final boolean byteFallback;
  private final Style style;
  private final boolean gemmaChat;
  /**
   * When true, prepend {@code ▁} (Metaspace); Gemma uses replace-only normalizer.
   */
  private final boolean prependMetaSpace;
  /**
   * When true, chat may use a {@code <think>} scratchpad (Qwen-style). LFM2 GGUF is false.
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
      final boolean gemmaChat,
      final boolean prependMetaSpace,
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
    this.gemmaChat = gemmaChat;
    this.prependMetaSpace = prependMetaSpace;
    this.inviteThinking = inviteThinking;
  }

  public static Tokenizer fromPretrained(final Path modelDir) {
    try {
      Path tokenizerJson = modelDir.resolve("tokenizer.json");
      if (!Files.isRegularFile(tokenizerJson)) {
        return bare(modelDir);
      }
      Map<String, Object> root = Json.parseObject(Files.readString(tokenizerJson));
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

      Path config = modelDir.resolve("tokenizer_config.json");
      String chatTemplate = null;
      String eosToken = null;
      String padToken = null;
      if (Files.isRegularFile(config)) {
        Map<String, Object> tc = Json.parseObject(Files.readString(config));
        chatTemplate = Json.asString(tc.get("chat_template"));
        eosToken = tokenString(tc.get("eos_token"));
        padToken = tokenString(tc.get("pad_token"));
      }

      boolean gemmaChat = isGemmaStyle(modelDir, vocab, chatTemplate, root);
      Style style =
          gemmaChat || usesMetaspace(root, vocab) ? Style.METASPACE_BPE : Style.GPT2_BYTE_BPE;
      boolean prependMetaSpace = shouldPrependMetaSpace(root);

      int eos = resolveEos(vocab, eosToken, gemmaChat);
      int pad = resolvePad(vocab, padToken, eos);
      List<Integer> stopIds = new ArrayList<>();
      stopIds.add(eos);
      addStopIfPresent(vocab, stopIds, "<|endoftext|>");
      addStopIfPresent(vocab, stopIds, "<eos>");
      addStopIfPresent(vocab, stopIds, "<end_of_turn>");

      Path genConfig = modelDir.resolve("generation_config.json");
      if (Files.isRegularFile(genConfig)) {
        Map<String, Object> gc = Json.parseObject(Files.readString(genConfig));
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
      return new Tokenizer(
          vocab, merges, addedTexts, specialTexts, eos, stopIds, pad,
        chatTemplate, byteFallback, style, gemmaChat, prependMetaSpace, !gemmaChat
      );
    } catch (IOException e) {
      throw new IllegalStateException("failed to load tokenizer from " + modelDir, e);
    }
  }

  /**
   * Builds a tokenizer from GGUF {@code tokenizer.ggml.*} metadata (LFM2 and similar).
   * Prefer {@link com.igormaznitsa.nanollvm.models.LlmModelFactory} for application load paths.
   */
  public static Tokenizer fromGguf(final GgufTokenizerSource source) {
    requireNonNull(source, "source");
    List<String> tokens = source.metaStringArray("tokenizer.ggml.tokens");
    if (tokens.isEmpty()) {
      throw new IllegalStateException("GGUF missing tokenizer.ggml.tokens");
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
    boolean metaspace =
      tokens.stream().limit(4000).filter(t -> t.startsWith(META_SPACE)).count() > 200
        || ggmlModel.contains("llama")
        || ggmlModel.contains("spm");
    Style style = metaspace ? Style.METASPACE_BPE : Style.GPT2_BYTE_BPE;
    boolean prependMetaSpace = metaspace && !ggmlModel.contains("lfm");

    int eos = source.metaInt("tokenizer.ggml.eos_token_id", -1);
    if (eos < 0) {
      eos = resolveEos(vocab, null, false);
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
    boolean gemmaChat = chatTemplate != null && chatTemplate.contains("start_of_turn");
    boolean byteFallback = style == Style.GPT2_BYTE_BPE;

    return new Tokenizer(
      vocab, merges, addedTexts, specialTexts, eos, stopIds, pad,
      chatTemplate, byteFallback, style, gemmaChat, prependMetaSpace, false
    );
  }

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

  private static boolean isGemmaStyle(
      final Path modelDir,
      final Map<String, Integer> vocab,
      final String chatTemplate,
      final Map<String, Object> tokenizerRoot
  ) throws IOException {
    if (chatTemplate != null && chatTemplate.contains("start_of_turn")) {
      return true;
    }
    if (vocab.containsKey("<start_of_turn>") || vocab.containsKey("<bos>")) {
      Path cfg = modelDir.resolve(CONFIG_JSON);
      if (Files.isRegularFile(cfg)) {
        String mt = Json.asString(Json.parseObject(Files.readString(cfg)).get("model_type"));
        if (mt != null && mt.toLowerCase(Locale.ROOT).contains("gemma")) {
          return true;
        }
      }
    }
    return usesMetaspace(tokenizerRoot, vocab) && vocab.containsKey("<bos>");
  }

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

  /**
   * Metaspace pretokenizer usually prepends ▁; Gemma's Replace normalizer does not.
   */
  private static boolean shouldPrependMetaSpace(final Map<String, Object> root) {
    return containsMetaspaceNode(root.get("pre_tokenizer"));
  }

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

  private static int resolveEos(final Map<String, Integer> vocab, final String eosToken,
                                final boolean gemma) {
    if (eosToken != null && vocab.containsKey(eosToken)) {
      return vocab.get(eosToken);
    }
    if (gemma) {
      if (vocab.containsKey("<eos>")) {
        return vocab.get("<eos>");
      }
      if (vocab.containsKey("<end_of_turn>")) {
        return vocab.get("<end_of_turn>");
      }
      return 1;
    }
    if (vocab.containsKey("<|im_end|>")) {
      return vocab.get("<|im_end|>");
    }
    if (vocab.containsKey("<|endoftext|>")) {
      return vocab.get("<|endoftext|>");
    }
    return 151645;
  }

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

  private static void addStopIfPresent(final Map<String, Integer> vocab,
                                       final List<Integer> stopIds,
                                       final String token) {
    Integer id = vocab.get(token);
    if (id != null && !stopIds.contains(id)) {
      stopIds.add(id);
    }
  }

  private static Tokenizer bare(final Path modelDir) throws IOException {
    Path config = modelDir.resolve(CONFIG_JSON);
    int vocabSize = 151936;
    if (Files.isRegularFile(config)) {
      vocabSize =
          Json.asInt(Json.parseObject(Files.readString(config)).get("vocab_size"), vocabSize);
    }
    Map<String, Integer> vocab = new LinkedHashMap<>();
    for (int i = 0; i < Math.min(256, vocabSize); i++) {
      vocab.put(String.valueOf((char) i), i);
    }
    return new Tokenizer(
        vocab, Map.of(), Set.of(), Set.of(),
        vocabSize - 1, List.of(vocabSize - 1), vocabSize - 1,
      null, true, Style.GPT2_BYTE_BPE, false, false, false
    );
  }

  private static List<String> codepoints(final String text) {
    List<String> out = new ArrayList<>();
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      out.add(new String(Character.toChars(cp)));
      i += Character.charCount(cp);
    }
    return out;
  }

  public int eosTokenId() {
    return this.eosTokenId;
  }

  public List<Integer> stopTokenIds() {
    return this.stopTokenIds;
  }

  public int padTokenId() {
    return this.padTokenId;
  }

  /**
   * {@code gemma3} / metaspace vs Qwen ChatML.
   */
  public boolean isGemmaChat() {
    return this.gemmaChat;
  }

  /**
   * Decode UTF-8 but drop a trailing incomplete multi-byte sequence.
   * Needed for streamed token decode so Cyrillic (e.g. щ = D1 89) does not show as �.
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

  /**
   * Whether this chat path invites a {@code <think>} scratchpad (Qwen-style).
   * LFM2 GGUF and Gemma return {@code false}.
   */
  public boolean invitesThinking() {
    return this.inviteThinking;
  }

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

  public List<Integer> encode(final String text) {
    return this.style == Style.METASPACE_BPE ? this.encodeMetaspace(text) : this.encodeGpt2(text);
  }

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
        ids.addAll(this.bpe(this.codepoints(prepared)));
      }
      i = nextSpecial;
    }
    return ids;
  }

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

  public String decode(final List<Integer> tokenIds) {
    return this.decode(tokenIds, false);
  }

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
    if (this.style == Style.METASPACE_BPE) {
      return this.decodeMetaspace(sb.toString());
    }
    return this.tokensToUtf8(sb.toString());
  }

  public String applyChatTemplate(final List<Map<String, String>> messages,
                                  final boolean addGenerationPrompt) {
    return this.applyChatTemplate(messages, addGenerationPrompt, true);
  }

  private String applyGemmaChat(final List<Map<String, String>> messages,
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
        content = ChatPrompts.gemmaUserContent(system, content, firstUser);
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

  private String matchAdded(final String text, final int index) {
    for (String added : this.addedTokensByLength) {
      if (text.startsWith(added, index)) {
        return added;
      }
    }
    return null;
  }

  private List<String> utf8ToTokens(final String text) {
    byte[] bytes = text.getBytes(UTF_8);
    List<String> tokens = new ArrayList<>(bytes.length);
    for (byte b : bytes) {
      tokens.add(BYTE_ENCODER[b & 0xFF]);
    }
    return tokens;
  }

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

  public String applyChatTemplate(final List<Map<String, String>> messages,
                                  final boolean addGenerationPrompt,
                                  final boolean enableThinking) {
    if (this.gemmaChat ||
        (this.chatTemplate != null && this.chatTemplate.contains("start_of_turn"))) {
      return this.applyGemmaChat(messages, addGenerationPrompt);
    }
    if (this.usesChatMl()) {
      StringBuilder sb = new StringBuilder();
      for (Map<String, String> msg : messages) {
        String role = msg.getOrDefault("role", "user");
        String content = msg.getOrDefault("content", "");
        sb.append("<|im_start|>").append(role).append('\n').append(content).append("<|im_end|>\n");
      }
      if (addGenerationPrompt) {
        sb.append("<|im_start|>assistant\n");
        if (!enableThinking && this.inviteThinking) {
          sb.append("<think>\n\n</think>\n\n");
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

  private boolean usesChatMl() {
    return (this.chatTemplate != null && this.chatTemplate.contains("<|im_start|>"))
      || this.vocab.containsKey("<|im_start|>");
  }

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

  private enum Style {
    GPT2_BYTE_BPE,
    METASPACE_BPE
  }
}
