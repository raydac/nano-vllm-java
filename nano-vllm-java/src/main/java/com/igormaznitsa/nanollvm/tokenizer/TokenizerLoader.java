package com.igormaznitsa.nanollvm.tokenizer;

import static com.igormaznitsa.nanollvm.utils.NanoLlvmProps.CONFIG_JSON;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer.ChatFormat;
import com.igormaznitsa.nanollvm.tokenizer.UnigramCodec.UnigramScores;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Constructs {@link Tokenizer} from JSON, SentencePiece, or GGUF sidecars.
 *
 * @since 1.1.1
 */
final class TokenizerLoader {

  private static final String TOKENIZER_MODEL = "tokenizer.model";

  private TokenizerLoader() {
  }

  static Tokenizer fromPretrained(final Path modelDir) {
    try {
      Path tokenizerJson = modelDir.resolve("tokenizer.json");
      Path spm = modelDir.resolve(TOKENIZER_MODEL);
      String tokenizerConfig = readIfPresent(modelDir.resolve("tokenizer_config.json"));
      String generationConfig = readIfPresent(modelDir.resolve("generation_config.json"));
      String modelConfig = readIfPresent(modelDir.resolve(CONFIG_JSON));
      if (Files.isRegularFile(tokenizerJson)) {
        return fromJsonDocuments(
          Files.readString(tokenizerJson), tokenizerConfig, generationConfig, modelConfig);
      }
      if (Files.isRegularFile(spm)) {
        return fromSentencePiece(
          Files.readAllBytes(spm), tokenizerConfig, generationConfig);
      }
      return bare(modelConfig);
    } catch (IOException e) {
      throw new ModelLoadException("failed to load tokenizer from " + modelDir, e);
    }
  }

  static Tokenizer fromJsonDocuments(
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
      if (model == null) {
        throw new ModelLoadException("tokenizer.json missing model object");
      }
      HfVocab hfVocab = readHfVocab(model);
      Map<String, Integer> vocab = hfVocab.ids();
      Set<String> addedTexts = new HashSet<>();
      Set<String> specialTexts = new HashSet<>();
      collectAddedTokens(root.get("added_tokens"), vocab, addedTexts, specialTexts);

      Map<String, Integer> merges = readMerges(model.get("merges"));
      Sidecars sidecars = Sidecars.read(tokenizerConfigJson, generationConfigJson);
      ChatFormat chatFormat = detectChatFormat(vocab, sidecars.chatTemplate(), root);
      TokenVocab tokenVocab = new TokenVocab(vocab, addedTexts, specialTexts);
      TokenCodec codec = codecFromHf(
        Json.asString(model.get("type")),
        tokenVocab,
        vocab,
        hfVocab,
        merges,
        model,
        usesMetaspace(root, vocab),
        shouldPrependMetaSpace(root));

      int eos = resolveEos(vocab, sidecars.eosToken());
      int pad = resolvePad(vocab, sidecars.padToken(), eos);
      List<Integer> stopIds = stopIds(vocab, eos, sidecars.generationEosIds());
      boolean inviteThinking = vocab.containsKey("<think>") && vocab.containsKey("</think>");
      TextNormalizer normalizer = TextNormalizer.fromHf(root.get("normalizer"));
      return new Tokenizer(
        tokenVocab, codec, normalizer, chatFormat, eos, stopIds, pad, inviteThinking);
    } catch (RuntimeException e) {
      throw new ModelLoadException("failed to load tokenizer from JSON documents", e);
    }
  }

  static Tokenizer fromGguf(final GgufTokenizerSource source) {
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

    String ggmlModel = source.metaString("tokenizer.ggml.model", "gpt2").toLowerCase(ROOT);
    boolean bertWordPiece = ggmlModel.contains("bert");
    for (String marker : List.of("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")) {
      if (vocab.containsKey(marker)) {
        addedTexts.add(marker);
        specialTexts.add(marker);
      }
    }
    boolean metaspace =
      tokens.stream().limit(4000).filter(t -> t.startsWith(MetaspaceText.MARK)).count() > 200
        || ggmlModel.contains("llama")
        || ggmlModel.contains("spm");
    TokenVocab tokenVocab = new TokenVocab(vocab, addedTexts, specialTexts);
    TokenCodec codec;
    TextNormalizer normalizer = TextNormalizer.identity();
    if (bertWordPiece) {
      codec = WordPieceCodec.metaspace(tokenVocab, true);
      normalizer = TextNormalizer.lowercase();
    } else if (metaspace) {
      codec = new MetaspaceBpeCodec(tokenVocab, new BpeMerges(merges), false, false);
    } else {
      codec = new Gpt2ByteBpeCodec(tokenVocab, new BpeMerges(merges), true);
    }

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
    boolean inviteThinking = vocab.containsKey("<think>") && vocab.containsKey("</think>");
    return new Tokenizer(
      tokenVocab, codec, normalizer, chatFormat, eos, stopIds, pad, inviteThinking);
  }

  /**
   * Builds a tokenizer from a SentencePiece {@code tokenizer.model} protobuf.
   *
   * @since 1.1.1
   */
  static Tokenizer fromSentencePiece(
    final byte[] modelBytes,
    final String tokenizerConfigJson,
    final String generationConfigJson
  ) {
    requireNonNull(modelBytes, "modelBytes");
    SentencePieceModel.Parsed parsed = SentencePieceModel.parse(modelBytes);
    Map<String, Integer> vocab = new LinkedHashMap<>(parsed.vocab());
    Set<String> added = new HashSet<>(parsed.added());
    Set<String> specials = new HashSet<>(parsed.specials());
    TokenVocab tokenVocab = new TokenVocab(vocab, added, specials);
    TokenCodec codec = switch (parsed.modelType()) {
      case SentencePieceModel.BPE -> new MetaspaceBpeCodec(
        tokenVocab, scoresArray(parsed), parsed.addDummyPrefix(), parsed.byteFallback());
      case SentencePieceModel.WORD ->
        new WordLevelCodec(tokenVocab, unkToken(vocab, parsed.unkId()));
      case SentencePieceModel.CHAR -> new CharCodec(tokenVocab, unkToken(vocab, parsed.unkId()));
      default -> new UnigramCodec(
        tokenVocab,
        UnigramScores.of(vocab, parsed.scores(), parsed.unkId()),
        parsed.addDummyPrefix());
    };
    TextNormalizer normalizer = parsed.charsmap().length == 0
      ? TextNormalizer.identity()
      : TextNormalizer.of(PrecompiledCharsMap.parse(parsed.charsmap())::normalize);
    Sidecars sidecars = Sidecars.read(tokenizerConfigJson, generationConfigJson);
    ChatFormat chatFormat = detectChatFormat(vocab, sidecars.chatTemplate(), null);
    int eos = resolveEos(vocab, sidecars.eosToken());
    int pad = resolvePad(vocab, sidecars.padToken(), eos);
    List<Integer> stopIds = stopIds(vocab, eos, sidecars.generationEosIds());
    boolean inviteThinking = vocab.containsKey("<think>") && vocab.containsKey("</think>");
    return new Tokenizer(
      tokenVocab, codec, normalizer, chatFormat, eos, stopIds, pad, inviteThinking);
  }

  static Tokenizer bare(final String modelConfigJson) {
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
    TokenVocab tokenVocab = new TokenVocab(vocab, Set.of(), Set.of());
    return new Tokenizer(
      tokenVocab,
      new Gpt2ByteBpeCodec(tokenVocab, new BpeMerges(Map.of()), true),
      TextNormalizer.identity(),
      ChatFormat.PLAIN,
      eos,
      List.of(eos),
      eos,
      false);
  }

  private static TokenCodec codecFromHf(
    final String modelType,
    final TokenVocab tokenVocab,
    final Map<String, Integer> vocab,
    final HfVocab hfVocab,
    final Map<String, Integer> merges,
    final Map<String, Object> model,
    final boolean metaspace,
    final boolean prependMetaSpace
  ) {
    String type = modelType == null ? "" : modelType;
    if ("Unigram".equalsIgnoreCase(type) || hfVocab.unigram()) {
      return new UnigramCodec(
        tokenVocab,
        UnigramScores.of(vocab, hfVocab.scoresById(), Json.asInt(model.get("unk_id"), -1)),
        prependMetaSpace);
    }
    if ("WordPiece".equalsIgnoreCase(type)) {
      return WordPieceCodec.bert(
        tokenVocab, Json.asString(model.get("continuing_subword_prefix")));
    }
    if ("WordLevel".equalsIgnoreCase(type)) {
      String unk = Json.asString(model.get("unk_token"));
      return new WordLevelCodec(tokenVocab, unk == null ? "[UNK]" : unk);
    }
    boolean byteFallback = Json.asBoolean(model.get("byte_fallback"), !metaspace);
    if (metaspace) {
      return new MetaspaceBpeCodec(
        tokenVocab, new BpeMerges(merges), prependMetaSpace, byteFallback);
    }
    return new Gpt2ByteBpeCodec(tokenVocab, new BpeMerges(merges), byteFallback);
  }

  private static void collectAddedTokens(
    final Object addedNode,
    final Map<String, Integer> vocab,
    final Set<String> addedTexts,
    final Set<String> specialTexts
  ) {
    List<Object> added = Json.asArray(addedNode);
    if (added == null) {
      return;
    }
    for (Object item : added) {
      Map<String, Object> tok = Json.asObject(item);
      if (tok == null) {
        continue;
      }
      String content = Json.asString(tok.get("content"));
      int id = Json.asInt(tok.get("id"), -1);
      if (content == null || id < 0) {
        continue;
      }
      vocab.put(content, id);
      addedTexts.add(content);
      if (Json.asBoolean(tok.get("special"), false)
        || content.equals("<think>")
        || content.equals("</think>")
        || content.equals("<bos>")
        || content.equals("<eos>")
        || content.equals("<start_of_turn>")
        || content.equals("<end_of_turn>")
        || content.equals("<|turn>")
        || content.equals("<turn|>")) {
        specialTexts.add(content);
      }
    }
  }

  private static Map<String, Integer> readMerges(final Object mergeNode) {
    Map<String, Integer> merges = new HashMap<>();
    List<Object> mergeList = Json.asArray(mergeNode);
    if (mergeList == null) {
      return merges;
    }
    for (int i = 0; i < mergeList.size(); i++) {
      Object m = mergeList.get(i);
      String merge;
      if (m instanceof String s) {
        merge = s;
      } else if (m instanceof List<?> pair && pair.size() >= 2) {
        merge = pair.get(0) + " " + pair.get(1);
      } else {
        continue;
      }
      merges.put(merge, i);
    }
    return merges;
  }

  private static ChatFormat detectChatFormat(
    final Map<String, Integer> vocab,
    final String chatTemplate,
    final Map<String, Object> tokenizerRoot
  ) {
    if (chatTemplate != null && chatTemplate.contains("<|turn>")) {
      return ChatFormat.TURN_BASED;
    }
    if (vocab.containsKey("<|turn>")) {
      return ChatFormat.TURN_BASED;
    }
    if (chatTemplate != null && chatTemplate.contains("start_of_turn")) {
      return ChatFormat.TURN_BASED;
    }
    if (vocab.containsKey("<start_of_turn>")) {
      return ChatFormat.TURN_BASED;
    }
    if ((chatTemplate != null && chatTemplate.contains("<|im_start|>"))
      || vocab.containsKey("<|im_start|>")) {
      return ChatFormat.CHATML;
    }
    if (tokenizerRoot != null && usesMetaspace(tokenizerRoot, vocab)
      && vocab.containsKey("<bos>")) {
      return ChatFormat.TURN_BASED;
    }
    return ChatFormat.PLAIN;
  }

  private static boolean usesMetaspace(
    final Map<String, Object> root,
    final Map<String, Integer> vocab
  ) {
    if (containsMetaspaceNode(root.get("pre_tokenizer"))
      || containsMetaspaceNode(root.get("decoder"))) {
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
      if (t.startsWith(MetaspaceText.MARK)) {
        withMeta++;
      }
    }
    return withMeta > 200;
  }

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
      return " ".equals(pat) && MetaspaceText.MARK.equals(content);
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

  private static int resolvePad(
    final Map<String, Integer> vocab,
    final String padToken,
    final int eos
  ) {
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

  private static void addStopIfPresent(
    final Map<String, Integer> vocab,
    final List<Integer> stopIds,
    final String token
  ) {
    Integer id = vocab.get(token);
    if (id != null && !stopIds.contains(id)) {
      stopIds.add(id);
    }
  }

  private static List<Integer> stopIds(
    final Map<String, Integer> vocab,
    final int eos,
    final List<Integer> generationEosIds
  ) {
    List<Integer> stopIds = new ArrayList<>();
    stopIds.add(eos);
    addStopIfPresent(vocab, stopIds, "<|endoftext|>");
    addStopIfPresent(vocab, stopIds, "<eos>");
    addStopIfPresent(vocab, stopIds, "<end_of_turn>");
    addStopIfPresent(vocab, stopIds, "<turn|>");
    addStopIfPresent(vocab, stopIds, "<|im_end|>");
    for (int id : generationEosIds) {
      if (id >= 0 && !stopIds.contains(id)) {
        stopIds.add(id);
      }
    }
    return stopIds;
  }

  private static HfVocab readHfVocab(final Map<String, Object> model) {
    Object vocabNode = model.get("vocab");
    List<Object> rows = Json.asArray(vocabNode);
    if (rows != null) {
      Map<String, Integer> ids = new LinkedHashMap<>(rows.size() * 2);
      Map<Integer, Float> scores = new HashMap<>(rows.size() * 2);
      for (int i = 0; i < rows.size(); i++) {
        List<Object> row = Json.asArray(rows.get(i));
        if (row == null || row.isEmpty()) {
          continue;
        }
        String piece = Json.asString(row.getFirst());
        if (piece == null) {
          continue;
        }
        ids.put(piece, i);
        if (row.size() > 1) {
          scores.put(i, Json.asFloat(row.get(1), 0f));
        }
      }
      return new HfVocab(ids, scores, true);
    }
    Map<String, Object> map = Json.asObject(vocabNode);
    if (map == null) {
      throw new ModelLoadException("tokenizer.json model.vocab is missing");
    }
    Map<String, Integer> ids = new LinkedHashMap<>(map.size() * 2);
    for (var e : map.entrySet()) {
      ids.put(e.getKey(), Json.asInt(e.getValue(), 0));
    }
    boolean unigram = "Unigram".equalsIgnoreCase(Json.asString(model.get("type")));
    return new HfVocab(ids, Map.of(), unigram);
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

  private static String readIfPresent(final Path path) throws IOException {
    return Files.isRegularFile(path) ? Files.readString(path) : null;
  }

  private static float[] scoresArray(final SentencePieceModel.Parsed parsed) {
    int max = -1;
    for (int id : parsed.scores().keySet()) {
      max = Math.max(max, id);
    }
    float[] scores = new float[max + 1];
    for (var e : parsed.scores().entrySet()) {
      scores[e.getKey()] = e.getValue();
    }
    return scores;
  }

  private static String unkToken(final Map<String, Integer> vocab, final int unkId) {
    for (var e : vocab.entrySet()) {
      if (e.getValue() == unkId) {
        return e.getKey();
      }
    }
    return "<unk>";
  }

  private record HfVocab(
    Map<String, Integer> ids,
    Map<Integer, Float> scoresById,
    boolean unigram
  ) {
  }

  private record Sidecars(
    String chatTemplate,
    String eosToken,
    String padToken,
    List<Integer> generationEosIds
  ) {
    static Sidecars read(final String tokenizerConfigJson, final String generationConfigJson) {
      String chatTemplate = null;
      String eosToken = null;
      String padToken = null;
      if (tokenizerConfigJson != null && !tokenizerConfigJson.isBlank()) {
        Map<String, Object> tc = Json.parseObject(tokenizerConfigJson);
        chatTemplate = Json.asString(tc.get("chat_template"));
        eosToken = tokenString(tc.get("eos_token"));
        padToken = tokenString(tc.get("pad_token"));
      }
      List<Integer> generationEosIds = new ArrayList<>();
      if (generationConfigJson != null && !generationConfigJson.isBlank()) {
        Map<String, Object> gc = Json.parseObject(generationConfigJson);
        Object eosField = gc.get("eos_token_id");
        if (eosField instanceof List<?> list) {
          for (Object o : list) {
            int id = Json.asInt(o, -1);
            if (id >= 0) {
              generationEosIds.add(id);
            }
          }
        } else if (eosField != null) {
          int id = Json.asInt(eosField, -1);
          if (id >= 0) {
            generationEosIds.add(id);
          }
        }
      }
      return new Sidecars(chatTemplate, eosToken, padToken, List.copyOf(generationEosIds));
    }
  }
}
