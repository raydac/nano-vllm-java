package com.igormaznitsa.nanollvm.llm;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Json;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Sizes and symbol inventory of a tiny-voca text-to-speech checkpoint.
 *
 * <p>English releases ({@code n4ze3m/tiny-voca}) store hyperparameters under {@code model} and
 * the inventory as a {@code vocab} map. The Russian release ({@code egnch/tiny-voca-ru}) stores
 * hyperparameters under {@code config} and the inventory as a {@code symbols} list. Both share
 * one acoustic graph. Voices, phones, and pauses are read from the file.
 *
 * @param sampleRate        waveform Hertz
 * @param hop               acoustic hop in samples
 * @param channels          residual width
 * @param heads             attention heads
 * @param feedforward       block expand width
 * @param kernel            depthwise convolution taps (odd)
 * @param encoderLayers     token-stack depth
 * @param prosodyLayers     frame-stack depth before pitch
 * @param acousticLayers    frame-stack depth after pitch
 * @param predictorChannels duration-convolution width
 * @param synthFft          harmonic filter size
 * @param synthHop          harmonic filter hop
 * @param noiseFft          noise filter size
 * @param noiseHop          noise filter hop
 * @param cepCausal         causal cepstrum bins
 * @param cepAnti           anticausal cepstrum bins
 * @param noiseCep          noise cepstrum bins
 * @param harmonicLimit     fraction of the sample rate kept as harmonics
 * @param maxTokens         longest token sequence one call accepts
 * @param maxFrames         longest frame sequence one call accepts
 * @param f0Min             lowest pitch in Hertz
 * @param f0Max             highest pitch in Hertz
 * @param pitchReferenceHz  Hertz at log-pitch zero
 * @param symbols           inventory in id order, including pauses and {@code <unk>}
 * @param speakers          voice names in id order
 * @param language          {@code ru} when the inventory has Russian palatal phones, otherwise {@code en}
 * @since 1.5.1
 */
public record TinyVocaSpec(
  int sampleRate,
  int hop,
  int channels,
  int heads,
  int feedforward,
  int kernel,
  int encoderLayers,
  int prosodyLayers,
  int acousticLayers,
  int predictorChannels,
  int synthFft,
  int synthHop,
  int noiseFft,
  int noiseHop,
  int cepCausal,
  int cepAnti,
  int noiseCep,
  float harmonicLimit,
  int maxTokens,
  int maxFrames,
  float f0Min,
  float f0Max,
  float pitchReferenceHz,
  List<String> symbols,
  List<String> speakers,
  String language
) {

  private static final Set<String> PAUSES = Set.of(
    "<pad>", "<bos>", "<eos>", "<sp>",
    ",", ".", "?", "!", ";", ":", "\"", "'", "(", ")", "—", "…");

  private static final Set<String> VOWELS = Set.of(
    "ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
    "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
    "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩",
    "a", "o", "e", "u", "ɨ");

  public TinyVocaSpec {
    symbols = List.copyOf(requireNonNull(symbols, "symbols"));
    speakers = List.copyOf(requireNonNull(speakers, "speakers"));
    requireNonNull(language, "language");
    if (sampleRate < 1 || hop < 1 || channels < 1 || heads < 1) {
      throw new IllegalArgumentException("sample rate, hop, channels, and heads must be positive");
    }
    if (channels % heads != 0 || (channels / heads) % 2 != 0) {
      throw new IllegalArgumentException("channels must split into an even head width");
    }
    if (kernel < 1 || kernel % 2 == 0) {
      throw new IllegalArgumentException("kernel must be a positive odd width");
    }
    if (encoderLayers < 1 || prosodyLayers < 1 || acousticLayers < 1) {
      throw new IllegalArgumentException("each tiny-voca stack needs at least one block");
    }
    if (!powerOfTwo(synthFft) || !powerOfTwo(noiseFft)) {
      throw new IllegalArgumentException("filter sizes must be powers of two");
    }
    if (synthHop < 1 || noiseHop < 1 || hop % synthHop != 0 || hop % noiseHop != 0) {
      throw new IllegalArgumentException("filter hops must divide the acoustic hop");
    }
    if (symbols.isEmpty() || speakers.isEmpty()) {
      throw new IllegalArgumentException("tiny-voca needs symbols and speakers");
    }
    if (!symbols.contains("<unk>") || !symbols.contains("<bos>") || !symbols.contains("<eos>")) {
      throw new IllegalArgumentException("tiny-voca inventory needs <bos>, <eos>, and <unk>");
    }
  }

  /**
   * @param json checkpoint {@code config.json}; must not be {@code null}
   * @return parsed blueprint
   * @throws IllegalArgumentException if the text is not a tiny-voca config
   * @since 1.5.1
   */
  public static TinyVocaSpec parse(final String json) {
    Map<String, Object> root = Json.parseObject(requireNonNull(json, "json"));
    Map<String, Object> hyper = Json.asObject(root.get("model"));
    if (hyper == null) {
      hyper = Json.asObject(root.get("config"));
    }
    if (hyper == null) {
      hyper = root;
    }
    List<String> symbols = symbols(root);
    List<String> speakers = speakers(hyper, root);
    int channels = integer(hyper, root, "channels", 0);
    int heads = integer(hyper, root, "heads", 0);
    int hop = integer(hyper, root, "hop", 0);
    if (channels < 1 || heads < 1 || hop < 1 || symbols.isEmpty() || speakers.isEmpty()) {
      throw new IllegalArgumentException(
        "tiny-voca config needs channels, heads, hop, a symbol inventory, and speakers");
    }
    boolean russian = symbols.stream().anyMatch(TinyVocaSpec::russianPhone);
    return new TinyVocaSpec(
      integer(hyper, root, "sample_rate", 24_000),
      hop,
      channels,
      heads,
      integer(hyper, root, "feedforward", 640),
      integer(hyper, root, "kernel", 7),
      integer(hyper, root, "encoder_layers", 4),
      integer(hyper, root, "prosody_layers", 2),
      integer(hyper, root, "acoustic_layers", 5),
      integer(hyper, root, "predictor_channels", 128),
      integer(hyper, root, "synth_fft", 512),
      integer(hyper, root, "synth_hop", 128),
      integer(hyper, root, "noise_fft", 256),
      integer(hyper, root, "noise_hop", 64),
      integer(hyper, root, "cep_causal", 80),
      integer(hyper, root, "cep_anti", 16),
      integer(hyper, root, "noise_cep", 32),
      floating(hyper, root, "harmonic_limit", 0.47f),
      integer(hyper, root, "max_tokens", 512),
      integer(hyper, root, "max_frames", 3000),
      floating(hyper, root, "f0_min", 50f),
      floating(hyper, root, "f0_max", 600f),
      floating(hyper, root, "pitch_reference_hz", 200f),
      symbols,
      speakers,
      russian ? "ru" : "en");
  }

  /**
   * @param json raw {@code config.json} text
   * @return {@code true} when the text has the tiny-voca hyperparameter shape
   * @since 1.5.1
   */
  public static boolean looksLike(final String json) {
    if (json == null || json.isBlank()) {
      return false;
    }
    String folded = json.toLowerCase(Locale.ROOT);
    boolean inventory = folded.contains("\"vocab\"") || folded.contains("\"symbols\"");
    boolean voices = folded.contains("\"speakers\"") || folded.contains("\"voices\"");
    return inventory && voices
      && folded.contains("\"hop\"")
      && folded.contains("\"channels\"")
      && folded.contains("\"sample_rate\"");
  }

  private static boolean russianPhone(final String symbol) {
    return symbol.indexOf('\u02B2') >= 0 || "ɨ".equals(symbol) || "ʂ".equals(symbol);
  }

  private static boolean powerOfTwo(final int value) {
    return value >= 2 && (value & (value - 1)) == 0;
  }

  private static List<String> symbols(final Map<String, Object> root) {
    List<String> listed = strings(root.get("symbols"));
    if (!listed.isEmpty()) {
      return listed;
    }
    Map<String, Object> vocab = Json.asObject(root.get("vocab"));
    if (vocab == null || vocab.isEmpty()) {
      throw new IllegalArgumentException("tiny-voca config needs vocab or symbols");
    }
    String[] symbols = new String[vocab.size()];
    for (var entry : vocab.entrySet()) {
      int id = Json.asInt(entry.getValue(), -1);
      if (id < 0 || id >= symbols.length || symbols[id] != null) {
        throw new IllegalArgumentException("vocab id is not a dense range for " + entry.getKey());
      }
      symbols[id] = entry.getKey();
    }
    for (String symbol : symbols) {
      if (symbol == null || symbol.isBlank()) {
        throw new IllegalArgumentException("vocab ids are not a dense 0..n-1 range");
      }
    }
    return List.of(symbols);
  }

  private static List<String> speakers(
    final Map<String, Object> hyper,
    final Map<String, Object> root
  ) {
    List<String> speakers = strings(hyper.get("speakers"));
    if (speakers.isEmpty()) {
      speakers = strings(hyper.get("voices"));
    }
    if (speakers.isEmpty()) {
      speakers = strings(root.get("speakers"));
    }
    if (speakers.isEmpty()) {
      throw new IllegalArgumentException("tiny-voca config needs speakers");
    }
    return speakers;
  }

  private static List<String> strings(final Object value) {
    List<Object> array = Json.asArray(value);
    if (array == null) {
      return List.of();
    }
    List<String> out = new ArrayList<>(array.size());
    for (Object item : array) {
      String text = Json.asString(item);
      if (text != null && !text.isBlank()) {
        out.add(text);
      }
    }
    return List.copyOf(out);
  }

  private static int integer(
    final Map<String, Object> hyper,
    final Map<String, Object> root,
    final String key,
    final int fallback
  ) {
    if (hyper.containsKey(key)) {
      return Json.asInt(hyper.get(key), fallback);
    }
    if (root.containsKey(key)) {
      return Json.asInt(root.get(key), fallback);
    }
    return fallback;
  }

  private static float floating(
    final Map<String, Object> hyper,
    final Map<String, Object> root,
    final String key,
    final float fallback
  ) {
    if (hyper.containsKey(key)) {
      return Json.asFloat(hyper.get(key), fallback);
    }
    if (root.containsKey(key)) {
      return Json.asFloat(root.get(key), fallback);
    }
    return fallback;
  }

  /**
   * @param symbol inventory token
   * @return id, or {@code -1} when {@code symbol} is absent
   * @since 1.5.1
   */
  public int symbolId(final String symbol) {
    for (int i = 0; i < this.symbols.size(); i++) {
      if (this.symbols.get(i).equals(symbol)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * @param id inventory id
   * @return {@code true} when the id is a pause (special or punctuation), not {@code <unk>}
   * @since 1.5.1
   */
  public boolean isPause(final int id) {
    return id >= 0 && id < this.symbols.size() && PAUSES.contains(this.symbols.get(id));
  }

  /**
   * @param id inventory id
   * @return {@code true} when the id is a vowel phone
   * @since 1.5.1
   */
  public boolean isVowel(final int id) {
    return id >= 0 && id < this.symbols.size() && VOWELS.contains(this.symbols.get(id));
  }

  /**
   * @param name voice name from {@link #speakers()}
   * @return speaker id, or {@code -1} when the name is unknown
   * @since 1.5.1
   */
  public int voiceIndex(final String name) {
    if (name == null || name.isBlank()) {
      return -1;
    }
    String wanted = name.strip();
    for (int i = 0; i < this.speakers.size(); i++) {
      if (this.speakers.get(i).equals(wanted)) {
        return i;
      }
    }
    for (int i = 0; i < this.speakers.size(); i++) {
      if (this.speakers.get(i).equalsIgnoreCase(wanted)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * @return {@code true} when {@link #language()} is Russian
   * @since 1.5.1
   */
  public boolean isRussian() {
    return "ru".equals(this.language);
  }

  /**
   * Harmonic cepstrum width, including the per-subframe gain bins.
   *
   * @return output features of {@code vocoder.harmonic}
   * @since 1.5.1
   */
  public int harmonicBins() {
    return this.cepCausal + this.cepAnti + this.hop / this.synthHop;
  }

  /**
   * Noise cepstrum width, including the per-subframe gain bins.
   *
   * @return output features of {@code vocoder.noise}
   * @since 1.5.1
   */
  public int noiseBins() {
    return this.noiseCep + this.hop / this.noiseHop;
  }
}
