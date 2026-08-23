package com.igormaznitsa.nanollvm.models.internal;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_WHISPER;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Conv1d;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms.LayerNorm;
import com.igormaznitsa.nanollvm.layers.ScaledDotProductAttention;
import com.igormaznitsa.nanollvm.layers.VocabParallelEmbedding;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.audio.WhisperLogMel;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.IntStream;

/**
 * OpenAI Whisper encoder-decoder ASR from Hugging Face safetensors. Greedy multilingual
 * transcribe, optional language, 30 s chunking.
 *
 * @since 1.3.0
 */
public final class WhisperForAsr implements SpeechToText {

  private static final String SOT = "<|startoftranscript|>";
  private static final String EOT = "<|endoftext|>";
  private static final String TRANSCRIBE = "<|transcribe|>";
  private static final String NO_TIMESTAMPS = "<|notimestamps|>";
  private static final String[] LANGUAGE_CODES = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
    "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
    "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"
  };

  private final Config.WhisperSpec spec;
  private final Conv1d conv1;
  private final Conv1d conv2;
  private final Tensor encoderPositions;
  private final List<EncoderBlock> encoderBlocks;
  private final LayerNorm encoderNorm;
  private final VocabParallelEmbedding tokenEmbed;
  private final Tensor decoderPositions;
  private final List<DecoderBlock> decoderBlocks;
  private final LayerNorm decoderNorm;
  private final Linear.Row logitsHead;

  public WhisperForAsr(final Config.HfConfig config, final WeightBag weights) {
    requireNonNull(config, "config");
    requireNonNull(weights, "weights");
    this.spec = requireNonNull(config.whisper(), "whisper spec");
    float eps = config.rmsNormEps();
    this.conv1 = new Conv1d(
      tensor(weights, "model.encoder.conv1.weight"),
      tensor(weights, "model.encoder.conv1.bias"),
      1,
      1);
    this.conv2 = new Conv1d(
      tensor(weights, "model.encoder.conv2.weight"),
      tensor(weights, "model.encoder.conv2.bias"),
      2,
      1);
    this.encoderPositions = tensor(weights, "model.encoder.embed_positions.weight");
    this.encoderBlocks = IntStream.range(0, config.numHiddenLayers())
      .mapToObj(i -> EncoderBlock.load(config, weights, i, eps))
      .toList();
    this.encoderNorm = layerNorm(weights, "model.encoder.layer_norm", eps);
    this.tokenEmbed = new VocabParallelEmbedding(
      tensor(weights, "model.decoder.embed_tokens.weight"));
    this.decoderPositions = tensor(weights, "model.decoder.embed_positions.weight");
    this.decoderBlocks = IntStream.range(0, this.spec.decoderLayers())
      .mapToObj(i -> DecoderBlock.load(config, weights, i, eps))
      .toList();
    this.decoderNorm = layerNorm(weights, "model.decoder.layer_norm", eps);
    this.logitsHead = weights.find("proj_out.weight")
      .or(() -> weights.find("model.decoder.embed_tokens.weight"))
      .map(weight -> new Linear.Row(weight, null))
      .orElseGet(() -> new Linear.Row(this.tokenEmbed.weight(), null));
  }

  private static Tensor tensor(final WeightBag weights, final String name) {
    return weights.find(name)
      .or(() -> name.startsWith("model.")
        ? weights.find(name.substring("model.".length()))
        : weights.find("model." + name))
      .orElseThrow(() -> new IllegalArgumentException("missing weight: " + name));
  }

  private static LayerNorm layerNorm(
    final WeightBag weights,
    final String prefix,
    final float eps
  ) {
    return new LayerNorm(
      tensor(weights, prefix + ".weight"),
      tensor(weights, prefix + ".bias"),
      eps);
  }

  private static Linear.Row row(
    final WeightBag weights,
    final String weightName,
    final String biasName
  ) {
    Tensor bias = weights.find(biasName)
      .or(() -> biasName.startsWith("model.")
        ? weights.find(biasName.substring("model.".length()))
        : weights.find("model." + biasName))
      .orElse(null);
    return new Linear.Row(tensor(weights, weightName), bias);
  }

  private static Tensor transposeChannels(final Tensor channelsLength) {
    int channels = channelsLength.size(0);
    int length = channelsLength.size(1);
    float[] src = channelsLength.data();
    int off = channelsLength.offset();
    float[] dst = new float[length * channels];
    for (int t = 0; t < length; t++) {
      for (int c = 0; c < channels; c++) {
        dst[t * channels + c] = src[off + c * length + t];
      }
    }
    return Tensor.of(dst, length, channels);
  }

  private static Tensor firstRows(final Tensor table, final int rows) {
    int dim = table.size(1);
    if (rows > table.size(0)) {
      throw new IllegalArgumentException(
        "sequence length %d exceeds position table %d".formatted(rows, table.size(0)));
    }
    float[] data = new float[rows * dim];
    System.arraycopy(table.data(), table.offset(), data, 0, data.length);
    return Tensor.of(data, rows, dim);
  }

  private static Tensor lastRow(final Tensor hidden) {
    int dim = hidden.size(1);
    int seq = hidden.size(0);
    float[] row = new float[dim];
    System.arraycopy(hidden.data(), hidden.offset() + (seq - 1) * dim, row, 0, dim);
    return Tensor.of(row, 1, dim);
  }

  private static Tensor idsAsFloat(final List<Integer> tokens) {
    float[] ids = new float[tokens.size()];
    for (int i = 0; i < tokens.size(); i++) {
      ids[i] = tokens.get(i);
    }
    return Tensor.of(ids, tokens.size());
  }

  private static int requireToken(final Tokenizer tokenizer, final String token) {
    return tokenizer.tokenId(token)
      .orElseThrow(() -> new IllegalStateException("tokenizer missing " + token));
  }

  private static Integer resolveLanguageId(final Tokenizer tokenizer, final String language) {
    if (language == null || language.isBlank()) {
      return null;
    }
    String code = language.strip().toLowerCase(Locale.ROOT);
    if (code.length() > 2 && code.charAt(2) == '-') {
      code = code.substring(0, 2);
    }
    return tokenizer.tokenId("<|" + code + "|>")
      .orElseThrow(() -> new IllegalArgumentException("unknown Whisper language: " + language));
  }

  private static boolean[] suppressMask(
    final Tokenizer tokenizer,
    final int vocab,
    final int eot
  ) {
    boolean[] suppress = new boolean[vocab];
    for (int id = 0; id < vocab; id++) {
      if (id == eot) {
        continue;
      }
      String piece = tokenizer.decode(List.of(id), false);
      suppress[id] = piece.startsWith("<|") && piece.endsWith("|>");
    }
    return suppress;
  }

  private static boolean allSuppressed(final boolean[] suppress) {
    for (boolean bit : suppress) {
      if (!bit) {
        return false;
      }
    }
    return true;
  }

  private static int argmax(final Tensor logits, final boolean[] suppress) {
    float[] data = logits.data();
    int off = logits.offset();
    int n = logits.size(logits.shape().length - 1);
    int best = 0;
    float bestValue = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < n; i++) {
      if (i < suppress.length && suppress[i]) {
        continue;
      }
      float value = data[off + i];
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    return best;
  }

  @Override
  public String architectureName() {
    return ARCH_WHISPER;
  }

  @Override
  public String transcribe(
    final float[] pcm,
    final int sampleRate,
    final String language,
    final Tokenizer tokenizer
  ) {
    requireNonNull(pcm, "pcm");
    requireNonNull(tokenizer, "tokenizer");
    if (sampleRate < 1) {
      throw new IllegalArgumentException("sampleRate must be >= 1");
    }
    float[] at16k = sampleRate == WhisperLogMel.SAMPLE_RATE
      ? pcm
      : WhisperLogMel.resampleLinear(pcm, sampleRate, WhisperLogMel.SAMPLE_RATE);
    if (at16k.length == 0) {
      return "";
    }
    int sot = requireToken(tokenizer, SOT);
    int eot = requireToken(tokenizer, EOT);
    int transcribe = requireToken(tokenizer, TRANSCRIBE);
    int noTimestamps = requireToken(tokenizer, NO_TIMESTAMPS);
    Integer langId = resolveLanguageId(tokenizer, language);
    StringBuilder text = new StringBuilder();
    for (int origin = 0; origin < at16k.length; origin += WhisperLogMel.CHUNK_SAMPLES) {
      int end = Math.min(origin + WhisperLogMel.CHUNK_SAMPLES, at16k.length);
      String piece = this.transcribeChunk(
        Arrays.copyOfRange(at16k, origin, end),
        tokenizer,
        sot,
        eot,
        transcribe,
        noTimestamps,
        langId);
      if (!piece.isBlank()) {
        if (!text.isEmpty()) {
          text.append(' ');
        }
        text.append(piece.strip());
      }
    }
    return text.toString();
  }

  private String transcribeChunk(
    final float[] chunk,
    final Tokenizer tokenizer,
    final int sot,
    final int eot,
    final int transcribe,
    final int noTimestamps,
    final Integer langId
  ) {
    Context context = new Context();
    context.bindMatmul(MatmulRuntime.sequential());
    Tensor encoderHidden = this.encode(
      WhisperLogMel.features(chunk, WhisperLogMel.SAMPLE_RATE), context);
    List<CrossCache> memory = this.decoderBlocks.stream()
      .map(block -> block.crossCache(encoderHidden, context))
      .toList();
    List<Integer> tokens = new ArrayList<>();
    tokens.add(sot);
    int language = langId != null
      ? langId
      : this.detectLanguage(tokens, memory, context, tokenizer, eot);
    tokens.add(language);
    tokens.add(transcribe);
    tokens.add(noTimestamps);
    boolean[] suppress = suppressMask(tokenizer, this.tokenEmbed.numEmbeddings(), eot);
    int maxTokens = this.spec.maxTargetPositions();
    while (tokens.size() < maxTokens) {
      int next = this.argmaxLogit(tokens, memory, context, suppress);
      if (next == eot) {
        break;
      }
      tokens.add(next);
    }
    return tokenizer.decode(tokens.subList(4, tokens.size()), true).strip();
  }

  private Tensor encode(final Tensor logMel, final Context context) {
    Tensor hidden = Ops.gelu(this.conv1.forward(logMel));
    hidden = Ops.gelu(this.conv2.forward(hidden));
    hidden = transposeChannels(hidden);
    hidden = Ops.add(hidden, firstRows(this.encoderPositions, hidden.size(0)));
    for (EncoderBlock block : this.encoderBlocks) {
      hidden = block.forward(hidden, context);
    }
    return this.encoderNorm.forward(hidden);
  }

  private int detectLanguage(
    final List<Integer> prefix,
    final List<CrossCache> memory,
    final Context context,
    final Tokenizer tokenizer,
    final int eot
  ) {
    boolean[] suppress = new boolean[this.tokenEmbed.numEmbeddings()];
    Arrays.fill(suppress, true);
    for (String code : LANGUAGE_CODES) {
      tokenizer.tokenId("<|" + code + "|>").ifPresent(id -> suppress[id] = false);
    }
    if (eot >= 0 && eot < suppress.length) {
      suppress[eot] = true;
    }
    if (allSuppressed(suppress)) {
      return requireToken(tokenizer, "<|en|>");
    }
    return this.argmaxLogit(prefix, memory, context, suppress);
  }

  private int argmaxLogit(
    final List<Integer> tokens,
    final List<CrossCache> memory,
    final Context context,
    final boolean[] suppress
  ) {
    Tensor hidden = Ops.add(
      this.tokenEmbed.forward(idsAsFloat(tokens), context),
      firstRows(this.decoderPositions, tokens.size()));
    for (int i = 0; i < this.decoderBlocks.size(); i++) {
      hidden = this.decoderBlocks.get(i).forward(hidden, memory.get(i), context);
    }
    hidden = this.decoderNorm.forward(hidden);
    return argmax(this.logitsHead.forward(lastRow(hidden), context), suppress);
  }

  private record EncoderBlock(
    LayerNorm attnNorm,
    Linear.Row qProj,
    Linear.Row kProj,
    Linear.Row vProj,
    Linear.Row oProj,
    ScaledDotProductAttention attn,
    LayerNorm mlpNorm,
    Linear.Row fc1,
    Linear.Row fc2
  ) {
    static EncoderBlock load(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layer,
      final float eps
    ) {
      String p = "model.encoder.layers." + layer + ".";
      float scale = (float) Math.pow(config.headDim(), -0.5);
      return new EncoderBlock(
        layerNorm(weights, p + "self_attn_layer_norm", eps),
        row(weights, p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias"),
        row(weights, p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias"),
        row(weights, p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias"),
        row(weights, p + "self_attn.out_proj.weight", p + "self_attn.out_proj.bias"),
        new ScaledDotProductAttention(
          config.numAttentionHeads(), config.headDim(), scale, false),
        layerNorm(weights, p + "final_layer_norm", eps),
        row(weights, p + "fc1.weight", p + "fc1.bias"),
        row(weights, p + "fc2.weight", p + "fc2.bias"));
    }

    Tensor forward(final Tensor hidden, final Context context) {
      Tensor normed = this.attnNorm.forward(hidden);
      Tensor attnOut = this.oProj.forward(
        this.attn.forward(
          this.qProj.forward(normed, context),
          this.kProj.forward(normed, context),
          this.vProj.forward(normed, context),
          context),
        context);
      Tensor residual = Ops.add(hidden, attnOut);
      Tensor mlpIn = this.mlpNorm.forward(residual);
      Tensor mlpOut = this.fc2.forward(Ops.gelu(this.fc1.forward(mlpIn, context)), context);
      return Ops.add(residual, mlpOut);
    }
  }

  private record DecoderBlock(
    LayerNorm selfNorm,
    Linear.Row selfQ,
    Linear.Row selfK,
    Linear.Row selfV,
    Linear.Row selfO,
    ScaledDotProductAttention selfAttn,
    LayerNorm crossNorm,
    Linear.Row crossQ,
    Linear.Row crossK,
    Linear.Row crossV,
    Linear.Row crossO,
    ScaledDotProductAttention crossAttn,
    LayerNorm mlpNorm,
    Linear.Row fc1,
    Linear.Row fc2
  ) {
    static DecoderBlock load(
      final Config.HfConfig config,
      final WeightBag weights,
      final int layer,
      final float eps
    ) {
      String p = "model.decoder.layers." + layer + ".";
      float scale = (float) Math.pow(config.headDim(), -0.5);
      int heads = config.numAttentionHeads();
      int headDim = config.headDim();
      return new DecoderBlock(
        layerNorm(weights, p + "self_attn_layer_norm", eps),
        row(weights, p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias"),
        row(weights, p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias"),
        row(weights, p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias"),
        row(weights, p + "self_attn.out_proj.weight", p + "self_attn.out_proj.bias"),
        new ScaledDotProductAttention(heads, headDim, scale, true),
        layerNorm(weights, p + "encoder_attn_layer_norm", eps),
        row(weights, p + "encoder_attn.q_proj.weight", p + "encoder_attn.q_proj.bias"),
        row(weights, p + "encoder_attn.k_proj.weight", p + "encoder_attn.k_proj.bias"),
        row(weights, p + "encoder_attn.v_proj.weight", p + "encoder_attn.v_proj.bias"),
        row(weights, p + "encoder_attn.out_proj.weight", p + "encoder_attn.out_proj.bias"),
        new ScaledDotProductAttention(heads, headDim, scale, false),
        layerNorm(weights, p + "final_layer_norm", eps),
        row(weights, p + "fc1.weight", p + "fc1.bias"),
        row(weights, p + "fc2.weight", p + "fc2.bias"));
    }

    CrossCache crossCache(final Tensor encoderHidden, final Context context) {
      return new CrossCache(
        this.crossK.forward(encoderHidden, context),
        this.crossV.forward(encoderHidden, context));
    }

    Tensor forward(final Tensor hidden, final CrossCache memory, final Context context) {
      Tensor selfIn = this.selfNorm.forward(hidden);
      Tensor selfOut = this.selfO.forward(
        this.selfAttn.forward(
          this.selfQ.forward(selfIn, context),
          this.selfK.forward(selfIn, context),
          this.selfV.forward(selfIn, context),
          context),
        context);
      Tensor afterSelf = Ops.add(hidden, selfOut);
      Tensor crossIn = this.crossNorm.forward(afterSelf);
      Tensor crossOut = this.crossO.forward(
        this.crossAttn.forward(
          this.crossQ.forward(crossIn, context),
          memory.key(),
          memory.value(),
          context),
        context);
      Tensor afterCross = Ops.add(afterSelf, crossOut);
      Tensor mlpIn = this.mlpNorm.forward(afterCross);
      Tensor mlpOut = this.fc2.forward(Ops.gelu(this.fc1.forward(mlpIn, context)), context);
      return Ops.add(afterCross, mlpOut);
    }
  }

  private record CrossCache(Tensor key, Tensor value) {
  }
}
