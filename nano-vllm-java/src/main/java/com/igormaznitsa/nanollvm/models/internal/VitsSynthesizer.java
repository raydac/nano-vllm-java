package com.igormaznitsa.nanollvm.models.internal;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Conv1d;
import com.igormaznitsa.nanollvm.layers.ConvTranspose1d;
import com.igormaznitsa.nanollvm.layers.ScaledDotProductAttention;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Ops;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tensor.VectorMath;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/**
 * Piper/VITS {@code SynthesizerTrn.infer()} on ONNX initializers (no ORT).
 *
 * @since 1.3.0
 */
final class VitsSynthesizer {

  private static final float LEAKY = 0.1f;
  private static final float LN_EPS = 1e-5f;
  private static final float SPLINE_MIN_BIN = 1e-3f;
  private static final float SPLINE_MIN_DERIVATIVE = 1e-3f;
  private static final float SPLINE_TAIL_BOUND = 5f;

  private final WeightBag weights;
  private final Tensor embedding;
  private final int hiddenChannels;
  private final int interChannels;
  private final List<EncoderLayer> encoder;
  private final Conv1d encoderProj;
  private final Duration duration;
  private final List<Coupling> couplings;
  private final Generator generator;
  private boolean identityMask;

  VitsSynthesizer(final Config.HfConfig config, final WeightBag weights) {
    this.weights = requireNonNull(weights, "weights");
    requireNonNull(config, "config");
    this.embedding = this.require("enc_p.emb.weight");
    this.hiddenChannels = this.embedding.size(1);
    this.encoder = this.loadEncoder();
    this.encoderProj = this.conv("enc_p.proj", 1, 0);
    this.interChannels = this.encoderProj.outChannels() / 2;
    this.duration = Duration.load(this);
    this.couplings = this.loadCouplings();
    this.generator = Generator.load(this);
  }

  static Tensor leakyRelu(final Tensor x) {
    Tensor y = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] yd = y.data();
    int off = x.offset();
    int n = x.numel();
    for (int i = 0; i < n; i++) {
      float v = xd[off + i];
      yd[i] = v < 0f ? LEAKY * v : v;
    }
    return y;
  }

  static Tensor gelu(final Tensor x) {
    return Ops.gelu(x);
  }

  static Tensor relu(final Tensor x) {
    Tensor y = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] yd = y.data();
    int off = x.offset();
    int n = x.numel();
    for (int i = 0; i < n; i++) {
      yd[i] = Math.max(0f, xd[off + i]);
    }
    return y;
  }

  static Tensor tanh(final Tensor x) {
    Tensor y = Tensor.zeros(x.shape());
    float[] xd = x.data();
    float[] yd = y.data();
    int off = x.offset();
    int n = x.numel();
    for (int i = 0; i < n; i++) {
      yd[i] = (float) Math.tanh(xd[off + i]);
    }
    return y;
  }

  static int intPow(final int base, final int exp) {
    int value = 1;
    for (int i = 0; i < exp; i++) {
      value = Math.multiplyExact(value, base);
    }
    return value;
  }

  static float inverseLinearTailRationalQuadratic(
    final float input,
    final float[] unnormalizedWidths,
    final float[] unnormalizedHeights,
    final float[] unnormalizedDerivatives,
    final float tailBound
  ) {
    if (!(input >= -tailBound && input <= tailBound)) {
      return input;
    }
    int bins = unnormalizedWidths.length;
    double[] unnormD = new double[bins + 1];
    double tailConstant = Math.log(Math.expm1(1.0 - SPLINE_MIN_DERIVATIVE));
    unnormD[0] = tailConstant;
    unnormD[bins] = tailConstant;
    for (int i = 0; i < bins - 1; i++) {
      unnormD[i + 1] = unnormalizedDerivatives[i];
    }
    return (float) rationalQuadraticInverse(
      input,
      softmaxBins(unnormalizedWidths, SPLINE_MIN_BIN),
      softmaxBins(unnormalizedHeights, SPLINE_MIN_BIN),
      unnormD,
      -tailBound,
      tailBound);
  }

  private static double[] softmaxBins(final float[] unnormalized, final float minBin) {
    int bins = unnormalized.length;
    double max = Double.NEGATIVE_INFINITY;
    for (float value : unnormalized) {
      max = Math.max(max, value);
    }
    double[] exp = new double[bins];
    double sum = 0.0;
    for (int i = 0; i < bins; i++) {
      exp[i] = Math.exp(unnormalized[i] - max);
      sum += exp[i];
    }
    double leftover = 1.0 - minBin * bins;
    double[] binsOut = new double[bins];
    for (int i = 0; i < bins; i++) {
      binsOut[i] = minBin + leftover * (exp[i] / sum);
    }
    return binsOut;
  }

  private static double rationalQuadraticInverse(
    final double input,
    final double[] widths,
    final double[] heights,
    final double[] unnormalizedDerivatives,
    final double left,
    final double right
  ) {
    int bins = widths.length;
    double[] cumWidths = new double[bins + 1];
    double[] cumHeights = new double[bins + 1];
    cumWidths[0] = left;
    cumHeights[0] = left;
    double widthAcc = 0.0;
    double heightAcc = 0.0;
    double widthSpan = right - left;
    for (int i = 0; i < bins; i++) {
      widthAcc += widths[i];
      heightAcc += heights[i];
      cumWidths[i + 1] = left + widthSpan * widthAcc;
      cumHeights[i + 1] = left + widthSpan * heightAcc;
    }
    cumWidths[bins] = right;
    cumHeights[bins] = right;
    for (int i = 0; i < bins; i++) {
      widths[i] = cumWidths[i + 1] - cumWidths[i];
      heights[i] = cumHeights[i + 1] - cumHeights[i];
    }
    double[] derivatives = new double[bins + 1];
    for (int i = 0; i <= bins; i++) {
      derivatives[i] = SPLINE_MIN_DERIVATIVE + softplus(unnormalizedDerivatives[i]);
    }
    int bin = 0;
    for (int i = 0; i < bins; i++) {
      if (input >= cumHeights[i + 1]) {
        bin = i + 1;
      }
    }
    bin = Math.min(bin, bins - 1);
    double inputCumWidth = cumWidths[bin];
    double inputBinWidth = widths[bin];
    double inputCumHeight = cumHeights[bin];
    double inputHeight = heights[bin];
    double delta = inputHeight / inputBinWidth;
    double inputDelta = delta;
    double inputDerivative = derivatives[bin];
    double inputDerivativePlusOne = derivatives[bin + 1];
    double a =
      (input - inputCumHeight) * (inputDerivative + inputDerivativePlusOne - 2.0 * inputDelta)
        + inputHeight * (inputDelta - inputDerivative);
    double b = inputHeight * inputDerivative
      - (input - inputCumHeight) * (inputDerivative + inputDerivativePlusOne - 2.0 * inputDelta);
    double c = -inputDelta * (input - inputCumHeight);
    if (Math.abs(a) < 1e-12) {
      return inputCumWidth + (-c / b) * inputBinWidth;
    }
    double discriminant = Math.max(0.0, b * b - 4.0 * a * c);
    double root = (2.0 * c) / (-b - Math.sqrt(discriminant));
    return inputCumWidth + root * inputBinWidth;
  }

  private static double softplus(final double x) {
    if (x > 20.0) {
      return x;
    }
    return Math.log1p(Math.exp(x));
  }

  /**
   * Reverse VITS: duration, flow, HiFi-GAN vocoder. Binds {@code runtime} for the call.
   *
   * @param phonemeIds  G2P ids; must not be empty
   * @param noiseScale  posterior noise
   * @param lengthScale duration stretch
   * @param noiseW      duration-predictor noise
   * @param runtime     dense kernel runtime; must not be {@code null}
   * @param random      duration / prior noise source; must not be {@code null}
   * @return mono PCM in {@code [-1, 1]}
   */
  float[] infer(
    final int[] phonemeIds,
    final float noiseScale,
    final float lengthScale,
    final float noiseW,
    final MatmulRuntime runtime,
    final Random random
  ) {
    requireNonNull(phonemeIds, "phonemeIds");
    requireNonNull(runtime, "runtime");
    requireNonNull(random, "random");
    if (phonemeIds.length == 0) {
      throw new IllegalArgumentException("phonemeIds must not be empty");
    }
    Context context = new Context();
    context.bindMatmul(runtime);
    Kernels kernels = new Kernels(runtime, context);
    try {
      return this.inferUnlocked(phonemeIds, noiseScale, lengthScale, noiseW, kernels, random);
    } finally {
      context.clear();
    }
  }

  private float[] inferUnlocked(
    final int[] phonemeIds,
    final float noiseScale,
    final float lengthScale,
    final float noiseW,
    final Kernels kernels,
    final Random random
  ) {
    this.identityMask = true;
    try {
      Encoded encoded = this.encode(phonemeIds, kernels);
      Tensor logw =
        this.duration.logw(encoded.hidden(), encoded.mask(), noiseW, random, kernels);
      int[] durations = this.ceilDurations(logw, encoded.mask(), lengthScale);
      int yLen = 0;
      for (int duration : durations) {
        yLen += duration;
      }
      yLen = Math.max(yLen, 1);
      Tensor mP = this.expand(encoded.mean(), durations, yLen);
      Tensor logsP = this.expand(encoded.logStd(), durations, yLen);
      Tensor yMask = this.onesMask(yLen);
      Tensor z = this.samplePrior(mP, logsP, yMask, noiseScale, random);
      for (int i = this.couplings.size() - 1; i >= 0; i--) {
        z = this.flipChannels(z);
        z = this.couplings.get(i).reverse(z, yMask, kernels);
      }
      return this.toMono(this.generator.forward(z, kernels));
    } finally {
      this.identityMask = false;
    }
  }

  private Encoded encode(final int[] ids, final Kernels kernels) {
    int time = ids.length;
    Tensor hidden = Tensor.zeros(this.hiddenChannels, time);
    float scale = (float) Math.sqrt(this.hiddenChannels);
    float[] emb = this.embedding.data();
    int embOff = this.embedding.offset();
    float[] hd = hidden.data();
    for (int t = 0; t < time; t++) {
      int row = Math.clamp(ids[t], 0, this.embedding.size(0) - 1);
      int src = embOff + row * this.hiddenChannels;
      for (int c = 0; c < this.hiddenChannels; c++) {
        hd[c * time + t] = emb[src + c] * scale;
      }
    }
    Tensor mask = this.onesMask(time);
    hidden = this.mulMask(hidden, mask);
    for (EncoderLayer layer : this.encoder) {
      hidden = layer.forward(hidden, mask, kernels);
    }
    Tensor stats = this.mulMask(kernels.apply(this.encoderProj, hidden), mask);
    Tensor mean = this.channelSlice(stats, 0, this.interChannels);
    Tensor logs = this.channelSlice(stats, this.interChannels, this.interChannels);
    return new Encoded(hidden, mean, logs, mask);
  }

  private List<EncoderLayer> loadEncoder() {
    List<EncoderLayer> layers = new ArrayList<>();
    for (int i = 0; this.weights.has("enc_p.encoder.attn_layers." + i + ".conv_q.weight"); i++) {
      layers.add(EncoderLayer.load(this, i));
    }
    if (layers.isEmpty()) {
      throw new IllegalStateException(
        "piper text encoder layers were not found in ONNX initializers");
    }
    return List.copyOf(layers);
  }

  private List<Coupling> loadCouplings() {
    List<Coupling> flows = new ArrayList<>();
    for (int i = 0; i < 64; i++) {
      if (this.weights.has("flow.flows." + i + ".pre.weight")) {
        flows.add(Coupling.load(this, "flow.flows." + i));
      }
    }
    if (flows.isEmpty()) {
      throw new IllegalStateException(
        "piper residual couplings were not found in ONNX initializers");
    }
    return List.copyOf(flows);
  }

  private int[] ceilDurations(final Tensor logw, final Tensor mask, final float lengthScale) {
    int time = logw.size(1);
    int[] out = new int[time];
    float[] d = logw.data();
    int off = logw.offset();
    float[] md = mask.data();
    for (int t = 0; t < time; t++) {
      if (md[t] <= 0f) {
        out[t] = 0;
        continue;
      }
      out[t] = Math.max(1, (int) Math.ceil(Math.exp(d[off + t]) * lengthScale));
    }
    return out;
  }

  private Tensor expand(final Tensor src, final int[] durations, final int yLen) {
    int channels = src.size(0);
    Tensor out = Tensor.zeros(channels, yLen);
    float[] s = src.data();
    int sOff = src.offset();
    float[] d = out.data();
    int srcTime = src.size(1);
    int dst = 0;
    for (int t = 0; t < srcTime && dst < yLen; t++) {
      int span = durations[t];
      for (int k = 0; k < span && dst < yLen; k++, dst++) {
        for (int c = 0; c < channels; c++) {
          d[c * yLen + dst] = s[sOff + c * srcTime + t];
        }
      }
    }
    return out;
  }

  private Tensor samplePrior(
    final Tensor mean,
    final Tensor logs,
    final Tensor mask,
    final float noiseScale,
    final Random random
  ) {
    int channels = mean.size(0);
    int time = mean.size(1);
    Tensor z = Tensor.zeros(channels, time);
    float[] m = mean.data();
    float[] l = logs.data();
    float[] md = mask.data();
    float[] d = z.data();
    int mOff = mean.offset();
    int lOff = logs.offset();
    int maskOff = mask.offset();
    for (int c = 0; c < channels; c++) {
      for (int t = 0; t < time; t++) {
        int i = c * time + t;
        float n = (float) random.nextGaussian() * noiseScale;
        float maskValue = this.identityMask ? 1f : md[maskOff + t];
        d[i] = (m[mOff + i] + n * (float) Math.exp(l[lOff + i])) * maskValue;
      }
    }
    return z;
  }

  Tensor require(final String name) {
    return this.weights.require(name);
  }

  Optional<Tensor> find(final String name) {
    return this.weights.find(name);
  }

  Conv1d conv(final String prefix, final int stride, final int padding) {
    return this.conv(prefix, stride, padding, 1, 1);
  }

  Conv1d conv(
    final String prefix,
    final int stride,
    final int padding,
    final int dilation,
    final int groups
  ) {
    Tensor weight = this.require(prefix + ".weight");
    Tensor bias = this.find(prefix + ".bias").orElse(null);
    ConvLayout layout = this.weights.convLayout(prefix + ".weight").orElse(null);
    if (layout != null && layout.isTransposed()) {
      throw new IllegalStateException(prefix + " is a ConvTranspose weight");
    }
    return new Conv1d(
      weight,
      bias,
      layout == null ? stride : layout.strideOr(stride),
      layout == null ? padding : layout.paddingOr(padding),
      layout == null ? dilation : layout.dilationOr(dilation),
      layout == null ? groups : layout.groupsOr(groups));
  }

  ConvTranspose1d convTranspose(final String prefix, final int stride, final int padding) {
    Tensor weight = this.require(prefix + ".weight");
    Tensor bias = this.find(prefix + ".bias").orElse(null);
    ConvLayout layout = this.weights.convLayout(prefix + ".weight").orElse(null);
    return new ConvTranspose1d(
      weight,
      bias,
      layout == null ? stride : layout.strideOr(stride),
      layout == null ? padding : layout.paddingOr(padding),
      layout == null ? 0 : layout.outputPaddingOr(0),
      layout == null ? 1 : layout.dilationOr(1),
      layout == null ? 1 : layout.groupsOr(1));
  }

  Tensor onesMask(final int time) {
    return Tensor.ones(1, time);
  }

  Tensor mulMask(final Tensor x, final Tensor mask) {
    if (this.identityMask) {
      return x;
    }
    int channels = x.size(0);
    int time = x.size(1);
    Tensor y = Tensor.zeros(channels, time);
    float[] xd = x.data();
    float[] md = mask.data();
    float[] yd = y.data();
    int xOff = x.offset();
    int mOff = mask.offset();
    for (int c = 0; c < channels; c++) {
      for (int t = 0; t < time; t++) {
        yd[c * time + t] = xd[xOff + c * time + t] * md[mOff + t];
      }
    }
    return y;
  }

  Tensor add(final Tensor a, final Tensor b) {
    return Ops.add(a, b);
  }

  Tensor channelNorm(final Tensor x, final String gammaName, final String betaName) {
    int channels = x.size(0);
    int time = x.size(1);
    Tensor gamma = this.require(gammaName);
    Tensor beta = this.require(betaName);
    if (gamma.numel() != channels || beta.numel() != channels) {
      throw new IllegalArgumentException("channelNorm gamma/beta length must equal channels");
    }
    Tensor y = Tensor.zeros(channels, time);
    float[] xd = x.data();
    float[] yd = y.data();
    float[] gd = gamma.data();
    float[] bd = beta.data();
    int xOff = x.offset();
    int gOff = gamma.offset();
    int bOff = beta.offset();
    for (int t = 0; t < time; t++) {
      float sum = 0f;
      for (int c = 0; c < channels; c++) {
        sum += xd[xOff + c * time + t];
      }
      float mean = sum / channels;
      float var = 0f;
      for (int c = 0; c < channels; c++) {
        float d = xd[xOff + c * time + t] - mean;
        var += d * d;
      }
      float inv = (float) (1.0 / Math.sqrt(var / channels + LN_EPS));
      for (int c = 0; c < channels; c++) {
        yd[c * time + t] =
          (xd[xOff + c * time + t] - mean) * inv * gd[gOff + c] + bd[bOff + c];
      }
    }
    return y;
  }

  Tensor channelsToTokens(final Tensor x) {
    int channels = x.size(0);
    int time = x.size(1);
    Tensor y = Tensor.zeros(time, channels);
    float[] xd = x.data();
    float[] yd = y.data();
    int off = x.offset();
    for (int t = 0; t < time; t++) {
      for (int c = 0; c < channels; c++) {
        yd[t * channels + c] = xd[off + c * time + t];
      }
    }
    return y;
  }

  Tensor tokensToChannels(final Tensor x) {
    int time = x.size(0);
    int channels = x.size(1);
    Tensor y = Tensor.zeros(channels, time);
    float[] xd = x.data();
    float[] yd = y.data();
    int off = x.offset();
    for (int t = 0; t < time; t++) {
      for (int c = 0; c < channels; c++) {
        yd[c * time + t] = xd[off + t * channels + c];
      }
    }
    return y;
  }

  Tensor channelSlice(final Tensor x, final int start, final int length) {
    int time = x.size(1);
    Tensor y = Tensor.zeros(length, time);
    float[] xd = x.data();
    float[] yd = y.data();
    int off = x.offset();
    int srcChannels = x.size(0);
    for (int c = 0; c < length; c++) {
      System.arraycopy(xd, off + (start + c) * time, yd, c * time, time);
    }
    if (srcChannels < start + length) {
      throw new IllegalArgumentException("channel slice out of range");
    }
    return y;
  }

  Tensor flipChannels(final Tensor x) {
    int channels = x.size(0);
    int time = x.size(1);
    Tensor y = Tensor.zeros(channels, time);
    float[] xd = x.data();
    float[] yd = y.data();
    int off = x.offset();
    for (int c = 0; c < channels; c++) {
      System.arraycopy(xd, off + (channels - 1 - c) * time, yd, c * time, time);
    }
    return y;
  }

  Tensor concatChannels(final Tensor a, final Tensor b) {
    int time = a.size(1);
    int ca = a.size(0);
    int cb = b.size(0);
    Tensor y = Tensor.zeros(ca + cb, time);
    float[] yd = y.data();
    System.arraycopy(a.data(), a.offset(), yd, 0, ca * time);
    System.arraycopy(b.data(), b.offset(), yd, ca * time, cb * time);
    return y;
  }

  private float[] toMono(final Tensor audio) {
    int time = audio.size(audio.shape().length - 1);
    float[] src = audio.data();
    int off = audio.offset();
    float[] pcm = new float[time];
    System.arraycopy(src, off, pcm, 0, time);
    return pcm;
  }

  private record Encoded(Tensor hidden, Tensor mean, Tensor logStd, Tensor mask) {
  }

  private record Kernels(MatmulRuntime matmul, Context context) {
    Tensor apply(final Conv1d layer, final Tensor input) {
      return layer.forward(input, this.matmul);
    }

    Tensor apply(final ConvTranspose1d layer, final Tensor input) {
      return layer.forward(input, this.matmul);
    }
  }

  private record EncoderLayer(Conv1d query, Conv1d key, Conv1d value, Conv1d out,
                              ScaledDotProductAttention attention, Tensor relKey, Tensor relValue,
                              int heads, int headDim, VitsSynthesizer owner, int index, Conv1d ffn1,
                              Conv1d ffn2) {
    private EncoderLayer(
      final VitsSynthesizer owner,
      final int index,
      final Conv1d query,
      final Conv1d key,
      final Conv1d value,
      final Conv1d out,
      final ScaledDotProductAttention attention,
      final Tensor relKey,
      final Tensor relValue,
      final int heads,
      final int headDim,
      final Conv1d ffn1,
      final Conv1d ffn2
    ) {
      this(query, key, value, out, attention, relKey, relValue, heads, headDim, owner, index, ffn1,
        ffn2);
    }

    static EncoderLayer load(final VitsSynthesizer owner, final int index) {
      String attn = "enc_p.encoder.attn_layers." + index;
      Conv1d query = owner.conv(attn + ".conv_q", 1, 0);
      int hidden = query.outChannels();
      Tensor relKey = owner.find(attn + ".emb_rel_k").orElse(null);
      Tensor relValue = owner.find(attn + ".emb_rel_v").orElse(null);
      int headDim = relKey != null ? relKey.size(relKey.shape().length - 1) : hidden / 2;
      int heads = hidden / headDim;
      if (heads < 1 || hidden % headDim != 0) {
        heads = hidden % 2 == 0 ? 2 : 1;
        while (heads > 1 && hidden % heads != 0) {
          heads--;
        }
        headDim = hidden / heads;
      }
      int ffnPad = 1;
      return new EncoderLayer(
        owner,
        index,
        query,
        owner.conv(attn + ".conv_k", 1, 0),
        owner.conv(attn + ".conv_v", 1, 0),
        owner.conv(attn + ".conv_o", 1, 0),
        new ScaledDotProductAttention(heads, headDim, 1f / (float) Math.sqrt(headDim), false),
        relKey,
        relValue,
        heads,
        headDim,
        owner.conv("enc_p.encoder.ffn_layers." + index + ".conv_1", 1, ffnPad),
        owner.conv("enc_p.encoder.ffn_layers." + index + ".conv_2", 1, ffnPad));
    }

    Tensor forward(final Tensor x, final Tensor mask, final Kernels kernels) {
      Tensor q = this.owner.channelsToTokens(kernels.apply(this.query, x));
      Tensor k = this.owner.channelsToTokens(kernels.apply(this.key, x));
      Tensor v = this.owner.channelsToTokens(kernels.apply(this.value, x));
      Tensor attendedTokens = this.relKey == null || this.relValue == null
        ? this.attention.forward(q, k, v, kernels.context())
        : this.attendRelative(q, k, v);
      Tensor attended = this.owner.tokensToChannels(attendedTokens);
      Tensor residual = this.owner.add(x, kernels.apply(this.out, attended));
      Tensor norm1 = this.owner.channelNorm(
        residual,
        "enc_p.encoder.norm_layers_1." + this.index + ".gamma",
        "enc_p.encoder.norm_layers_1." + this.index + ".beta");
      Tensor ffn = kernels.apply(
        this.ffn2,
        VitsSynthesizer.relu(kernels.apply(this.ffn1, this.owner.mulMask(norm1, mask))));
      Tensor residual2 = this.owner.add(norm1, ffn);
      return this.owner.mulMask(
        this.owner.channelNorm(
          residual2,
          "enc_p.encoder.norm_layers_2." + this.index + ".gamma",
          "enc_p.encoder.norm_layers_2." + this.index + ".beta"),
        mask);
    }

    private Tensor attendRelative(final Tensor q, final Tensor k, final Tensor v) {
      int qLen = q.size(0);
      int width = q.size(1);
      int window = (this.relKey.size(1) - 1) / 2;
      float scale = 1f / (float) Math.sqrt(this.headDim);
      Tensor out = Tensor.zeros(qLen, width);
      float[] qd = q.data();
      float[] kd = k.data();
      float[] vd = v.data();
      float[] od = out.data();
      int qOff = q.offset();
      int kOff = k.offset();
      int vOff = v.offset();
      float[] ek = this.relKey.data();
      float[] ev = this.relValue.data();
      int ekOff = this.relKey.offset();
      int evOff = this.relValue.offset();
      float[] scores = new float[qLen];
      for (int h = 0; h < this.heads; h++) {
        int headBase = h * this.headDim;
        for (int i = 0; i < qLen; i++) {
          int qi = qOff + i * width + headBase;
          float max = Float.NEGATIVE_INFINITY;
          for (int j = 0; j < qLen; j++) {
            float score = VectorMath.dot(qd, qi, kd, kOff + j * width + headBase, this.headDim)
              * scale;
            int rel = j - i;
            if (rel >= -window && rel <= window) {
              score += VectorMath.dot(
                qd, qi, ek, ekOff + (window + rel) * this.headDim, this.headDim) * scale;
            }
            scores[j] = score;
            if (score > max) {
              max = score;
            }
          }
          float sum = 0f;
          for (int j = 0; j < qLen; j++) {
            float e = (float) Math.exp(scores[j] - max);
            scores[j] = e;
            sum += e;
          }
          float inv = 1f / sum;
          int oi = i * width + headBase;
          Arrays.fill(od, oi, oi + this.headDim, 0f);
          for (int j = 0; j < qLen; j++) {
            float p = scores[j] * inv;
            VectorMath.axpy(od, oi, p, vd, vOff + j * width + headBase, this.headDim);
            int rel = j - i;
            if (rel >= -window && rel <= window) {
              VectorMath.axpy(od, oi, p, ev, evOff + (window + rel) * this.headDim, this.headDim);
            }
          }
        }
      }
      return out;
    }
  }

  private record Duration(VitsSynthesizer owner, Conv1d pre, List<DdsLayer> dds, Conv1d proj,
                          Tensor affineM, List<ConvFlow> convFlows, boolean stochastic) {

    static Duration load(final VitsSynthesizer owner) {
      if (owner.weights.has("dp.pre.weight")) {
        List<DdsLayer> dds = DdsLayer.loadAll(owner, "dp.convs", 3);
        List<ConvFlow> convFlows = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
          if (owner.weights.has("dp.flows." + i + ".pre.weight")) {
            convFlows.add(ConvFlow.load(owner, "dp.flows." + i));
          }
        }
        if (convFlows.isEmpty()) {
          throw new IllegalStateException("piper stochastic duration conv flows were not found");
        }
        return new Duration(
          owner,
          owner.conv("dp.pre", 1, 0),
          dds,
          owner.conv("dp.proj", 1, 0),
          owner.find("dp.flows.0.m").orElse(null),
          List.copyOf(convFlows),
          true);
      }
      if (owner.weights.has("dp.conv_1.weight")) {
        return new Duration(
          owner,
          owner.conv("dp.conv_1", 1, 1),
          List.of(),
          owner.conv("dp.proj", 1, 0),
          null,
          List.of(),
          false);
      }
      throw new IllegalStateException("piper duration predictor weights were not found");
    }

    Tensor logw(
      final Tensor x,
      final Tensor mask,
      final float noiseW,
      final Random noise,
      final Kernels kernels
    ) {
      Tensor h = this.owner.mulMask(kernels.apply(this.pre, x), mask);
      for (DdsLayer layer : this.dds) {
        h = layer.forward(h, mask, kernels);
      }
      h = this.owner.mulMask(kernels.apply(this.proj, h), mask);
      if (!this.stochastic) {
        return this.owner.mulMask(h, mask);
      }
      int time = x.size(1);
      Tensor z = Tensor.zeros(2, time);
      float[] zd = z.data();
      for (int i = 0; i < z.numel(); i++) {
        zd[i] = (float) noise.nextGaussian() * noiseW;
      }
      for (int i = this.convFlows.size() - 1; i >= 0; i--) {
        z = this.owner.flipChannels(z);
        z = this.convFlows.get(i).reverse(z, mask, h, kernels);
      }
      z = this.owner.flipChannels(z);
      z = this.affineReverse(z);
      return this.owner.channelSlice(this.owner.mulMask(z, mask), 0, 1);
    }

    private Tensor affineReverse(final Tensor z) {
      if (this.affineM == null) {
        return z;
      }
      int channels = z.size(0);
      int time = z.size(1);
      Tensor y = Tensor.zeros(channels, time);
      float[] zd = z.data();
      float[] yd = y.data();
      float[] md = this.affineM.data();
      int zOff = z.offset();
      int mOff = this.affineM.offset();
      for (int c = 0; c < channels; c++) {
        float mean = md[mOff + c];
        for (int t = 0; t < time; t++) {
          yd[c * time + t] = zd[zOff + c * time + t] - mean;
        }
      }
      return y;
    }
  }

  private record ConvFlow(VitsSynthesizer owner, Conv1d pre, List<DdsLayer> convs, Conv1d proj,
                          int filterChannels, int numBins) {

    static ConvFlow load(final VitsSynthesizer owner, final String prefix) {
      Conv1d pre = owner.conv(prefix + ".pre", 1, 0);
      Conv1d proj = owner.conv(prefix + ".proj", 1, 0);
      int numBins = (proj.outChannels() + 1) / 3;
      return new ConvFlow(
        owner,
        pre,
        DdsLayer.loadAll(owner, prefix + ".convs", 3),
        proj,
        pre.outChannels(),
        numBins);
    }

    Tensor reverse(final Tensor x, final Tensor mask, final Tensor condition,
                   final Kernels kernels) {
      int half = x.size(0) / 2;
      Tensor x0 = this.owner.channelSlice(x, 0, half);
      Tensor x1 = this.owner.channelSlice(x, half, x.size(0) - half);
      Tensor h = this.owner.add(kernels.apply(this.pre, x0), condition);
      for (DdsLayer layer : this.convs) {
        h = layer.forward(h, mask, kernels);
      }
      h = this.owner.mulMask(kernels.apply(this.proj, h), mask);
      Tensor warped = this.splineInverse(x1, h);
      return this.owner.mulMask(this.owner.concatChannels(x0, warped), mask);
    }

    private Tensor splineInverse(final Tensor x1, final Tensor params) {
      int time = x1.size(1);
      Tensor y = Tensor.zeros(1, time);
      float[] xd = x1.data();
      int xOff = x1.offset();
      float[] pd = params.data();
      int pOff = params.offset();
      float[] yd = y.data();
      float scale = (float) Math.sqrt(this.filterChannels);
      float[] widths = new float[this.numBins];
      float[] heights = new float[this.numBins];
      float[] derivs = new float[this.numBins - 1];
      for (int t = 0; t < time; t++) {
        for (int b = 0; b < this.numBins; b++) {
          widths[b] = pd[pOff + b * time + t] / scale;
          heights[b] = pd[pOff + (this.numBins + b) * time + t] / scale;
        }
        for (int b = 0; b < this.numBins - 1; b++) {
          derivs[b] = pd[pOff + (2 * this.numBins + b) * time + t];
        }
        yd[t] = inverseLinearTailRationalQuadratic(
          xd[xOff + t], widths, heights, derivs, SPLINE_TAIL_BOUND);
      }
      return y;
    }
  }

  private record DdsLayer(VitsSynthesizer owner, Conv1d depthwise, Conv1d pointwise, String gamma1,
                          String beta1, String gamma2, String beta2) {

    static List<DdsLayer> loadAll(final VitsSynthesizer owner, final String prefix,
                                  final int nLayers) {
      List<DdsLayer> layers = new ArrayList<>();
      for (int i = 0; i < nLayers; i++) {
        String sep = prefix + ".convs_sep." + i;
        if (!owner.weights.has(sep + ".weight")) {
          break;
        }
        int channels = owner.require(sep + ".weight").size(0);
        int kernel = owner.require(sep + ".weight").size(2);
        int dilation = intPow(kernel, i);
        int padding = (kernel * dilation - dilation) / 2;
        layers.add(new DdsLayer(
          owner,
          owner.conv(sep, 1, padding, dilation, channels),
          owner.conv(prefix + ".convs_1x1." + i, 1, 0),
          prefix + ".norms_1." + i + ".gamma",
          prefix + ".norms_1." + i + ".beta",
          prefix + ".norms_2." + i + ".gamma",
          prefix + ".norms_2." + i + ".beta"));
      }
      return List.copyOf(layers);
    }

    Tensor forward(final Tensor x, final Tensor mask, final Kernels kernels) {
      Tensor hidden = kernels.apply(this.depthwise, this.owner.mulMask(x, mask));
      hidden = VitsSynthesizer.gelu(this.owner.channelNorm(hidden, this.gamma1, this.beta1));
      hidden = kernels.apply(this.pointwise, hidden);
      hidden = VitsSynthesizer.gelu(this.owner.channelNorm(hidden, this.gamma2, this.beta2));
      return this.owner.mulMask(this.owner.add(x, hidden), mask);
    }
  }

  private record Coupling(VitsSynthesizer owner, Conv1d pre, List<WaveNetLayer> waveNet,
                          Conv1d post) {

    static Coupling load(final VitsSynthesizer owner, final String prefix) {
      List<WaveNetLayer> wn = new ArrayList<>();
      for (int i = 0; owner.weights.has(prefix + ".enc.in_layers." + i + ".weight"); i++) {
        wn.add(WaveNetLayer.load(owner, prefix + ".enc", i));
      }
      return new Coupling(
        owner,
        owner.conv(prefix + ".pre", 1, 0),
        List.copyOf(wn),
        owner.conv(prefix + ".post", 1, 0));
    }

    Tensor reverse(final Tensor x, final Tensor mask, final Kernels kernels) {
      int half = x.size(0) / 2;
      Tensor x0 = this.owner.channelSlice(x, 0, half);
      Tensor x1 = this.owner.channelSlice(x, half, x.size(0) - half);
      Tensor h = this.owner.mulMask(kernels.apply(this.pre, x0), mask);
      Tensor residual = h;
      Tensor skips = Tensor.zeros(h.shape());
      for (int i = 0; i < this.waveNet.size(); i++) {
        Tensor projected = this.waveNet.get(i).project(residual, mask, kernels);
        int hidden = residual.size(0);
        if (i == this.waveNet.size() - 1) {
          skips = this.owner.add(skips, projected);
        } else {
          residual = this.owner.mulMask(
            this.owner.add(residual, this.owner.channelSlice(projected, 0, hidden)),
            mask);
          skips = this.owner.add(skips, this.owner.channelSlice(projected, hidden, hidden));
        }
      }
      h = this.owner.mulMask(skips, mask);
      Tensor stats = this.owner.mulMask(kernels.apply(this.post, h), mask);
      Tensor mean = this.owner.channelSlice(stats, 0, x1.size(0));
      Tensor shifted = Tensor.zeros(x1.shape());
      float[] a = x1.data();
      float[] m = mean.data();
      float[] d = shifted.data();
      int ao = x1.offset();
      int mo = mean.offset();
      for (int i = 0; i < x1.numel(); i++) {
        d[i] = a[ao + i] - m[mo + i];
      }
      return this.owner.concatChannels(x0, this.owner.mulMask(shifted, mask));
    }
  }

  private record WaveNetLayer(VitsSynthesizer owner, Conv1d inConv, Conv1d resSkip) {

    static WaveNetLayer load(final VitsSynthesizer owner, final String prefix, final int index) {
      Tensor inW = owner.require(prefix + ".in_layers." + index + ".weight");
      int kernel = inW.size(2);
      int dilation = 1;
      int padding = (kernel - 1) / 2;
      return new WaveNetLayer(
        owner,
        owner.conv(prefix + ".in_layers." + index, 1, padding, dilation, 1),
        owner.conv(prefix + ".res_skip_layers." + index, 1, 0));
    }

    Tensor project(final Tensor x, final Tensor mask, final Kernels kernels) {
      Tensor acts = kernels.apply(this.inConv, this.owner.mulMask(x, mask));
      int hidden = acts.size(0) / 2;
      Tensor tanhGate = VitsSynthesizer.tanh(this.owner.channelSlice(acts, 0, hidden));
      Tensor sigmoidGate = this.sigmoid(this.owner.channelSlice(acts, hidden, hidden));
      Tensor fused = Tensor.zeros(hidden, x.size(1));
      float[] td = tanhGate.data();
      float[] sd = sigmoidGate.data();
      float[] fd = fused.data();
      for (int i = 0; i < fused.numel(); i++) {
        fd[i] = td[i] * sd[i];
      }
      return this.owner.mulMask(kernels.apply(this.resSkip, fused), mask);
    }

    private Tensor sigmoid(final Tensor x) {
      Tensor y = Tensor.zeros(x.shape());
      float[] xd = x.data();
      float[] yd = y.data();
      int off = x.offset();
      for (int i = 0; i < x.numel(); i++) {
        yd[i] = (float) (1.0 / (1.0 + Math.exp(-xd[off + i])));
      }
      return y;
    }
  }

  private record Generator(VitsSynthesizer owner, Conv1d pre, List<ConvTranspose1d> ups,
                           List<List<ResBlock>> resblocks, Conv1d post) {

    static Generator load(final VitsSynthesizer owner) {
      List<ConvTranspose1d> ups = new ArrayList<>();
      for (int i = 0; owner.weights.has("dec.ups." + i + ".weight"); i++) {
        Tensor w = owner.require("dec.ups." + i + ".weight");
        int kernel = w.size(2);
        int stride = guessUpsampleStride(kernel);
        int padding = Math.max(0, (kernel - stride) / 2);
        ups.add(owner.convTranspose("dec.ups." + i, stride, padding));
      }
      if (ups.isEmpty()) {
        throw new IllegalStateException("piper HiFi-GAN upsamplers were not found");
      }
      List<List<ResBlock>> blocks = new ArrayList<>();
      int cursor = 0;
      for (int u = 0; u < ups.size(); u++) {
        List<ResBlock> group = new ArrayList<>();
        for (int j = 0; j < 3; j++) {
          String prefix = "dec.resblocks." + cursor;
          if (!owner.weights.has(prefix + ".convs.0.weight")
            && !owner.weights.has(prefix + ".convs1.0.weight")
            && !owner.weights.has(prefix + ".convs1.0.weight_g")) {
            break;
          }
          group.add(ResBlock.load(owner, prefix));
          cursor++;
        }
        blocks.add(List.copyOf(group));
      }
      return new Generator(
        owner,
        owner.conv("dec.conv_pre", 1, 3),
        List.copyOf(ups),
        List.copyOf(blocks),
        owner.conv("dec.conv_post", 1, 3));
    }

    private static int guessUpsampleStride(final int kernel) {
      return switch (kernel) {
        case 16 -> 8;
        case 4 -> 2;
        case 8 -> 4;
        default -> Math.max(1, kernel / 2);
      };
    }

    Tensor forward(final Tensor z, final Kernels kernels) {
      Tensor x = kernels.apply(this.pre, z);
      for (int i = 0; i < this.ups.size(); i++) {
        x = kernels.apply(this.ups.get(i), VitsSynthesizer.leakyRelu(x));
        List<ResBlock> group = this.resblocks.get(i);
        if (group.isEmpty()) {
          continue;
        }
        Tensor acc = group.getFirst().forward(kernels, x);
        for (int j = 1; j < group.size(); j++) {
          acc = this.owner.add(acc, group.get(j).forward(kernels, x));
        }
        float inv = 1f / group.size();
        Tensor scaled = Tensor.zeros(acc.shape());
        VectorMath.scale(acc.data(), acc.offset(), inv, scaled.data(), 0, acc.numel());
        x = scaled;
      }
      return VitsSynthesizer.tanh(kernels.apply(this.post, VitsSynthesizer.leakyRelu(x)));
    }
  }

  private record ResBlock(List<Conv1d> residualConvs, List<Conv1d> convs1, List<Conv1d> convs2) {

    static ResBlock load(final VitsSynthesizer owner, final String prefix) {
      if (owner.weights.has(prefix + ".convs.0.weight")) {
        return loadResBlock2(owner, prefix);
      }
      return loadResBlock1(owner, prefix);
    }

    private static ResBlock loadResBlock2(final VitsSynthesizer owner, final String prefix) {
      List<Conv1d> convs = new ArrayList<>();
      for (int i = 0; owner.weights.has(prefix + ".convs." + i + ".weight"); i++) {
        String weightName = prefix + ".convs." + i + ".weight";
        Tensor w = owner.require(weightName);
        int kernel = w.size(2);
        int dilation = owner.weights.convDilation(weightName, 1);
        int padding = (kernel * dilation - dilation) / 2;
        convs.add(owner.conv(prefix + ".convs." + i, 1, padding, dilation, 1));
      }
      if (convs.isEmpty()) {
        throw new IllegalStateException("empty residual block " + prefix);
      }
      return new ResBlock(List.copyOf(convs), List.of(), List.of());
    }

    private static ResBlock loadResBlock1(final VitsSynthesizer owner, final String prefix) {
      List<Conv1d> convs1 = new ArrayList<>();
      List<Conv1d> convs2 = new ArrayList<>();
      for (int i = 0; owner.weights.has(prefix + ".convs1." + i + ".weight"); i++) {
        Tensor w = owner.require(prefix + ".convs1." + i + ".weight");
        int kernel = w.size(2);
        int dilation = switch (i) {
          case 1 -> 3;
          case 2 -> 5;
          default -> 1;
        };
        int padding = (kernel * dilation - dilation) / 2;
        convs1.add(owner.conv(prefix + ".convs1." + i, 1, padding, dilation, 1));
        Tensor w2 = owner.require(prefix + ".convs2." + i + ".weight");
        int k2 = w2.size(2);
        convs2.add(owner.conv(prefix + ".convs2." + i, 1, k2 / 2, 1, 1));
      }
      if (convs1.isEmpty()) {
        throw new IllegalStateException("empty residual block " + prefix);
      }
      return new ResBlock(List.of(), List.copyOf(convs1), List.copyOf(convs2));
    }

    private static Tensor addInPlaceCopy(final Tensor a, final Tensor b) {
      return Ops.add(a, b);
    }

    Tensor forward(final Kernels kernels, final Tensor x) {
      if (!this.residualConvs.isEmpty()) {
        Tensor hidden = x;
        for (Conv1d conv : this.residualConvs) {
          hidden = addInPlaceCopy(hidden, kernels.apply(conv, VitsSynthesizer.leakyRelu(hidden)));
        }
        return hidden;
      }
      Tensor hidden = x;
      for (int i = 0; i < this.convs1.size(); i++) {
        Tensor h = kernels.apply(this.convs1.get(i), VitsSynthesizer.leakyRelu(hidden));
        h = kernels.apply(this.convs2.get(i), VitsSynthesizer.leakyRelu(h));
        hidden = addInPlaceCopy(hidden, h);
      }
      return hidden;
    }
  }
}
