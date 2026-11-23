package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.internal.Context;
import com.igormaznitsa.nanollvm.layers.Conv1d;
import com.igormaznitsa.nanollvm.layers.Linear;
import com.igormaznitsa.nanollvm.layers.Norms;
import com.igormaznitsa.nanollvm.llm.TinyVocaSpec;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * Tiny-voca acoustic graph: token encoder, duration, prosody, and a source-filter vocoder.
 * Weights are named like the training checkpoint ({@code tokens.weight},
 * {@code encoder.blocks.0.attention.qkv.weight}, …).
 */
public final class TinyVocaGraph {

  private static final float NORM_EPS = 1e-5f;

  private final TinyVocaSpec spec;
  private final float[] tokens;
  private final float[] stress;
  private final float[] voices;
  private final float[] frameVoices;
  private final Stack encoder;
  private final Conv1d durationConv1;
  private final Norms.LayerNorm durationNorm1;
  private final Conv1d durationConv2;
  private final Norms.LayerNorm durationNorm2;
  private final Linear durationOut;
  private final Linear framePosition;
  private final Stack prosody;
  private final Linear prosodyHead;
  private final Linear prosodyProjection;
  private final Stack acoustic;
  private final Linear harmonic;
  private final Linear noise;

  public TinyVocaGraph(final TinyVocaSpec spec, final Map<String, Tensor> weights) {
    this.spec = requireNonNull(spec, "spec");
    requireNonNull(weights, "weights");
    int channels = spec.channels();
    int speakers = spec.speakers().size();
    this.tokens = matrix(weights, "tokens.weight", spec.symbols().size(), channels).toFloatArray();
    this.stress = matrix(weights, "stress.weight", 3, channels).toFloatArray();
    this.voices = matrix(weights, "voices.weight", speakers, channels).toFloatArray();
    this.frameVoices = matrix(weights, "frame_voices.weight", speakers, channels).toFloatArray();
    this.encoder = stack(weights, "encoder", spec.encoderLayers(), spec);
    int predictor = spec.predictorChannels();
    this.durationConv1 = conv(weights, "duration.conv1", predictor, channels, 3);
    this.durationNorm1 = layerNorm(weights, "duration.norm1", predictor);
    this.durationConv2 = conv(weights, "duration.conv2", predictor, predictor, 3);
    this.durationNorm2 = layerNorm(weights, "duration.norm2", predictor);
    this.durationOut = linear(weights, "duration.out", 2, predictor);
    this.framePosition = linear(weights, "frame_position", channels, 3);
    this.prosody = stack(weights, "prosody_stack", spec.prosodyLayers(), spec);
    this.prosodyHead = linear(weights, "prosody_head", 3, channels);
    this.prosodyProjection = linear(weights, "prosody_projection", channels, 3);
    this.acoustic = stack(weights, "acoustic_stack", spec.acousticLayers(), spec);
    this.harmonic = linear(weights, "vocoder.harmonic", spec.harmonicBins(), channels);
    this.noise = linear(weights, "vocoder.noise", spec.noiseBins(), channels);
  }

  /**
   * Parameter name to shape, including biases. Used to build a synthetic checkpoint.
   *
   * @param spec blueprint; must not be {@code null}
   * @return insertion-ordered shapes
   */
  public static Map<String, int[]> parameterShapes(final TinyVocaSpec spec) {
    requireNonNull(spec, "spec");
    int channels = spec.channels();
    int feedforward = spec.feedforward();
    int kernel = spec.kernel();
    int predictor = spec.predictorChannels();
    int speakers = spec.speakers().size();
    Map<String, int[]> shapes = new LinkedHashMap<>();
    shapes.put("tokens.weight", new int[] {spec.symbols().size(), channels});
    shapes.put("stress.weight", new int[] {3, channels});
    shapes.put("voices.weight", new int[] {speakers, channels});
    shapes.put("frame_voices.weight", new int[] {speakers, channels});
    putStack(shapes, "encoder", spec.encoderLayers(), channels, feedforward, kernel);
    shapes.put("duration.conv1.weight", new int[] {predictor, channels, 3});
    shapes.put("duration.conv1.bias", new int[] {predictor});
    putNorm(shapes, "duration.norm1", predictor);
    shapes.put("duration.conv2.weight", new int[] {predictor, predictor, 3});
    shapes.put("duration.conv2.bias", new int[] {predictor});
    putNorm(shapes, "duration.norm2", predictor);
    shapes.put("duration.out.weight", new int[] {2, predictor});
    shapes.put("duration.out.bias", new int[] {2});
    shapes.put("frame_position.weight", new int[] {channels, 3});
    shapes.put("frame_position.bias", new int[] {channels});
    putStack(shapes, "prosody_stack", spec.prosodyLayers(), channels, feedforward, kernel);
    shapes.put("prosody_head.weight", new int[] {3, channels});
    shapes.put("prosody_head.bias", new int[] {3});
    shapes.put("prosody_projection.weight", new int[] {channels, 3});
    shapes.put("prosody_projection.bias", new int[] {channels});
    putStack(shapes, "acoustic_stack", spec.acousticLayers(), channels, feedforward, kernel);
    shapes.put("vocoder.harmonic.weight", new int[] {spec.harmonicBins(), channels});
    shapes.put("vocoder.harmonic.bias", new int[] {spec.harmonicBins()});
    shapes.put("vocoder.noise.weight", new int[] {spec.noiseBins(), channels});
    shapes.put("vocoder.noise.bias", new int[] {spec.noiseBins()});
    return shapes;
  }

  private static int[] owners(final int[] durations, final int frames) {
    int[] owner = new int[frames];
    int time = 0;
    for (int n = 0; n < durations.length; n++) {
      for (int k = 0; k < durations[n]; k++, time++) {
        owner[time] = n;
      }
    }
    return owner;
  }

  private static int roundEven(final double value) {
    return (int) Math.rint(value);
  }

  private static double lifter(final int bin) {
    return bin == 0 ? 2.0 : 1.0 / (1.0 + bin / 4.0);
  }

  private static void putStack(
    final Map<String, int[]> shapes,
    final String name,
    final int layers,
    final int channels,
    final int feedforward,
    final int kernel
  ) {
    putNorm(shapes, name + ".norm", channels);
    for (int i = 0; i < layers; i++) {
      String at = name + ".blocks." + i + ".";
      putNorm(shapes, at + "norm1", channels);
      putNorm(shapes, at + "norm2", channels);
      shapes.put(at + "attention.qkv.weight", new int[] {3 * channels, channels});
      shapes.put(at + "attention.qkv.bias", new int[] {3 * channels});
      shapes.put(at + "attention.out.weight", new int[] {channels, channels});
      shapes.put(at + "attention.out.bias", new int[] {channels});
      shapes.put(at + "depthwise.weight", new int[] {channels, 1, kernel});
      shapes.put(at + "depthwise.bias", new int[] {channels});
      shapes.put(at + "expand.weight", new int[] {feedforward, channels});
      shapes.put(at + "expand.bias", new int[] {feedforward});
      shapes.put(at + "project.weight", new int[] {channels, feedforward});
      shapes.put(at + "project.bias", new int[] {channels});
    }
  }

  private static void putNorm(final Map<String, int[]> shapes, final String name, final int width) {
    shapes.put(name + ".weight", new int[] {width});
    shapes.put(name + ".bias", new int[] {width});
  }

  private static Stack stack(
    final Map<String, Tensor> weights,
    final String name,
    final int layers,
    final TinyVocaSpec spec
  ) {
    Block[] blocks = new Block[layers];
    for (int i = 0; i < layers; i++) {
      blocks[i] = block(weights, name + ".blocks." + i, spec);
    }
    return new Stack(blocks, layerNorm(weights, name + ".norm", spec.channels()), spec.heads());
  }

  private static Block block(
    final Map<String, Tensor> weights,
    final String name,
    final TinyVocaSpec spec
  ) {
    int channels = spec.channels();
    int feedforward = spec.feedforward();
    return new Block(
      layerNorm(weights, name + ".norm1", channels),
      linear(weights, name + ".attention.qkv", 3 * channels, channels),
      linear(weights, name + ".attention.out", channels, channels),
      layerNorm(weights, name + ".norm2", channels),
      conv(weights, name + ".depthwise", channels, 1, spec.kernel(), channels),
      linear(weights, name + ".expand", feedforward, channels),
      linear(weights, name + ".project", channels, feedforward));
  }

  private static Linear linear(
    final Map<String, Tensor> weights,
    final String name,
    final int out,
    final int in
  ) {
    return new Linear(matrix(weights, name + ".weight", out, in),
      bias(weights, name + ".bias", out));
  }

  private static Norms.LayerNorm layerNorm(
    final Map<String, Tensor> weights,
    final String name,
    final int width
  ) {
    return new Norms.LayerNorm(
      vector(weights, name + ".weight", width),
      vector(weights, name + ".bias", width),
      NORM_EPS);
  }

  private static Conv1d conv(
    final Map<String, Tensor> weights,
    final String name,
    final int out,
    final int in,
    final int kernel
  ) {
    return conv(weights, name, out, in, kernel, 1);
  }

  private static Conv1d conv(
    final Map<String, Tensor> weights,
    final String name,
    final int out,
    final int inPerGroup,
    final int kernel,
    final int groups
  ) {
    Tensor weight = convWeight(weights, name + ".weight", out, inPerGroup, kernel);
    return new Conv1d(weight, bias(weights, name + ".bias", out), 1, kernel / 2, 1, groups);
  }

  private static Tensor matrix(
    final Map<String, Tensor> weights,
    final String name,
    final int out,
    final int in
  ) {
    Tensor tensor = require(weights, name);
    if (tensor.shape().length != 2) {
      throw new IllegalArgumentException(name + " must be a matrix");
    }
    if (tensor.size(0) == out && tensor.size(1) == in) {
      return tensor;
    }
    if (tensor.size(0) == in && tensor.size(1) == out) {
      return transpose(tensor);
    }
    throw new IllegalArgumentException(
      name + " shape " + Arrays.toString(tensor.shape()) + " is not [" + out + ", " + in + "]");
  }

  private static Tensor convWeight(
    final Map<String, Tensor> weights,
    final String name,
    final int out,
    final int inPerGroup,
    final int kernel
  ) {
    Tensor tensor = require(weights, name);
    int[] shape = tensor.shape();
    if (shape.length == 4 && shape[3] == 1) {
      tensor = tensor.reshape(shape[0], shape[1], shape[2]);
      shape = tensor.shape();
    }
    if (shape.length == 2 && inPerGroup == 1 && shape[0] == out && shape[1] == kernel) {
      tensor = tensor.reshape(out, 1, kernel);
      shape = tensor.shape();
    }
    if (shape.length != 3 || shape[0] != out || shape[1] != inPerGroup || shape[2] != kernel) {
      throw new IllegalArgumentException(
        name + " shape " + Arrays.toString(shape)
          + " is not [" + out + ", " + inPerGroup + ", " + kernel + "]");
    }
    return tensor;
  }

  private static Tensor vector(
    final Map<String, Tensor> weights,
    final String name,
    final int width
  ) {
    Tensor tensor = require(weights, name);
    if (tensor.numel() != width) {
      throw new IllegalArgumentException(name + " length must be " + width);
    }
    return tensor;
  }

  private static Tensor bias(
    final Map<String, Tensor> weights,
    final String name,
    final int width
  ) {
    Tensor tensor = weights.get(name);
    if (tensor == null) {
      return null;
    }
    if (tensor.numel() != width) {
      throw new IllegalArgumentException(name + " length must be " + width);
    }
    return tensor;
  }

  private static Tensor require(final Map<String, Tensor> weights, final String name) {
    Tensor tensor = weights.get(name);
    if (tensor == null) {
      throw new IllegalArgumentException("missing tiny-voca weight " + name);
    }
    return tensor;
  }

  private static Tensor transpose(final Tensor tensor) {
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    float[] source = tensor.toFloatArray();
    float[] dest = new float[source.length];
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        dest[col * rows + row] = source[row * cols + col];
      }
    }
    return Tensor.of(dest, cols, rows);
  }

  private static float[] apply(
    final Linear layer,
    final float[] rows,
    final int count,
    final int in,
    final Context context
  ) {
    return layer.forward(Tensor.of(rows, count, in), context).toFloatArray();
  }

  private static float[] norm(
    final Norms.LayerNorm layer,
    final float[] rows,
    final int count,
    final int width
  ) {
    return layer.forward(Tensor.of(rows, count, width)).toFloatArray();
  }

  private static float[] gelu(final float[] values) {
    float[] out = new float[values.length];
    double scale = 1.0 / Math.sqrt(2.0);
    for (int i = 0; i < values.length; i++) {
      double value = values[i];
      out[i] = (float) (0.5 * value * (1.0 + erf(value * scale)));
    }
    return out;
  }

  private static double erf(final double value) {
    double abs = Math.abs(value);
    if (abs > 6.0) {
      return Math.copySign(1.0, value);
    }
    double term = value;
    double sum = value;
    for (int n = 1; Math.abs(term) > 1e-17 * Math.abs(sum); n++) {
      term *= 2 * value * value / (2 * n + 1);
      sum += term;
    }
    return 2 / Math.sqrt(Math.PI) * Math.exp(-value * value) * sum;
  }

  private static float[] convTimeMajor(
    final float[] timeMajor,
    final int time,
    final int channels,
    final Conv1d conv,
    final MatmulRuntime runtime
  ) {
    float[] channelMajor = new float[time * channels];
    for (int t = 0; t < time; t++) {
      for (int c = 0; c < channels; c++) {
        channelMajor[c * time + t] = timeMajor[t * channels + c];
      }
    }
    Tensor filtered = conv.forward(Tensor.of(channelMajor, channels, time), runtime);
    int outChannels = filtered.size(0);
    int outTime = filtered.size(1);
    float[] back = new float[outTime * outChannels];
    float[] data = filtered.toFloatArray();
    for (int t = 0; t < outTime; t++) {
      for (int c = 0; c < outChannels; c++) {
        back[t * outChannels + c] = data[c * outTime + t];
      }
    }
    return back;
  }

  private static void add(final float[] into, final float[] extra) {
    for (int i = 0; i < into.length; i++) {
      into[i] += extra[i];
    }
  }

  private static float[] gaussian(final int count, final Random random) {
    float[] out = new float[count];
    for (int i = 0; i < count; i += 2) {
      double radius = Math.sqrt(-2.0 * Math.log(1.0 - random.nextDouble()));
      double angle = 2.0 * Math.PI * random.nextDouble();
      out[i] = (float) (radius * Math.cos(angle));
      if (i + 1 < count) {
        out[i + 1] = (float) (radius * Math.sin(angle));
      }
    }
    return out;
  }

  private static float[] pulseTrain(
    final float[] f0,
    final int hop,
    final int sampleRate,
    final float limit
  ) {
    int frames = f0.length;
    double[] knots = new double[frames + 1];
    for (int i = 0; i < knots.length; i++) {
      knots[i] = Math.log(Math.max(f0[Math.min(i, frames - 1)], 1f));
    }
    float[] out = new float[frames * hop];
    double turns = 0.0;
    for (int n = 0; n < out.length; n++) {
      int low = n / hop;
      double weight = n / (double) hop - low;
      double frequency = Math.exp(knots[low] * (1.0 - weight) + knots[low + 1] * weight);
      turns += frequency / sampleRate;
      double theta = (2.0 * Math.PI * turns) % (2.0 * Math.PI);
      double harmonics = Math.max(limit * sampleRate / frequency, 1.0);
      int whole = (int) Math.floor(harmonics);
      double below = 2.0 * Math.sin(theta / 2.0);
      double core = Math.abs(below) < 1e-7
        ? whole
        : Math.sin((whole + 0.5) * theta) / below - 0.5;
      out[n] = (float) ((core + (harmonics - whole) * Math.cos((whole + 1) * theta))
        * Math.sqrt(2.0 / harmonics));
    }
    return out;
  }

  private static float[] filtered(
    final float[] source,
    final int size,
    final int hop,
    final int frames,
    final Cepstrum cepstrum
  ) {
    int half = size / 2;
    double[] window = new double[size];
    for (int n = 0; n < size; n++) {
      window[n] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * n / size);
    }
    double[] cos = new double[size / 2];
    double[] sin = new double[size / 2];
    for (int k = 0; k < cos.length; k++) {
      double angle = -2.0 * Math.PI * k / size;
      cos[k] = Math.cos(angle);
      sin[k] = Math.sin(angle);
    }
    double[] re = new double[size];
    double[] im = new double[size];
    double[] cr = new double[size];
    double[] ci = new double[size];
    double[] sum = new double[frames * hop + size];
    double[] weight = new double[sum.length];
    for (int frame = 0; frame < frames; frame++) {
      int first = frame * hop - half;
      for (int n = 0; n < size; n++) {
        int at = first + n;
        re[n] = at >= 0 && at < source.length ? source[at] * window[n] : 0.0;
        im[n] = 0.0;
      }
      fft(re, im, cos, sin);
      Arrays.fill(cr, 0.0);
      Arrays.fill(ci, 0.0);
      cepstrum.place(frame, cr);
      fft(cr, ci, cos, sin);
      for (int k = 0; k <= half; k++) {
        double gain = Math.exp(Math.clamp(cr[k], -14.0, 6.0));
        double gainCos = gain * Math.cos(ci[k]);
        double gainSin = gain * Math.sin(ci[k]);
        double left = re[k];
        double right = im[k];
        re[k] = left * gainCos - right * gainSin;
        im[k] = -(left * gainSin + right * gainCos);
        if (k > 0 && k < half) {
          re[size - k] = re[k];
          im[size - k] = -im[k];
        }
      }
      fft(re, im, cos, sin);
      for (int n = 0; n < size; n++) {
        sum[frame * hop + n] += re[n] / size * window[n];
        weight[frame * hop + n] += window[n] * window[n];
      }
    }
    float[] out = new float[source.length];
    for (int n = 0; n < out.length; n++) {
      double scale = weight[n + half];
      out[n] = scale == 0.0 ? 0f : (float) (sum[n + half] / scale);
    }
    return out;
  }

  private static void fft(final double[] re, final double[] im, final double[] cos,
                          final double[] sin) {
    int size = re.length;
    for (int i = 1, j = 0; i < size; i++) {
      int bit = size >> 1;
      for (; (j & bit) != 0; bit >>= 1) {
        j ^= bit;
      }
      j ^= bit;
      if (i < j) {
        double swap = re[i];
        re[i] = re[j];
        re[j] = swap;
        swap = im[i];
        im[i] = im[j];
        im[j] = swap;
      }
    }
    for (int span = 2; span <= size; span *= 2) {
      int half = span / 2;
      for (int k = 0; k < half; k++) {
        double turnCos = cos[k * size / span];
        double turnSin = sin[k * size / span];
        for (int a = k; a < size; a += span) {
          int b = a + half;
          double x = re[b] * turnCos - im[b] * turnSin;
          double y = re[b] * turnSin + im[b] * turnCos;
          re[b] = re[a] - x;
          im[b] = im[a] - y;
          re[a] += x;
          im[a] += y;
        }
      }
    }
  }

  /**
   * One utterance to mono PCM at {@link TinyVocaSpec#sampleRate()}.
   *
   * @param ids        inventory ids
   * @param stress     stress aligned with {@code ids}
   * @param voice      speaker id
   * @param speed      duration divisor, {@code > 0}
   * @param pitchShift added to log pitch
   * @param pitchRange scale around the voiced mean; {@code 1} leaves the prediction
   * @param runtime    matmul pool; must not be {@code null}
   * @param random     vocoder noise; must not be {@code null}
   * @return samples, length {@code frames * hop}
   */
  public float[] synthesize(
    final int[] ids,
    final int[] stress,
    final int voice,
    final float speed,
    final float pitchShift,
    final float pitchRange,
    final MatmulRuntime runtime,
    final Random random
  ) {
    requireNonNull(ids, "ids");
    requireNonNull(stress, "stress");
    requireNonNull(runtime, "runtime");
    requireNonNull(random, "random");
    this.requireUtterance(ids, stress, voice, speed);
    Context context = new Context();
    context.bindMatmul(runtime);
    int channels = this.spec.channels();
    float[] embedded = this.embed(ids, stress, voice, channels);
    float[] hidden = this.encoder.forward(embedded, ids.length, context);

    int[] durations = this.durations(hidden, ids, speed, context);
    int frames = 0;
    for (int duration : durations) {
      frames += duration;
    }
    if (frames < 1 || frames > this.spec.maxFrames()) {
      throw new UtteranceTooLongException("the utterance is too long or empty: split the text");
    }

    float[] placed = this.place(hidden, durations, frames, voice, channels, context);
    float[] melody = this.prosody.forward(placed, frames, context);
    float[] prosody = apply(this.prosodyHead, melody, frames, channels, context);
    float[] f0 = new float[frames];
    float[] told = this.clampProsody(prosody, ids, durations, pitchShift, pitchRange, f0);
    float[] projected = apply(this.prosodyProjection, told, frames, 3, context);
    add(projected, melody);
    float[] acoustic = this.acoustic.forward(projected, frames, context);
    return this.vocode(acoustic, f0, frames, channels, random, context);
  }

  private void requireUtterance(
    final int[] ids,
    final int[] stress,
    final int voice,
    final float speed
  ) {
    if (ids.length < 1 || ids.length > this.spec.maxTokens() || stress.length != ids.length) {
      throw new IllegalArgumentException(
        "expected 1 to " + this.spec.maxTokens() + " phoneme ids and as many stress values");
    }
    if (voice < 0 || voice >= this.spec.speakers().size()) {
      throw new IllegalArgumentException("unknown voice index " + voice);
    }
    if (!(speed > 0f) || Float.isNaN(speed)) {
      throw new IllegalArgumentException("speed must be positive");
    }
  }

  private float[] embed(final int[] ids, final int[] stress, final int voice, final int channels) {
    float[] rows = new float[ids.length * channels];
    int voiceAt = voice * channels;
    for (int n = 0; n < ids.length; n++) {
      int tokenAt = ids[n] * channels;
      int stressAt = stress[n] * channels;
      int row = n * channels;
      for (int i = 0; i < channels; i++) {
        rows[row + i] =
          this.tokens[tokenAt + i] + this.stress[stressAt + i] + this.voices[voiceAt + i];
      }
    }
    return rows;
  }

  private int[] durations(
    final float[] hidden,
    final int[] ids,
    final float speed,
    final Context context
  ) {
    int count = ids.length;
    int channels = this.spec.channels();
    float[] wide = convTimeMajor(hidden, count, channels, this.durationConv1, context.matmul());
    wide = gelu(wide);
    wide = norm(this.durationNorm1, wide, count, this.spec.predictorChannels());
    wide = convTimeMajor(wide, count, this.spec.predictorChannels(), this.durationConv2,
      context.matmul());
    wide = gelu(wide);
    wide = norm(this.durationNorm2, wide, count, this.spec.predictorChannels());
    float[] logits = apply(this.durationOut, wide, count, this.spec.predictorChannels(), context);
    int[] durations = new int[count];
    double cap = Math.log1p(400);
    for (int n = 0; n < count; n++) {
      float length = (float) Math.expm1(Math.clamp(logits[2 * n], 0d, cap));
      boolean present = !this.spec.isPause(ids[n]) || logits[2 * n + 1] > 0f;
      durations[n] = present ? Math.max(roundEven(length / speed), 1) : 0;
    }
    return durations;
  }

  private float[] place(
    final float[] hidden,
    final int[] durations,
    final int frames,
    final int voice,
    final int channels,
    final Context context
  ) {
    int[] owner = new int[frames];
    float[] place = new float[frames * 3];
    int time = 0;
    for (int n = 0; n < durations.length; n++) {
      for (int k = 0; k < durations[n]; k++, time++) {
        owner[time] = n;
        float along = (k + 0.5f) / durations[n];
        place[3 * time] = along;
        place[3 * time + 1] = 1f - along;
        place[3 * time + 2] = (float) (Math.log1p(durations[n]) / 4.0);
      }
    }
    float[] rows = apply(this.framePosition, place, frames, 3, context);
    int voiceAt = voice * channels;
    for (int t = 0; t < frames; t++) {
      int from = owner[t] * channels;
      int row = t * channels;
      for (int i = 0; i < channels; i++) {
        rows[row + i] += hidden[from + i] + this.frameVoices[voiceAt + i];
      }
    }
    return rows;
  }

  private float[] clampProsody(
    final float[] prosody,
    final int[] ids,
    final int[] durations,
    final float pitchShift,
    final float pitchRange,
    final float[] f0
  ) {
    int frames = f0.length;
    int[] owner = owners(durations, frames);
    float voicedSum = 0f;
    int voicedCount = 0;
    for (int t = 0; t < frames; t++) {
      if (this.spec.isVowel(ids[owner[t]])) {
        prosody[3 * t + 1] = Math.max(prosody[3 * t + 1], 2f);
      }
      if (prosody[3 * t + 1] > 0f) {
        voicedSum += prosody[3 * t];
        voicedCount++;
      }
    }
    float mean = voicedSum / Math.max(voicedCount, 1);
    float[] told = new float[frames * 3];
    for (int t = 0; t < frames; t++) {
      float raw = prosody[3 * t];
      float pitch = pitchRange == 1f ? raw : mean + (raw - mean) * pitchRange;
      told[3 * t] = Math.clamp(pitch, -2f, 1.5f);
      told[3 * t + 1] = prosody[3 * t + 1] > 0f ? 1f : 0f;
      told[3 * t + 2] = Math.clamp(prosody[3 * t + 2], -4.5f, 3f);
      float hertz = this.spec.pitchReferenceHz() * (float) Math.exp(pitch + pitchShift);
      f0[t] = Math.clamp(hertz, this.spec.f0Min(), this.spec.f0Max());
    }
    return told;
  }

  private float[] vocode(
    final float[] acoustic,
    final float[] f0,
    final int frames,
    final int channels,
    final Random random,
    final Context context
  ) {
    int samples = frames * this.spec.hop();
    float[] harmonic = apply(this.harmonic, acoustic, frames, channels, context);
    float[] noise = apply(this.noise, acoustic, frames, channels, context);
    int harmonicSteps = this.spec.hop() / this.spec.synthHop();
    int noiseSteps = this.spec.hop() / this.spec.noiseHop();
    float[] voiced = filtered(
      pulseTrain(f0, this.spec.hop(), this.spec.sampleRate(), this.spec.harmonicLimit()),
      this.spec.synthFft(),
      this.spec.synthHop(),
      harmonicSteps * frames + 3,
      (frame, cepstrum) -> placeHarmonic(harmonic, frames, harmonicSteps, frame, cepstrum));
    float[] breath = filtered(
      gaussian(samples, random),
      this.spec.noiseFft(),
      this.spec.noiseHop(),
      noiseSteps * frames + 3,
      (frame, cepstrum) -> placeNoise(noise, frames, noiseSteps, frame, cepstrum));
    add(voiced, breath);
    return voiced;
  }

  private void placeHarmonic(
    final float[] bins,
    final int frames,
    final int steps,
    final int frame,
    final double[] cepstrum
  ) {
    int width = this.spec.harmonicBins();
    int sub = Math.min(frame, steps * frames - 1);
    int here = sub / steps;
    int next = Math.min((sub + 1) / steps, frames - 1);
    for (int i = 0; i < this.spec.cepCausal(); i++) {
      cepstrum[i] = 0.5 * (bins[here * width + i] + bins[next * width + i]) * lifter(i);
    }
    for (int i = 0; i < this.spec.cepAnti(); i++) {
      int at = this.spec.cepCausal() + i;
      cepstrum[this.spec.synthFft() - 1 - i] =
        0.5 * (bins[here * width + at] + bins[next * width + at]) * lifter(i + 1);
    }
    cepstrum[0] += bins[here * width + this.spec.cepCausal() + this.spec.cepAnti() + sub % steps];
  }

  private void placeNoise(
    final float[] bins,
    final int frames,
    final int steps,
    final int frame,
    final double[] cepstrum
  ) {
    int width = this.spec.noiseBins();
    int sub = Math.min(frame, steps * frames - 1);
    int here = sub / steps;
    for (int i = 0; i < this.spec.noiseCep(); i++) {
      cepstrum[i] = bins[here * width + i] * lifter(i);
    }
    cepstrum[0] += bins[here * width + this.spec.noiseCep() + sub % steps];
  }

  @FunctionalInterface
  private interface Cepstrum {
    void place(int frame, double[] cepstrum);
  }

  private static final class Stack {
    private final Block[] blocks;
    private final Norms.LayerNorm norm;
    private final int heads;

    private Stack(final Block[] blocks, final Norms.LayerNorm norm, final int heads) {
      this.blocks = blocks;
      this.norm = norm;
      this.heads = heads;
    }

    private static double[][] rotary(final int count, final int width) {
      double[][] turn = new double[count][width];
      for (int t = 0; t < count; t++) {
        for (int i = 0; i < width; i += 2) {
          double angle = t / Math.pow(10_000.0, (double) i / width);
          turn[t][i] = Math.cos(angle);
          turn[t][i + 1] = Math.sin(angle);
        }
      }
      return turn;
    }

    private float[] forward(final float[] rows, final int count, final Context context) {
      int channels = rows.length / count;
      int width = channels / this.heads;
      double[][] turn = rotary(count, width);
      float[] hidden = rows;
      for (Block block : this.blocks) {
        hidden = block.forward(hidden, count, this.heads, turn, context);
      }
      return norm(this.norm, hidden, count, channels);
    }
  }

  private static final class Block {
    private final Norms.LayerNorm norm1;
    private final Linear qkv;
    private final Linear out;
    private final Norms.LayerNorm norm2;
    private final Conv1d depthwise;
    private final Linear expand;
    private final Linear project;

    private Block(
      final Norms.LayerNorm norm1,
      final Linear qkv,
      final Linear out,
      final Norms.LayerNorm norm2,
      final Conv1d depthwise,
      final Linear expand,
      final Linear project
    ) {
      this.norm1 = norm1;
      this.qkv = qkv;
      this.out = out;
      this.norm2 = norm2;
      this.depthwise = depthwise;
      this.expand = expand;
      this.project = project;
    }

    private float[] forward(
      final float[] rows,
      final int count,
      final int heads,
      final double[][] turn,
      final Context context
    ) {
      int channels = rows.length / count;
      float[] attended = attend(
        norm(this.norm1, rows, count, channels), count, channels, heads, turn, context);
      add(rows, attended);
      float[] near = convTimeMajor(
        norm(this.norm2, rows, count, channels), count, channels, this.depthwise, context.matmul());
      float[] expanded = gelu(apply(this.expand, near, count, channels, context));
      add(rows, apply(this.project, expanded, count, this.expandWidth(expanded, count), context));
      return rows;
    }

    private int expandWidth(final float[] expanded, final int count) {
      return expanded.length / count;
    }

    private float[] attend(
      final float[] rows,
      final int count,
      final int channels,
      final int heads,
      final double[][] turn,
      final Context context
    ) {
      float[] qkv = apply(this.qkv, rows, count, channels, context);
      int width = channels / heads;
      float scale = (float) (1.0 / Math.sqrt(width));
      float[] mixed = new float[count * channels];
      float[] queries = new float[count * width];
      float[] keys = new float[count * width];
      float[] values = new float[width * count];
      float[] share = new float[count * count];
      for (int head = 0; head < heads; head++) {
        for (int t = 0; t < count; t++) {
          int queryAt = t * 3 * channels + head * width;
          int keyAt = queryAt + channels;
          int valueAt = keyAt + channels;
          int row = t * width;
          for (int i = 0; i < width; i += 2) {
            double cos = turn[t][i];
            double sin = turn[t][i + 1];
            queries[row + i] = (float) (qkv[queryAt + i] * cos - qkv[queryAt + i + 1] * sin);
            queries[row + i + 1] = (float) (qkv[queryAt + i] * sin + qkv[queryAt + i + 1] * cos);
            keys[row + i] = (float) (qkv[keyAt + i] * cos - qkv[keyAt + i + 1] * sin);
            keys[row + i + 1] = (float) (qkv[keyAt + i] * sin + qkv[keyAt + i + 1] * cos);
          }
          for (int i = 0; i < width; i++) {
            values[i * count + t] = qkv[valueAt + i];
          }
        }
        for (int t = 0; t < count; t++) {
          int row = t * count;
          float top = Float.NEGATIVE_INFINITY;
          for (int j = 0; j < count; j++) {
            float dot = 0f;
            int queryRow = t * width;
            int keyRow = j * width;
            for (int i = 0; i < width; i++) {
              dot += queries[queryRow + i] * keys[keyRow + i];
            }
            share[row + j] = dot;
            if (dot > top) {
              top = dot;
            }
          }
          float total = 0f;
          for (int j = 0; j < count; j++) {
            float score = (float) Math.exp((share[row + j] - top) * scale);
            share[row + j] = score;
            total += score;
          }
          float inverse = 1f / total;
          for (int j = 0; j < count; j++) {
            share[row + j] *= inverse;
          }
        }
        for (int t = 0; t < count; t++) {
          int dest = t * channels + head * width;
          for (int i = 0; i < width; i++) {
            float sum = 0f;
            for (int j = 0; j < count; j++) {
              sum += share[t * count + j] * values[i * count + j];
            }
            mixed[dest + i] = sum;
          }
        }
      }
      return apply(this.out, mixed, count, channels, context);
    }
  }
}
