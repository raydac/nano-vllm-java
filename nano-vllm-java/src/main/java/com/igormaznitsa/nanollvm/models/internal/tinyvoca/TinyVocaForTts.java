package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import static com.igormaznitsa.nanollvm.models.internal.WeightNames.ARCH_TINY_VOCA;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.TinyVocaSpec;
import com.igormaznitsa.nanollvm.models.internal.TextToSpeech;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Random;

/**
 * Tiny-voca text-to-speech: phonemize, then the source-filter graph. The loaded voice,
 * speed, pitch, and range come from {@link com.igormaznitsa.nanollvm.models.LlmOptionalData}.
 * A leading {@code voice=NAME}, {@code speed=}, {@code pitch=}, or {@code range=} on the text
 * overrides that call. {@code espeak-ng-data} is unused.
 */
public final class TinyVocaForTts implements TextToSpeech {

  private static final float PEAK = 0.99f;

  private final TinyVocaSpec spec;
  private final TinyVocaGraph graph;
  private final TinyVocaFront front;
  private final int defaultVoice;
  private final float defaultSpeed;
  private final float defaultPitch;
  private final float defaultRange;

  public TinyVocaForTts(
    final TinyVocaSpec spec,
    final TinyVocaGraph graph,
    final TinyVocaFront front,
    final String voice,
    final float speed,
    final float pitch,
    final float range
  ) {
    this.spec = requireNonNull(spec, "spec");
    this.graph = requireNonNull(graph, "graph");
    this.front = requireNonNull(front, "front");
    int index = voice == null || voice.isBlank() ? 0 : spec.voiceIndex(voice);
    if (index < 0) {
      throw new IllegalArgumentException(this.unknownVoice(voice));
    }
    this.defaultVoice = index;
    this.defaultSpeed = Math.clamp(speed, 0.25f, 4f);
    this.defaultPitch = Math.clamp(pitch, -2f, 2f);
    this.defaultRange = Math.clamp(range, 0f, 3f);
  }

  private static float[] concat(final List<float[]> parts) {
    int length = 0;
    for (float[] part : parts) {
      length += part.length;
    }
    float[] joined = new float[length];
    int cursor = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, joined, cursor, part.length);
      cursor += part.length;
    }
    for (float sample : joined) {
      if (!Float.isFinite(sample)) {
        throw new IllegalStateException("tiny-voca produced non-finite audio");
      }
    }
    return joined;
  }

  private static float[] limitPeak(final float[] samples) {
    float loudest = 0f;
    for (float sample : samples) {
      loudest = Math.max(loudest, Math.abs(sample));
    }
    if (loudest > PEAK) {
      float scale = PEAK / loudest;
      for (int i = 0; i < samples.length; i++) {
        samples[i] *= scale;
      }
    }
    return samples;
  }

  @Override
  public String architectureName() {
    return ARCH_TINY_VOCA;
  }

  @Override
  public int sampleRate() {
    return this.spec.sampleRate();
  }

  @Override
  public float[] synthesize(
    final CharSequence text,
    final Path espeakData,
    final MatmulRuntime runtime,
    final Random random
  ) {
    requireNonNull(text, "text");
    requireNonNull(runtime, "runtime");
    requireNonNull(random, "random");
    Controls controls = controls(text.toString(), this.defaultVoice);
    if (controls.body().isBlank()) {
      throw new IllegalArgumentException("text must not be blank");
    }
    List<float[]> parts = new ArrayList<>();
    for (String sentence : TinyVocaText.sentences(TinyVocaText.finished(controls.body()))) {
      float[] audio = this.read(sentence, controls, runtime, random);
      if (audio != null) {
        parts.add(audio);
      }
    }
    if (parts.isEmpty()) {
      throw new IllegalArgumentException(this.spec.isRussian()
        ? "в тексте нечего произнести"
        : "the text has nothing to pronounce");
    }
    return limitPeak(concat(parts));
  }

  private float[] read(
    final String piece,
    final Controls controls,
    final MatmulRuntime runtime,
    final Random random
  ) {
    Optional<TinyVocaUtterance> phones = this.front.phonemize(piece);
    if (phones.isEmpty()) {
      return null;
    }
    TinyVocaUtterance utterance = phones.get();
    int limit = Math.max(8, this.spec.maxTokens() - 12);
    if (utterance.ids().length > limit) {
      return this.split(piece, controls, runtime, random);
    }
    try {
      return this.graph.synthesize(
        utterance.ids(),
        utterance.stress(),
        controls.voice(),
        controls.speed(),
        controls.pitchShift(),
        controls.pitchRange(),
        runtime,
        random);
    } catch (UtteranceTooLongException tooLong) {
      return this.split(piece, controls, runtime, random);
    }
  }

  private float[] split(
    final String piece,
    final Controls controls,
    final MatmulRuntime runtime,
    final Random random
  ) {
    List<String> parts = TinyVocaText.shorter(piece);
    if (parts.isEmpty()) {
      throw new IllegalArgumentException("the utterance is too long or empty: split the text");
    }
    List<float[]> audio = new ArrayList<>();
    for (String part : parts) {
      float[] read = this.read(part, controls, runtime, random);
      if (read != null) {
        audio.add(read);
      }
    }
    return audio.isEmpty() ? null : concat(audio);
  }

  private String unknownVoice(final String voice) {
    return "unknown voice '" + voice + "'. This checkpoint has " +
      String.join(", ", this.spec.speakers()) + ".";
  }

  private Controls controls(final String text, final int fallbackVoice) {
    String body = text.strip();
    int voice = fallbackVoice;
    float speed = this.defaultSpeed;
    float pitch = this.defaultPitch;
    float range = this.defaultRange;
    while (true) {
      int equals = body.indexOf('=');
      int space = body.indexOf(' ');
      if (equals <= 0 || space < 0 || equals > space) {
        break;
      }
      String key = body.substring(0, equals).toLowerCase(Locale.ROOT);
      String value = body.substring(equals + 1, space);
      if (!key.equals("voice") && !key.equals("speed") && !key.equals("pitch") &&
        !key.equals("range")) {
        break;
      }
      body = body.substring(space + 1).strip();
      switch (key) {
        case "voice" -> {
          int index = this.spec.voiceIndex(value);
          if (index < 0) {
            throw new IllegalArgumentException(this.unknownVoice(value));
          }
          voice = index;
        }
        case "speed" -> speed = Math.clamp(Float.parseFloat(value), 0.25f, 4f);
        case "pitch" -> pitch = Math.clamp(Float.parseFloat(value), -2f, 2f);
        case "range" -> range = Math.clamp(Float.parseFloat(value), 0f, 3f);
        default -> {
        }
      }
    }
    return new Controls(body, voice, speed, pitch, range);
  }

  private record Controls(String body, int voice, float speed, float pitchShift, float pitchRange) {
  }
}
