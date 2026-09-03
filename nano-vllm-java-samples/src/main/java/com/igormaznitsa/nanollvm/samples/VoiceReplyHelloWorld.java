package com.igormaznitsa.nanollvm.samples;

import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInSound;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.LlmOutEmbedding;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import com.igormaznitsa.nanollvm.models.LlmOutText;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier.LabeledText;
import com.igormaznitsa.nanollvm.samples.utils.EmbeddingClassifier.Prediction;
import com.igormaznitsa.nanollvm.samples.utils.SampleChatPrompts;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Stream;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineEvent;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.TargetDataLine;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Linear voice desk: load Whisper + a BERT encoder + a chat model + Piper, then
 * speech → transcript → few-shot mood of the <em>words</em> → chat reply → spoken WAV
 * (played through {@code javax.sound.sampled} when a mixer is present).
 *
 * <p>Mood is classified on the transcript, not on pitch or tone. Each {@link LLM} gets
 * {@link LLM.Builder#dedicatedMatmulPool()} so a server can run several of these engines
 * at once (one engine per in-flight call; share the four {@link LlmModel}s).
 *
 * <p>Needs four local checkpoints (tiny Whisper / E5 / Gemma3-270M / Lessac if present,
 * with fallbacks). No WAV argument: records 15 seconds from the default microphone
 * ({@code javax.sound.sampled} {@link TargetDataLine}), then trims leading and trailing
 * silence; if that line is missing, Piper
 * speaks a canned happy line instead. Small chat checkpoints may answer in a word. Args: optional input WAV, optional output path (default {@code voice-reply.wav}).
 * From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.VoiceReplyHelloWorld}
 *
 * @since 1.3.0
 */
public final class VoiceReplyHelloWorld {

  private static final String CHAT_SYSTEM_EN = """
    You are speaking aloud. Reply in one or two short sentences.
    Match the listener's mood. Do not name the mood label.
    """.strip();
  private static final String CHAT_SYSTEM_RU = """
    Ты говоришь вслух. Ответь одной-двумя короткими фразами.
    Подстройся под настроение слушателя. Не называй метку настроения.
    """.strip();

  private static final String QUERY_PREFIX = "query: ";
  private static final int CHAT_MAX_TOKENS = 96;
  private static final int RECORD_SECONDS = 15;
  private static final int SILENCE_PEAK_FLOOR = 256;
  private static final int SILENCE_PAD_MILLIS = 200;

  private VoiceReplyHelloWorld() {
  }

  public static void main(final String[] args) throws Exception {
    Path whisperDir = BundledModels.requireWhisper();
    Path embedDir = BundledModels.requireEmbeddingEncoder();
    Path chatDir = BundledModels.requireChatDemo();
    Path piperDir = BundledModels.requirePiperVoice();
    boolean english = BundledModels.isEnglishPiperVoice(piperDir);
    boolean e5Prefix = BundledModels.usesE5QueryPrefix(embedDir);
    Optional<Path> inputWav = cliPath(args, 0);
    Path outputWav = cliPath(args, 1)
      .orElse(Path.of("voice-reply.wav").toAbsolutePath().normalize());

    System.out.println("whisper=" + whisperDir);
    System.out.println("embed=" + embedDir);
    System.out.println("chat=" + chatDir);
    System.out.println("piper=" + piperDir);
    System.out.println("Each engine uses a dedicated matmul pool; one LLM is not concurrent.");

    long started = System.currentTimeMillis();
    try (LlmModel whisperModel = LlmModelFactory.make(whisperDir);
         LlmModel embedModel = LlmModelFactory.make(embedDir);
         LlmModel chatModel = LlmModelFactory.make(chatDir);
         LlmModel piperModel = LlmModelFactory.open(piperDir)
           .optionalData(LlmOptionalData.ESPEAK_DATA, piperDir.resolve("espeak-ng-data"))
           .make();
         LLM stt = pooledEngine(whisperModel);
         LLM encoder = pooledEngine(embedModel);
         LLM talk = chatEngine(chatModel, english);
         LLM tts = pooledEngine(piperModel)) {
      byte[] userWav = userUtterance(inputWav, tts, english);
      System.out.println("Starting process...");
      String transcript = requireHeard(transcribe(stt, userWav, speechLocale(english)));
      Prediction mood = classifyMood(encoder, transcript, english, e5Prefix);
      String reply = answer(talk, mood.label(), transcript, english);
      byte[] spoken = synthesize(tts, requireSpoken(reply));
      Files.write(outputWav, spoken);

      System.out.println("transcript: " + transcript);
      System.out.println("mood: " + mood.label() + "  " + mood.scores());
      System.out.println("reply: " + reply);
      System.out.printf(Locale.ROOT, "wrote %d bytes to %s%n", spoken.length, outputWav);
      System.out.println("playing reply...");
      System.out.println(playWav(spoken)
        ? "played reply"
        : "no Java Sound output; WAV is on disk");
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  static List<LabeledText> moodExamples(final boolean english) {
    return english ? List.of(
      new LabeledText("happy", "I am so glad we finished this today."),
      new LabeledText("happy", "What a wonderful surprise, I love it."),
      new LabeledText("happy", "This is the best news I have heard all week."),
      new LabeledText("sad", "I feel terrible about what happened."),
      new LabeledText("sad", "This is the worst day I can remember."),
      new LabeledText("sad", "I miss them and I cannot stop thinking about it."),
      new LabeledText("angry", "I cannot believe they ignored us again."),
      new LabeledText("angry", "Stop this nonsense right now."),
      new LabeledText("angry", "I am furious about this delay."),
      new LabeledText("calm", "The meeting is at three in the afternoon."),
      new LabeledText("calm", "Please send the document when you can."),
      new LabeledText("calm", "The train arrives on platform four.")
    ) : List.of(
      new LabeledText("happy", "Я так рад, что мы наконец закончили это сегодня."),
      new LabeledText("happy", "Какой чудесный сюрприз, мне это очень нравится."),
      new LabeledText("happy", "Это лучшая новость за всю неделю."),
      new LabeledText("sad", "Мне очень тяжело из-за того, что случилось."),
      new LabeledText("sad", "Это худший день, который я помню."),
      new LabeledText("sad", "Я скучаю и не могу перестать об этом думать."),
      new LabeledText("angry", "Не могу поверить, что нас снова проигнорировали."),
      new LabeledText("angry", "Немедленно прекратите этот вздор."),
      new LabeledText("angry", "Я в ярости из-за этой задержки."),
      new LabeledText("calm", "Встреча назначена на три часа дня."),
      new LabeledText("calm", "Пожалуйста, пришлите документ, когда сможете."),
      new LabeledText("calm", "Поезд прибывает на четвёртую платформу.")
    );
  }

  static String cannedUserSpeech(final boolean english) {
    return english
      ? "I am so glad we finally shipped the release today."
      : "Я так рад, что мы наконец выпустили релиз сегодня.";
  }

  static String chatUserTurn(final String mood, final String transcript, final boolean english) {
    requireNonNull(mood, "mood");
    requireNonNull(transcript, "transcript");
    return english
      ? """
      Mood: %s
      They said: %s
      Reply in one short sentence. Not just a mood word.
      """.formatted(mood, transcript)
      : """
      Настроение: %s
      Они сказали: %s
      Ответь одним коротким предложением. Не одним словом настроения.
      """.formatted(mood, transcript);
  }

  private static byte[] userUtterance(
    @SuppressWarnings("OptionalUsedAsFieldOrParameterType") final Optional<Path> inputWav,
    final LLM tts,
    final boolean english
  ) throws IOException {
    if (inputWav.isPresent()) {
      return Files.readAllBytes(inputWav.get());
    }
    return microphoneOrCanned(tts, english);
  }

  private static byte[] microphoneOrCanned(final LLM tts, final boolean english) {
    System.out.printf(Locale.ROOT, "No WAV argument: speak now (%d seconds)...%n", RECORD_SECONDS);
    Optional<byte[]> captured = recordDefaultInput(RECORD_SECONDS);
    if (captured.isEmpty()) {
      System.out.println("No microphone: speaking a canned line with Piper, then transcribing it.");
      return synthesize(tts, cannedUserSpeech(english));
    }
    byte[] wav = captured.get();
    if (wav.length <= 44) {
      System.out.println("Heard only silence.");
      throw new IllegalStateException(
        "Heard only silence — speak into the default microphone, or pass an uncompressed spoken WAV as the first argument");
    }
    System.out.printf(
      Locale.ROOT,
      "Recorded %d bytes from the default microphone (silence trimmed).%n",
      wav.length);
    return wav;
  }

  private static LLM pooledEngine(final LlmModel model) {
    return LLM.builder(model).dedicatedMatmulPool().build();
  }

  private static LLM chatEngine(final LlmModel model, final boolean english) {
    LLM.Builder builder = LLM.builder(model)
      .dedicatedMatmulPool()
      .maxModelLen(2048)
      .sampling(SampleChatPrompts.samplingForDemo(model.tokenizer(), CHAT_MAX_TOKENS));
    if (model.tokenizer().isTurnBasedChat()) {
      builder.noSystemPrompt();
    } else {
      builder.systemPrompt(english ? CHAT_SYSTEM_EN : CHAT_SYSTEM_RU);
    }
    return builder.build();
  }

  private static String transcribe(final LLM stt, final byte[] wav, final Locale language) {
    LlmOutText text = stt.generate(LlmInSound.ofWav(wav, language), LlmModality.TEXT);
    return text.text();
  }

  private static byte[] synthesize(final LLM tts, final String text) {
    LlmOutSoundData sound = tts.generate(LlmInText.of(text), LlmModality.AUDIO);
    return sound.wav();
  }

  private static Prediction classifyMood(
    final LLM encoder,
    final String transcript,
    final boolean english,
    final boolean e5Prefix
  ) {
    EmbeddingClassifier.Trainer trainer = EmbeddingClassifier.trainer();

    for (LabeledText example : moodExamples(english)) {
      trainer.add(example.label(), embed(encoder, embedInput(example.text(), e5Prefix)));
    }

    return trainer.fit().classify(embed(encoder, embedInput(transcript, e5Prefix)));
  }

  private static String answer(
    final LLM talk,
    final String mood,
    final String transcript,
    final boolean english
  ) {
    return talk.chat()
      .enableThinking(false)
      .send(chatUserTurn(mood, transcript, english))
      .answer();
  }

  private static float[] embed(final LLM encoder, final String text) {
    LlmOutEmbedding embedding = encoder.generate(LlmInText.of(text), LlmModality.EMBEDDING);
    return embedding.vector();
  }

  private static String embedInput(final String text, final boolean e5Prefix) {
    return e5Prefix ? QUERY_PREFIX + text : text;
  }

  private static Locale speechLocale(final boolean english) {
    return english ? Locale.ENGLISH : Locale.forLanguageTag("ru");
  }

  private static String requireHeard(final String transcript) {
    if (transcript == null || transcript.isBlank()) {
      throw new IllegalStateException(
        "Whisper heard nothing — speak into the default microphone, or pass an uncompressed spoken WAV as the first argument");
    }
    return transcript.strip();
  }

  private static String requireSpoken(final String reply) {
    if (reply == null || reply.isBlank()) {
      throw new IllegalStateException("chat model returned an empty reply");
    }
    return reply.strip();
  }

  static boolean playWav(final byte[] wav) {
    requireNonNull(wav, "wav");
    if (wav.length == 0) {
      return false;
    }
    try (AudioInputStream encoded = AudioSystem.getAudioInputStream(new ByteArrayInputStream(wav));
         AudioInputStream pcm = playablePcm(encoded);
         Clip clip = acquireClip(pcm.getFormat())) {
      return playClip(clip, pcm);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return false;
    } catch (UnsupportedAudioFileException | LineUnavailableException | IOException
             | IllegalArgumentException | SecurityException e) {
      return false;
    }
  }

  private static AudioInputStream playablePcm(final AudioInputStream encoded) {
    AudioFormat source = encoded.getFormat();
    return Stream.of(
        source,
        pcm16le(source.getSampleRate(), source.getChannels()),
        pcm16le(44_100f, 1))
      .filter(format -> format.equals(source) || AudioSystem.isConversionSupported(format, source))
      .filter(VoiceReplyHelloWorld::supportsClip)
      .findFirst().filter(format -> !format.equals(source))
      .map(format -> AudioSystem.getAudioInputStream(format, encoded)).orElse(encoded);
  }

  private static boolean supportsClip(final AudioFormat format) {
    return AudioSystem.isLineSupported(new DataLine.Info(Clip.class, format));
  }

  private static boolean supportsTarget(final AudioFormat format) {
    return AudioSystem.isLineSupported(new DataLine.Info(TargetDataLine.class, format));
  }

  static Optional<byte[]> recordDefaultInput(final int seconds) {
    if (seconds < 1) {
      return Optional.empty();
    }
    return captureFormat().flatMap(format -> captureOrEmpty(format, seconds));
  }

  private static Optional<AudioFormat> captureFormat() {
    return Stream.of(
        pcm16le(16_000f, 1),
        pcm16le(44_100f, 1),
        pcm16le(48_000f, 1),
        pcm16le(16_000f, 2),
        pcm16le(44_100f, 2),
        pcm16le(48_000f, 2))
      .filter(VoiceReplyHelloWorld::supportsTarget)
      .findFirst();
  }

  private static Optional<byte[]> captureOrEmpty(final AudioFormat format, final int seconds) {
    try (TargetDataLine line = AudioSystem.getTargetDataLine(format)) {
      byte[] wav = captureWav(line, format, seconds);
      return Optional.of(wav);
    } catch (LineUnavailableException | IllegalArgumentException | SecurityException e) {
      return Optional.empty();
    }
  }

  private static byte[] captureWav(
    final TargetDataLine line,
    final AudioFormat format,
    final int seconds
  ) throws LineUnavailableException {
    line.open(format);
    line.flush();
    line.start();
    byte[] pcm = readPcm(line, bytesForDuration(format, seconds));
    line.stop();
    line.flush();
    byte[] mono = format.getChannels() <= 1 ? pcm : downmixStereoPcm16(pcm);
    int sampleRate = Math.round(format.getSampleRate());
    byte[] trimmed = trimSilencePcm16(mono, sampleRate);
    if (trimmed.length < 2) {
      return new byte[0];
    }
    return pcm16MonoWav(sampleRate, trimmed);
  }

  private static int bytesForDuration(final AudioFormat format, final int seconds) {
    int frameSize = format.getFrameSize();
    if (frameSize < 1) {
      throw new IllegalArgumentException("frame size must be >= 1");
    }
    return Math.multiplyExact(seconds,
      Math.multiplyExact(Math.round(format.getSampleRate()), frameSize));
  }

  private static byte[] readPcm(final TargetDataLine line, final int bytesToRead) {
    byte[] pcm = new byte[bytesToRead];
    int frameSize = Math.max(1, line.getFormat().getFrameSize());
    int chunk = Math.max(frameSize, line.getBufferSize() / 8);
    int offset = 0;
    while (offset < bytesToRead) {
      int n = line.read(pcm, offset, Math.min(chunk, bytesToRead - offset));
      if (n <= 0) {
        break;
      }
      offset += n;
    }
    return offset == pcm.length ? pcm : Arrays.copyOf(pcm, Math.max(0, offset));
  }

  static byte[] trimSilencePcm16(final byte[] pcm, final int sampleRate) {
    requireNonNull(pcm, "pcm");
    if (sampleRate < 1) {
      throw new IllegalArgumentException("sampleRate must be >= 1");
    }
    int frames = pcm.length / 2;
    if (frames == 0) {
      return new byte[0];
    }
    int peak = peakAbsolute(pcm, frames);
    if (peak < SILENCE_PEAK_FLOOR) {
      return new byte[0];
    }
    int threshold = Math.max(SILENCE_PEAK_FLOOR, peak / 40);
    int first = firstAudibleFrame(pcm, frames, threshold);
    int last = lastAudibleFrame(pcm, frames, threshold);
    if (first < 0 || last < first) {
      return new byte[0];
    }
    int pad = Math.max(0, sampleRate * SILENCE_PAD_MILLIS / 1_000);
    int start = Math.max(0, first - pad);
    int end = Math.min(frames, last + 1 + pad);
    if (start == 0 && end == frames) {
      return pcm.length == frames * 2 ? pcm : Arrays.copyOf(pcm, frames * 2);
    }
    return Arrays.copyOfRange(pcm, start * 2, end * 2);
  }

  private static int peakAbsolute(final byte[] pcm, final int frames) {
    ByteBuffer buf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
    int peak = 0;
    for (int i = 0; i < frames; i++) {
      peak = Math.max(peak, Math.abs(buf.getShort()));
    }
    return peak;
  }

  private static int firstAudibleFrame(final byte[] pcm, final int frames, final int threshold) {
    ByteBuffer buf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < frames; i++) {
      if (Math.abs(buf.getShort()) >= threshold) {
        return i;
      }
    }
    return -1;
  }

  private static int lastAudibleFrame(final byte[] pcm, final int frames, final int threshold) {
    ByteBuffer buf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = frames - 1; i >= 0; i--) {
      if (Math.abs(buf.getShort(i * 2)) >= threshold) {
        return i;
      }
    }
    return -1;
  }

  static byte[] downmixStereoPcm16(final byte[] interleaved) {
    requireNonNull(interleaved, "interleaved");
    int frames = interleaved.length / 4;
    byte[] mono = new byte[frames * 2];
    ByteBuffer in = ByteBuffer.wrap(interleaved).order(ByteOrder.LITTLE_ENDIAN);
    ByteBuffer out = ByteBuffer.wrap(mono).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < frames; i++) {
      out.putShort((short) ((in.getShort() + in.getShort()) / 2));
    }
    return mono;
  }

  static byte[] pcm16MonoWav(final int sampleRate, final byte[] pcm) {
    requireNonNull(pcm, "pcm");
    if (sampleRate < 1) {
      throw new IllegalArgumentException("sampleRate must be >= 1");
    }
    if ((pcm.length & 1) != 0) {
      throw new IllegalArgumentException("PCM16 byte length must be even");
    }
    ByteBuffer buf = ByteBuffer.allocate(44 + pcm.length).order(ByteOrder.LITTLE_ENDIAN);
    buf.put("RIFF".getBytes(US_ASCII));
    buf.putInt(36 + pcm.length);
    buf.put("WAVE".getBytes(US_ASCII));
    buf.put("fmt ".getBytes(US_ASCII));
    buf.putInt(16);
    buf.putShort((short) 1);
    buf.putShort((short) 1);
    buf.putInt(sampleRate);
    buf.putInt(sampleRate * 2);
    buf.putShort((short) 2);
    buf.putShort((short) 16);
    buf.put("data".getBytes(US_ASCII));
    buf.putInt(pcm.length);
    buf.put(pcm);
    return buf.array();
  }

  private static AudioFormat pcm16le(final float sampleRate, final int channels) {
    int n = Math.max(1, channels);
    return new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, sampleRate, 16, n, n * 2, sampleRate,
      false);
  }

  private static Clip acquireClip(final AudioFormat format) throws LineUnavailableException {
    return (Clip) AudioSystem.getLine(new DataLine.Info(Clip.class, format));
  }

  private static boolean playClip(final Clip clip, final AudioInputStream pcm)
    throws IOException, LineUnavailableException, InterruptedException {
    CountDownLatch finished = new CountDownLatch(1);
    clip.addLineListener(event -> {
      if (LineEvent.Type.STOP.equals(event.getType()) ||
        LineEvent.Type.CLOSE.equals(event.getType())) {
        finished.countDown();
      }
    });
    clip.open(pcm);
    if (clip.getFrameLength() <= 0) {
      return false;
    }
    clip.start();
    if (!finished.await(playbackDeadlineMillis(clip), MILLISECONDS)) {
      clip.stop();
    }
    return true;
  }

  private static long playbackDeadlineMillis(final Clip clip) {
    int frames = clip.getFrameLength();
    float rate = clip.getFormat().getFrameRate();
    if (frames <= 0 || !Float.isFinite(rate) || rate <= 0f) {
      return 5_000L;
    }
    return Math.min(120_000L, (long) (frames / rate * 1_000.0) + 2_000L);
  }

  private static Optional<Path> cliPath(final String[] args, final int index) {
    if (args == null || args.length <= index || args[index] == null || args[index].isBlank()) {
      return Optional.empty();
    }
    return Optional.of(Path.of(args[index]).toAbsolutePath().normalize());
  }
}
