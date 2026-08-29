package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInSound;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutText;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;

/**
 * Minimal Whisper speech-to-text demo (default {@code models/whisper-base}): load a Hugging Face
 * safetensors checkpoint and transcribe PCM or an uncompressed WAV file.
 *
 * <p>This is a speech model — {@link LLM#builder} then {@link LLM#generate} with
 * {@link LlmInSound} → {@link LlmModality#TEXT}. CTranslate2 {@code model.bin} folders are rejected.
 *
 * <p>Args: optional model folder (default {@code models/whisper-base}), optional WAV path. When no
 * WAV is given, {@link SynthesizeHelloWorld}'s {@code piper-hello.wav} is used if that file exists,
 * otherwise a 0.5s 440 Hz tone ({@link LlmInSound#ofPcm}). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.TranscribeHelloWorld}
 *
 * @since 1.3.0
 */
public final class TranscribeHelloWorld {

  private static final int SAMPLE_RATE = 16_000;

  private TranscribeHelloWorld() {
  }

  public static void main(final String[] args) throws Exception {
    Path modelDir = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.require(BundledModels.WHISPER_BASE);
    Optional<Path> wav = wavPath(args);
    LlmInSound input = wav.isPresent()
      ? LlmInSound.ofWav(Files.readAllBytes(wav.get()))
      : LlmInSound.ofPcm(tonePcm(), SAMPLE_RATE);

    System.out.println("Loading speech model from " + modelDir);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(modelDir);
         LLM llm = LLM.builder(model).allCpuThreads().build()) {
      System.out.println("architecture=" + model.architectureName()
        + " speech=" + model.isSpeechModel()
        + " modalities=" + model.modalities()
        + " usable=" + model.usableModalities());
      System.out.println(wav.map(path -> "WAV " + path).orElse("PCM " + SAMPLE_RATE + " Hz"));
      LlmOutText out = (LlmOutText) llm.generate(input, LlmModality.TEXT);
      String text = out.text();
      System.out.printf(Locale.ROOT, "transcript: %s%n", text.isBlank() ? "(empty)" : text);
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  static Optional<Path> wavPath(final String[] args) {
    if (args != null && args.length > 1 && args[1] != null && !args[1].isBlank()) {
      return Optional.of(Path.of(args[1]).toAbsolutePath().normalize());
    }
    Path synthesized = SynthesizeHelloWorld.defaultWavPath();
    return Files.isRegularFile(synthesized) ? Optional.of(synthesized) : Optional.empty();
  }

  private static float[] tonePcm() {
    int n = SAMPLE_RATE / 2;
    float[] pcm = new float[n];
    for (int i = 0; i < n; i++) {
      pcm[i] = (float) (0.2 * Math.sin(2.0 * Math.PI * 440.0 * i / SAMPLE_RATE));
    }
    return pcm;
  }
}
