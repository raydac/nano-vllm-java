package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInSound;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutText;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Minimal Whisper speech-to-text demo (default {@code models/whisper-base}): load a Hugging Face
 * safetensors checkpoint and transcribe an uncompressed WAV file.
 *
 * <p>This is a speech model — {@link LLM#builder} then {@link LLM#generate} with
 * {@link LlmInSound} → {@link LlmModality#TEXT}. CTranslate2 {@code model.bin} folders are rejected.
 *
 * <p>Args: optional model folder (default {@code models/whisper-base}), optional WAV path. When no
 * WAV is given, a 0.5s 440 Hz tone is synthesized (the transcript is usually empty or noise).
 * From the repository root:
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
    Path wav = args != null && args.length > 1 && args[1] != null && !args[1].isBlank()
      ? Path.of(args[1]).toAbsolutePath().normalize()
      : writeToneWav();

    System.out.println("Loading speech model from " + modelDir);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.make(modelDir);
         LLM llm = LLM.builder(model).build()) {
      System.out.println("architecture=" + model.architectureName()
        + " speech=" + model.isSpeechModel()
        + " modalities=" + model.modalities()
        + " usable=" + model.usableModalities());
      System.out.println("WAV " + wav);
      LlmOutText out = (LlmOutText) llm.generate(
        LlmInSound.ofWav(Files.readAllBytes(wav)), LlmModality.TEXT);
      String text = out.text();
      System.out.printf(Locale.ROOT, "transcript: %s%n", text.isBlank() ? "(empty)" : text);
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }

  private static Path writeToneWav() throws Exception {
    int n = SAMPLE_RATE / 2;
    byte[] pcm = new byte[n * 2];
    ByteBuffer buf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < n; i++) {
      double sample = 0.2 * Math.sin(2.0 * Math.PI * 440.0 * i / SAMPLE_RATE);
      buf.putShort((short) Math.round(sample * 32767.0));
    }
    byte[] wav = wav16Mono(pcm, SAMPLE_RATE);
    Path file = Files.createTempFile("nanollvm-whisper-tone-", ".wav");
    Files.write(file, wav);
    System.out.println("No WAV argument: wrote a 440 Hz tone to " + file);
    return file;
  }

  private static byte[] wav16Mono(final byte[] pcm, final int sampleRate) {
    ByteBuffer header = ByteBuffer.allocate(44 + pcm.length).order(ByteOrder.LITTLE_ENDIAN);
    header.put("RIFF".getBytes(StandardCharsets.US_ASCII));
    header.putInt(36 + pcm.length);
    header.put("WAVE".getBytes(StandardCharsets.US_ASCII));
    header.put("fmt ".getBytes(StandardCharsets.US_ASCII));
    header.putInt(16);
    header.putShort((short) 1);
    header.putShort((short) 1);
    header.putInt(sampleRate);
    header.putInt(sampleRate * 2);
    header.putShort((short) 2);
    header.putShort((short) 16);
    header.put("data".getBytes(StandardCharsets.US_ASCII));
    header.putInt(pcm.length);
    header.put(pcm);
    return header.array();
  }
}
