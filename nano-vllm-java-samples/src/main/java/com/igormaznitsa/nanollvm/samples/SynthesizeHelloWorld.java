package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Minimal Piper text-to-speech demo. Default voice is {@code models/piper-en-lessac-medium}
 * if present, otherwise {@code models/piper-ru-irina-medium}. Writes uncompressed WAV bytes.
 *
 * <p>This is a synthesis model — {@link LLM#builder} then
 * {@link LLM#generate} with {@link LlmInText} → {@link LlmModality#AUDIO}. espeak-ng-data is
 * optional ({@link LlmOptionalData#ESPEAK_DATA}, or {@code espeak-ng-data/} next to the voice); a
 * missing folder is ignored.
 *
 * <p>Args: optional model folder, optional text, optional WAV output path. From the repository
 * root: {@code mvn -pl nano-vllm-java-samples -q exec:java
 * -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.SynthesizeHelloWorld}
 *
 * @since 1.3.0
 */
public final class SynthesizeHelloWorld {

  private SynthesizeHelloWorld() {
  }

  public static void main(final String[] args) throws Exception {
    Path modelDir = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.requirePiperVoice();
    String text = args != null && args.length > 1 && args[1] != null && !args[1].isBlank()
      ? args[1]
      : BundledModels.defaultPiperText(modelDir);
    Path wavOut = args != null && args.length > 2 && args[2] != null && !args[2].isBlank()
      ? Path.of(args[2]).toAbsolutePath().normalize()
      : Path.of("piper-hello.wav").toAbsolutePath().normalize();
    Path espeak = modelDir.resolve("espeak-ng-data");

    System.out.println("Loading synthesis model from " + modelDir);
    long started = System.currentTimeMillis();
    try (LlmModel model = LlmModelFactory.open(modelDir)
      .optionalData(LlmOptionalData.ESPEAK_DATA, espeak)
      .make();
         LLM llm = LLM.builder(model).build()) {
      System.out.println("architecture=" + model.architectureName()
        + " synthesis=" + model.isSynthesisModel()
        + " modalities=" + model.modalities()
        + " usable=" + model.usableModalities());
      LlmOutSoundData sound = (LlmOutSoundData) llm.generate(LlmInText.of(text), LlmModality.AUDIO);
      Files.write(wavOut, sound.wav());
      System.out.printf(
        Locale.ROOT,
        "wrote %d bytes (%d Hz) to %s%n",
        sound.wav().length,
        sound.sampleRate(),
        wavOut);
    } finally {
      System.out.println("Time taken: " + (System.currentTimeMillis() - started) + "ms");
    }
  }
}
