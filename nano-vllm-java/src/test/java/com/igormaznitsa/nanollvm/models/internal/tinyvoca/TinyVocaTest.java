package com.igormaznitsa.nanollvm.models.internal.tinyvoca;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.TinyVocaSpec;
import com.igormaznitsa.nanollvm.models.LlmInText;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.LlmOutSoundData;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TinyVocaTest {

  private static final String ENGLISH = """
    {
      "model": {
        "sample_rate": 24000, "hop": 8, "channels": 8, "heads": 2, "feedforward": 16,
        "kernel": 3, "encoder_layers": 1, "prosody_layers": 1, "acoustic_layers": 1,
        "predictor_channels": 8, "synth_fft": 16, "synth_hop": 4, "noise_fft": 8, "noise_hop": 4,
        "cep_causal": 2, "cep_anti": 1, "noise_cep": 2, "harmonic_limit": 0.47,
        "max_tokens": 32, "max_frames": 64, "f0_min": 50, "f0_max": 600,
        "pitch_reference_hz": 200, "speakers": ["F2", "M2"]
      },
      "symbols": ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>", ".", "h", "ə", "l", "oʊ"]
    }
    """;

  private static final String RUSSIAN = """
    {
      "config": {
        "sample_rate": 24000, "hop": 8, "channels": 8, "heads": 2, "feedforward": 16,
        "kernel": 3, "encoder_layers": 1, "prosody_layers": 1, "acoustic_layers": 1,
        "predictor_channels": 8, "synth_fft": 16, "synth_hop": 4, "noise_fft": 8, "noise_hop": 4,
        "cep_causal": 2, "cep_anti": 1, "noise_cep": 2, "speakers": ["F", "M"]
      },
      "symbols": ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>", "mʲ", "i", "r", "a", "ɨ", "ʂ", "v", "ɡ", "ɐ", "u"]
    }
    """;

  private static List<String> symbols(final TinyVocaSpec spec, final TinyVocaUtterance utterance) {
    List<String> names = new ArrayList<>();
    for (int id : utterance.ids()) {
      names.add(spec.symbols().get(id));
    }
    return names;
  }

  private static void writeSafetensors(final Path file, final Map<String, Tensor> weights)
    throws Exception {
    Files.createDirectories(file.getParent().resolve("phonemizer"));
    StringBuilder header = new StringBuilder("{");
    List<byte[]> chunks = new ArrayList<>();
    long cursor = 0;
    boolean first = true;
    for (var entry : weights.entrySet()) {
      float[] data = entry.getValue().toFloatArray();
      ByteBuffer raw = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
      raw.asFloatBuffer().put(data);
      if (!first) {
        header.append(',');
      }
      first = false;
      header.append('"').append(entry.getKey()).append("\":{\"dtype\":\"F32\",\"shape\":[");
      int[] shape = entry.getValue().shape();
      for (int i = 0; i < shape.length; i++) {
        if (i > 0) {
          header.append(',');
        }
        header.append(shape[i]);
      }
      long next = cursor + raw.capacity();
      header.append("],\"data_offsets\":[").append(cursor).append(',').append(next).append("]}");
      chunks.add(raw.array());
      cursor = next;
    }
    header.append('}');
    byte[] json = header.toString().getBytes(UTF_8);
    ByteBuffer prefix = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    prefix.putLong(json.length);
    Files.write(file, prefix.array());
    Files.write(file, json, java.nio.file.StandardOpenOption.APPEND);
    for (byte[] chunk : chunks) {
      Files.write(file, chunk, java.nio.file.StandardOpenOption.APPEND);
    }
  }

  @Test
  void readsEnglishLexiconAndRussianLetters() {
    TinyVocaSpec english = TinyVocaSpec.parse(ENGLISH);
    assertFalse(english.isRussian());
    assertEquals(0, english.voiceIndex("F2"));
    assertEquals(1, english.voiceIndex("m2"));
    TinyVocaUtterance hello = DizioneFront.fromLexicon(english, "hello\th ə ˈl oʊ\n")
      .phonemize("Hello")
      .orElseThrow();
    assertEquals(List.of("<bos>", "h", "ə", "l", "oʊ", "<eos>"), symbols(english, hello));
    assertEquals(1, hello.stress()[4]);

    TinyVocaSpec russian = TinyVocaSpec.parse(RUSSIAN);
    assertTrue(russian.isRussian());
    TinyVocaUtterance world = new RussianVocaFront(russian, null).phonemize("мир").orElseThrow();
    assertEquals(List.of("<bos>", "mʲ", "i", "r", "<eos>"), symbols(russian, world));
    assertEquals(1, world.stress()[2]);
    TinyVocaUtterance uphill =
      new RussianVocaFront(russian, null).phonemize("в гору").orElseThrow();
    assertEquals(List.of("<bos>", "v", "ɡ", "ɐ", "r", "u", "<eos>"), symbols(russian, uphill));
    assertEquals(1, uphill.stress()[5]);
  }

  @Test
  void randomWeightsSpeakAFiniteClip(@TempDir final Path dir) throws Exception {
    TinyVocaSpec spec = TinyVocaSpec.parse(ENGLISH);
    Map<String, Tensor> weights = new LinkedHashMap<>();
    Random random = new Random(1);
    TinyVocaGraph.parameterShapes(spec).forEach((name, shape) -> {
      int count = 1;
      for (int dim : shape) {
        count *= dim;
      }
      float[] data = new float[count];
      if (name.contains("norm") && name.endsWith(".weight")) {
        java.util.Arrays.fill(data, 1f);
      } else if (!name.startsWith("duration.out")) {
        for (int i = 0; i < data.length; i++) {
          data[i] = (float) (random.nextGaussian() * 0.02);
        }
      }
      weights.put(name, Tensor.of(data, shape));
    });
    float[] audio = new TinyVocaGraph(spec, weights).synthesize(
      new int[] {1, 6, 7, 8, 9, 2},
      new int[] {0, 0, 0, 0, 1, 0},
      0, 1f, 0f, 1f,
      com.igormaznitsa.nanollvm.tensor.MatmulRuntime.sequential(),
      new Random(2));
    assertTrue(audio.length > 0 && audio.length % spec.hop() == 0);
    for (float sample : audio) {
      assertTrue(Float.isFinite(sample));
    }

    Files.createDirectories(dir.resolve("phonemizer"));
    Files.writeString(dir.resolve("config.json"), ENGLISH, UTF_8);
    Files.writeString(dir.resolve("phonemizer").resolve("lexicon.tsv"), "hello\th ə ˈl oʊ\n",
      UTF_8);
    writeSafetensors(dir.resolve("model.safetensors"), weights);
    assertTrue(ModelSupport.isTinyVocaCheckpoint(dir));
    try (LlmModel model = LlmModelFactory.open(dir)
      .optionalData(LlmOptionalData.SYNTHESIS_VOICE, "M2")
      .optionalData(LlmOptionalData.SYNTHESIS_SPEED, 1.25f)
      .optionalData(LlmOptionalData.SYNTHESIS_PITCH, 0.1f)
      .optionalData(LlmOptionalData.SYNTHESIS_RANGE, 1.5f)
      .make()) {
      assertEquals(WeightNames.ARCH_TINY_VOCA, model.architectureName());
      assertEquals("M2", model.optionalData(LlmOptionalData.SYNTHESIS_VOICE).orElseThrow());
      assertEquals(1.25f, model.optionalData(LlmOptionalData.SYNTHESIS_SPEED).orElseThrow());
      assertEquals(0.1f, model.optionalData(LlmOptionalData.SYNTHESIS_PITCH).orElseThrow());
      assertEquals(1.5f, model.optionalData(LlmOptionalData.SYNTHESIS_RANGE).orElseThrow());
      assertTrue(model.hfConfig().isTinyVoca());
      try (LLM llm = LLM.builder(model).build()) {
        LlmOutSoundData sound = llm.generate(LlmInText.of("voice=F2 hello"), LlmModality.AUDIO);
        assertEquals(24_000, sound.sampleRate());
        assertTrue(sound.wav().length > 44);
        assertEquals("RIFF", new String(sound.wav(), 0, 4, UTF_8));
      }
    }
    assertThrows(IllegalArgumentException.class, () -> new TinyVocaForTts(
      spec, new TinyVocaGraph(spec, weights), DizioneFront.fromLexicon(spec, "hello\th ə ˈl oʊ\n"),
      "nope", 1f, 0f, 1f));
  }
}
