package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.LlmModalities;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOptionalData;
import com.igormaznitsa.nanollvm.models.ModelSupport;
import com.igormaznitsa.nanollvm.models.internal.ConvLayout;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.models.internal.audio.WavPcm;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxGraphBundle;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

final class PiperLoadTest {

  @Test
  void loadsPiperAndSynthesizesWav() throws Exception {
    Path modelPath = OptionalModelAssumptions.requirePiper();
    Path espeak = modelPath.resolve("espeak-ng-data");

    try (LlmModel model = LlmModelFactory.open(modelPath)
      .optionalData(LlmOptionalData.ESPEAK_DATA, espeak)
      .make()) {
      assertTrue(model.isSynthesisModel());
      assertFalse(model.isSpeechModel());
      assertFalse(model.isEmbeddingModel());
      assertEquals(WeightNames.ARCH_PIPER, model.architectureName());
      assertEquals(LlmModalities.TEXT_TO_AUDIO, model.usableModalities());
      assertEquals(
        espeak.toAbsolutePath().normalize(),
        model.optionalData(LlmOptionalData.ESPEAK_DATA).orElseThrow());

      IllegalStateException embed = assertThrows(
        IllegalStateException.class, () -> model.embed("hello"));
      assertTrue(embed.getMessage().contains("text-to-speech"));

      try (LLM llm = LLM.builder(model).build()) {
        IllegalStateException chat = assertThrows(IllegalStateException.class, llm::chat);
        assertEquals(
          ModelSupport.synthesisEngineMisuseMessage(model.architectureName()), chat.getMessage());
        assertEquals(0, llm.config().numKvcacheBlocks());

        byte[] wav = llm.synthesize(
          modelPath.getFileName().toString().contains("-en-") ? "Hello" : "Привет");
        assertTrue(wav.length > 44);
        assertEquals("RIFF", new String(wav, 0, 4, StandardCharsets.US_ASCII));
        assertEquals("WAVE", new String(wav, 8, 4, StandardCharsets.US_ASCII));

        WavPcm.MonoPcm pcm = WavPcm.read(wav);
        float peak = 0f;
        for (float sample : pcm.samples()) {
          peak = Math.max(peak, Math.abs(sample));
        }
        assertTrue(peak > 0.05f, "synthesized peak was too quiet: " + peak);
        assertTrue(pcm.samples().length > 4000, "synthesized clip was too short");
      }
    }
  }

  @Test
  void parsesIrinaVocoderResBlock2Dilations() throws Exception {
    Path modelPath = OptionalModelAssumptions.requirePiperRussian();
    Path onnx;
    try (Stream<Path> files = Files.list(modelPath)) {
      onnx = files
        .filter(path -> path.getFileName().toString().endsWith(".onnx"))
        .findFirst()
        .orElseThrow();
    }
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(onnx);
    assertEquals(2, graph.convDilations().get("dec.resblocks.0.convs.1.weight"));
    assertEquals(2, graph.convDilations().get("dec.resblocks.1.convs.0.weight"));
    assertEquals(6, graph.convDilations().get("dec.resblocks.1.convs.1.weight"));
    assertEquals(3, graph.convDilations().get("dec.resblocks.2.convs.0.weight"));
    assertEquals(12, graph.convDilations().get("dec.resblocks.2.convs.1.weight"));
  }

  @Test
  void parsesUpsamplerGeometryFromOnnxGraph() throws Exception {
    Path modelPath = OptionalModelAssumptions.requirePiper();
    Path onnx;
    try (Stream<Path> files = Files.list(modelPath)) {
      onnx = files
        .filter(path -> path.getFileName().toString().endsWith(".onnx"))
        .findFirst()
        .orElseThrow();
    }
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(onnx);
    ConvLayout up0 = graph.convLayouts().get("dec.ups.0.weight");
    assertNotNull(up0);
    assertEquals(ConvLayout.Kind.CONV_TRANSPOSE, up0.kind());
    assertTrue(up0.stride() >= 2, "upsampler stride was " + up0.stride());
    ConvLayout residual = graph.convLayouts().get("dec.resblocks.0.convs.0.weight");
    assertNotNull(residual);
    assertEquals(ConvLayout.Kind.CONV, residual.kind());
  }
}
