package com.igormaznitsa.nanollvm.models.llmarch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import org.junit.jupiter.api.Test;

final class ArchitectureProcessorTest {

  @Test
  void registryReturnsOneProcessorPerFamily() {
    assertSame(Qwen3Processor.INSTANCE, ArchitectureProcessors.of(WeightNames.ARCH_QWEN3));
    assertSame(Gemma3Processor.INSTANCE, ArchitectureProcessors.of(WeightNames.ARCH_GEMMA3));
    assertSame(Gemma4Processor.INSTANCE, ArchitectureProcessors.of(WeightNames.ARCH_GEMMA4));
    assertSame(LlamaProcessor.INSTANCE, ArchitectureProcessors.of(WeightNames.ARCH_LLAMA));
    assertSame(Lfm2Processor.INSTANCE, ArchitectureProcessors.of(WeightNames.ARCH_LFM2));
    assertSame(BertProcessor.INSTANCE, ArchitectureProcessors.of(WeightNames.ARCH_BERT));
    assertSame(WhisperProcessor.INSTANCE, ArchitectureProcessors.of(WeightNames.ARCH_WHISPER));
  }

  @Test
  void chatFamiliesAreNotEmbedding() {
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_QWEN3).isEmbedding());
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_GEMMA3).isEmbedding());
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_GEMMA4).isEmbedding());
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_LLAMA).isEmbedding());
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_LFM2).isEmbedding());
    assertTrue(ArchitectureProcessors.of(WeightNames.ARCH_BERT).isEmbedding());
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_WHISPER).isEmbedding());
    assertTrue(ArchitectureProcessors.of(WeightNames.ARCH_WHISPER).isSpeech());
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_QWEN3).isSpeech());
    assertFalse(ArchitectureProcessors.of(WeightNames.ARCH_BERT).isSpeech());
  }

  @Test
  void unknownArchitectureHasNoProcessor() {
    IllegalArgumentException ex = assertThrows(
      IllegalArgumentException.class, () -> ArchitectureProcessors.of("mistral"));
    assertEquals("no architecture processor for 'mistral'", ex.getMessage());
  }

  @Test
  void chatFamiliesShareCausalTemplateAndBertIsEmbedding() {
    assertInstanceOf(CausalArchitecture.class, ArchitectureProcessors.of(WeightNames.ARCH_QWEN3));
    assertInstanceOf(CausalArchitecture.class, ArchitectureProcessors.of(WeightNames.ARCH_GEMMA3));
    assertInstanceOf(CausalArchitecture.class, ArchitectureProcessors.of(WeightNames.ARCH_GEMMA4));
    assertInstanceOf(CausalArchitecture.class, ArchitectureProcessors.of(WeightNames.ARCH_LLAMA));
    assertInstanceOf(CausalArchitecture.class, ArchitectureProcessors.of(WeightNames.ARCH_LFM2));
    assertInstanceOf(EmbeddingArchitecture.class, ArchitectureProcessors.of(WeightNames.ARCH_BERT));
    assertInstanceOf(SpeechArchitecture.class, ArchitectureProcessors.of(WeightNames.ARCH_WHISPER));
  }

  @Test
  void embeddingProcessorRejectsCausalGraph() {
    assertThrows(
      IllegalStateException.class,
      () -> BertProcessor.INSTANCE.createCausal(null, null));
  }

  @Test
  void chatProcessorRejectsEmbeddingGraph() {
    assertThrows(
      IllegalStateException.class,
      () -> Qwen3Processor.INSTANCE.createEmbedding(null, null));
  }

  @Test
  void speechProcessorRejectsChatAndEmbeddingGraphs() {
    assertThrows(
      IllegalStateException.class,
      () -> WhisperProcessor.INSTANCE.createCausal(null, null));
    assertThrows(
      IllegalStateException.class,
      () -> WhisperProcessor.INSTANCE.createEmbedding(null, null));
  }

  @Test
  void chatAndEmbeddingProcessorsRejectSpeechGraph() {
    assertThrows(
      IllegalStateException.class,
      () -> Qwen3Processor.INSTANCE.createSpeech(null, null));
    assertThrows(
      IllegalStateException.class,
      () -> BertProcessor.INSTANCE.createSpeech(null, null));
  }
}
