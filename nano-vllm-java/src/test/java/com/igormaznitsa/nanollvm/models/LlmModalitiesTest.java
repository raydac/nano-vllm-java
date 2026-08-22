package com.igormaznitsa.nanollvm.models;

import static java.util.Locale.ROOT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.llm.Config;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class LlmModalitiesTest {

  @Test
  void wireNamesRoundTrip() {
    for (LlmModality modality : LlmModality.values()) {
      assertEquals(Optional.of(modality), LlmModality.fromWire(modality.wireName()));
      assertEquals(Optional.of(modality),
        LlmModality.fromWire(" " + modality.wireName().toUpperCase(ROOT) + " "));
    }
    assertTrue(LlmModality.fromWire(null).isEmpty());
    assertTrue(LlmModality.fromWire("").isEmpty());
    assertTrue(LlmModality.fromWire("smell").isEmpty());
  }

  @Test
  void textChatAndEmbeddingPresets() {
    assertEquals(Set.of(LlmModality.TEXT), LlmModalities.TEXT_TO_TEXT.input());
    assertEquals(Set.of(LlmModality.TEXT), LlmModalities.TEXT_TO_TEXT.output());
    assertTrue(LlmModalities.TEXT_TO_TEXT.accepts(LlmModality.TEXT));
    assertTrue(LlmModalities.TEXT_TO_TEXT.emits(LlmModality.TEXT));
    assertFalse(LlmModalities.TEXT_TO_TEXT.emits(LlmModality.EMBEDDING));
    assertEquals("text->text", LlmModalities.TEXT_TO_TEXT.toString());

    assertEquals(Set.of(LlmModality.TEXT), LlmModalities.TEXT_TO_EMBEDDING.input());
    assertEquals(Set.of(LlmModality.EMBEDDING), LlmModalities.TEXT_TO_EMBEDDING.output());
    assertTrue(LlmModalities.TEXT_TO_EMBEDDING.emits(LlmModality.EMBEDDING));
    assertFalse(LlmModalities.TEXT_TO_EMBEDDING.emits(LlmModality.TEXT));
    assertEquals("text->embedding", LlmModalities.TEXT_TO_EMBEDDING.toString());
    assertEquals(LlmModalities.TEXT_TO_TEXT, LlmModalities.usable(false));
    assertEquals(LlmModalities.TEXT_TO_EMBEDDING, LlmModalities.usable(true));
  }

  @Test
  void checkpointGemma4DeclaresImageAudioVideoWhileRuntimeIsText() {
    Config.HfConfig gemma4 = Config.HfConfig.parse("""
      {
        "model_type": "gemma4",
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_config": {"model_type": "gemma4_vision"},
        "audio_config": {"model_type": "gemma4_audio"},
        "text_config": {"model_type": "gemma4_text", "hidden_size": 1536}
      }
      """);
    LlmModalities declared = LlmModalities.ofCheckpoint(gemma4, false);
    assertEquals("text+image+audio+video->text", declared.toString());
    assertTrue(declared.accepts(LlmModality.AUDIO));
    assertEquals(LlmModalities.TEXT_TO_TEXT, LlmModalities.usable(false));
    assertEquals(LlmModalities.TEXT_TO_EMBEDDING, LlmModalities.ofCheckpoint(gemma4, true));

    Config.HfConfig qwen = Config.HfConfig.parse("""
      {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}
      """);
    assertEquals(LlmModalities.TEXT_TO_TEXT, LlmModalities.ofCheckpoint(qwen, false));
  }

  @Test
  void freezesOrderAndRejectsEmptyInput() {
    LlmModalities pair = LlmModalities.of(
      new LinkedHashSet<>(List.of(LlmModality.IMAGE, LlmModality.TEXT)),
      Set.of());
    assertEquals("text+image->none", pair.toString());
    assertEquals(Set.of(LlmModality.TEXT, LlmModality.IMAGE), pair.input());
    assertTrue(pair.output().isEmpty());
    assertThrows(UnsupportedOperationException.class, () -> pair.input().add(LlmModality.AUDIO));

    assertThrows(IllegalArgumentException.class,
      () -> LlmModalities.of(Set.of(), Set.of(LlmModality.TEXT)));
    assertThrows(NullPointerException.class,
      () -> LlmModalities.of(null, Set.of(LlmModality.TEXT)));
    assertThrows(NullPointerException.class,
      () -> LlmModalities.of(Set.of(LlmModality.TEXT), null));
    assertThrows(NullPointerException.class, () -> LlmModalities.TEXT_TO_TEXT.accepts(null));
    assertThrows(NullPointerException.class, () -> LlmModalities.TEXT_TO_TEXT.emits(null));
  }
}
