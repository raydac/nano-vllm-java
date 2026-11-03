package com.igormaznitsa.nanollvm.weather.model;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.utils.NanoLlvmProps;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class WeatherModelsTest {

  private static void restoreModelsDir(final String previous) {
    if (previous == null) {
      System.clearProperty(NanoLlvmProps.PROP_MODELS_DIR);
    } else {
      System.setProperty(NanoLlvmProps.PROP_MODELS_DIR, previous);
    }
  }

  @Test
  void missingChatModelNamesTheQwenDownload() {
    MissingModels ex = assertThrows(
      MissingModels.class,
      () -> WeatherModels.chatModel(new String[] {"/no/such/weather-chat-model"}));

    assertTrue(ex.getMessage().contains("was not found"));
    assertTrue(ex.getMessage().contains("download-qwen3-0.6b.sh"));
  }

  @Test
  void emptyDirectoryIsNotAChatCheckpoint() throws IOException {
    Path dir = Files.createTempDirectory("weather-model");

    MissingModels ex = assertThrows(
      MissingModels.class,
      () -> WeatherModels.chatModel(new String[] {dir.toString()}));

    assertTrue(ex.getMessage().contains("not a checkpoint"));
  }

  @Test
  void startupNamesEveryMissingModel() throws IOException {
    Path root = Files.createTempDirectory("weather-models");
    String previous = System.setProperty(NanoLlvmProps.PROP_MODELS_DIR, root.toString());
    try {
      MissingModels ex =
        assertThrows(MissingModels.class, () -> WeatherModels.require(new String[0]));
      assertTrue(ex.getMessage().contains("cannot start"));
      assertTrue(ex.getMessage().contains("download-qwen3-0.6b.sh"));
      assertTrue(ex.getMessage().contains("download-piper-en-lessac-medium.sh"));
      assertTrue(ex.getMessage().contains("download-piper-ru-irina-medium.sh"));
    } finally {
      restoreModelsDir(previous);
    }
  }

  @Test
  void explicitChatModelIsThatDirectory() throws IOException {
    Path dir = Files.createTempDirectory("weather-model");
    Files.writeString(dir.resolve("config.json"), "{}", UTF_8);
    Files.write(dir.resolve("model.safetensors"), new byte[] {1});

    assertEquals(
      dir.toAbsolutePath().normalize(),
      WeatherModels.chatModel(new String[] {dir.toString()}));
  }

  @Test
  void lessacIsTheEnglishVoice() {
    assertTrue(WeatherModels.englishVoice(Path.of("models/piper-en-lessac-medium")));
    assertFalse(WeatherModels.englishVoice(Path.of("models/piper-ru-irina-medium")));
  }
}
