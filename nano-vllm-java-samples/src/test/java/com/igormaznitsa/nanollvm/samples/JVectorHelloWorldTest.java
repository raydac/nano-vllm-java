package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class JVectorHelloWorldTest {

  @Test
  void explicitPathsAreUsedAsGiven() {
    Path chat = Path.of("models/Gemma3-270M");
    Path encoder = Path.of("models/gte-small.Q2_K.gguf");
    String[] args = {chat.toString(), encoder.toString()};

    assertEquals(chat.toAbsolutePath().normalize(), JVectorHelloWorld.chatModel(args));
    assertEquals(encoder.toAbsolutePath().normalize(), JVectorHelloWorld.encoderModel(args));
  }

  @Test
  void encoderPathIsTheSecondArgument() {
    Path encoder = Path.of("models/custom-encoder");
    String[] args = {"  ", encoder.toString()};

    assertEquals(encoder.toAbsolutePath().normalize(), JVectorHelloWorld.encoderModel(args));
  }
}
