package com.igormaznitsa.nanollvm.models;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.internal.ModelFileBundle;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelFileSourceTest {

  @TempDir
  Path temp;

  @Test
  void loadsHfSidecarsAndWeightsIntoMemory() throws IOException {
    Map<ModelFileId, byte[]> files = Map.of(
      ModelFileId.CONFIG, "{\"model_type\":\"qwen3\"}".getBytes(UTF_8),
      ModelFileId.TOKENIZER, "{\"model\":{}}".getBytes(UTF_8),
      ModelFileId.MODEL_SAFE_TENSORS, new byte[] {1, 2, 3, 4});
    AtomicInteger weightOpens = new AtomicInteger();
    ModelFileSource source = new ModelFileSource() {
      @Override
      public InputStream open(final ModelFileId id) {
        if (id == ModelFileId.MODEL_SAFE_TENSORS) {
          weightOpens.incrementAndGet();
        }
        byte[] bytes = files.get(id);
        return bytes == null ? null : new ByteArrayInputStream(bytes);
      }

      @Override
      public String displayName() {
        return "test:hf-tiny";
      }
    };

    ModelFileBundle bundle = ModelFileBundle.load(source, null);
    assertFalse(bundle.isGguf());
    assertEquals("{\"model_type\":\"qwen3\"}", bundle.configJson());
    assertEquals(1, bundle.safetensors().size());
    assertEquals(4, bundle.safetensors().getFirst().bytes().length);
    assertEquals(1, weightOpens.get());
    assertTrue(bundle.virtualPath().toString().contains("nanollvm-memory"));
  }

  @Test
  void loadsGgufIntoMemory() throws IOException {
    byte[] payload = new byte[] {0x47, 0x47, 0x55, 0x46};
    ModelFileSource source = new ModelFileSource() {
      @Override
      public InputStream open(final ModelFileId id) {
        return id == ModelFileId.GGUF ? new ByteArrayInputStream(payload) : null;
      }

      @Override
      public String displayName() {
        return "test:gguf-tiny";
      }
    };

    ModelFileBundle bundle = ModelFileBundle.load(source, null);
    assertTrue(bundle.isGguf());
    assertEquals(4, bundle.ggufBuffer().remaining());
  }

  @Test
  void classpathAdapterReadsTestResources() throws IOException {
    ModelFileSource source = ModelFileSources.classpath(
      ModelFileSourceTest.class.getClassLoader(),
      "model-fixtures/tiny-hf");
    ModelFileBundle bundle = ModelFileBundle.load(source, null);
    assertFalse(bundle.isGguf());
    assertEquals("{\"model_type\":\"test\"}", bundle.configJson().strip());
    assertEquals("weights", new String(bundle.safetensors().getFirst().bytes(), UTF_8).strip());
  }

  @Test
  void missingConfigFailsFast() {
    ModelFileSource source = id -> null;
    assertThrows(ModelLoadException.class, () -> ModelFileBundle.load(source, null));
  }

  @Test
  void directoryAdapterOpensHfLayout() throws IOException {
    Path root = this.temp.resolve("hf");
    Files.createDirectories(root);
    Files.writeString(root.resolve("config.json"), "{}");
    Files.write(root.resolve("model.safetensors"), new byte[] {9});

    try (InputStream in = ModelFileSources.fromPath(root).open(ModelFileId.CONFIG)) {
      assertEquals("{}", new String(in.readAllBytes(), UTF_8));
    }
    try (InputStream missing = ModelFileSources.fromPath(root).open(ModelFileId.TOKENIZER)) {
      assertTrue(missing == null);
    }
  }
}
