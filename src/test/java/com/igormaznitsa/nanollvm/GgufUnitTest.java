package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.internal.GgufDequant;
import com.igormaznitsa.nanollvm.internal.GgufReader;
import com.igormaznitsa.nanollvm.layers.ShortConv;
import com.igormaznitsa.nanollvm.prompts.ChatPrompts;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import com.igormaznitsa.nanollvm.utils.BundledModels;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class GgufUnitTest {

  @Test
  void dequantQ4_0KnownBlock() {
    ByteBuffer buf = ByteBuffer.allocate(GgufDequant.BLOCK_Q4_0).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(0.5f));
    for (int i = 0; i < 16; i++) {
      buf.put((byte) 0x88);
    }
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, 2, GgufDequant.QK4_0);
    assertEquals(GgufDequant.QK4_0, out.length);
    assertEquals(0f, out[0], 1e-5f);
    assertEquals(0f, out[16], 1e-5f);
  }

  @Test
  void dequantF16RoundTrip() {
    ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    buf.putShort(Float.floatToFloat16(1.0f));
    buf.putShort(Float.floatToFloat16(-2.0f));
    buf.flip();
    float[] out = GgufDequant.dequantize(buf, 1, 2);
    assertEquals(1.0f, out[0], 1e-3f);
    assertEquals(-2.0f, out[1], 1e-3f);
  }

  @Test
  void shortConvKernelIdentityish() {
    int hidden = 2;
    int kernel = 3;
    float[] inProj = new float[3 * hidden * hidden];
    for (int o = 0; o < 3 * hidden; o++) {
      int in = o % hidden;
      inProj[o * hidden + in] = 1f;
    }
    float[] conv = new float[hidden * kernel];
    for (int h = 0; h < hidden; h++) {
      conv[h * kernel + (kernel - 1)] = 1f;
    }
    float[] outProj = new float[hidden * hidden];
    for (int i = 0; i < hidden; i++) {
      outProj[i * hidden + i] = 1f;
    }

    ShortConv layer = new ShortConv(
      Tensor.of(inProj, 3 * hidden, hidden),
      Tensor.of(conv, hidden, kernel),
      Tensor.of(outProj, hidden, hidden),
      0);

    Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, hidden);
    Tensor out = layer.forward(input);
    assertEquals(2, out.size(0));
    assertEquals(hidden, out.size(1));
    assertTrue(Float.isFinite(out.data()[0]));
    assertArrayEquals(new int[] {2, hidden}, out.shape());
  }

  @Test
  void lfm2GgufTokenizerUsesChatMlWithoutThinkInvite() throws Exception {
    Optional<Path> path = BundledModels.find(BundledModels.LFM2_5_2_6B_GGUF);
    assumeTrue(path.isPresent(), "LFM2 GGUF not downloaded");

    try (GgufReader reader = GgufReader.open(path.get())) {
      Tokenizer tok = Tokenizer.fromGguf(reader);
      assertFalse(tok.isGemmaChat());
      assertFalse(tok.invitesThinking());
      assertEquals(ChatPrompts.PLAIN_CHAT_SYSTEM, ChatPrompts.systemFor(tok));

      String chat = tok.applyChatTemplate(
        List.of(
          Map.of("role", "system", "content", "be brief"),
          Map.of("role", "user", "content", "hello")),
        true,
        false);
      assertTrue(chat.contains("<|im_start|>user"));
      assertTrue(chat.contains("<|im_start|>assistant"));
      assertFalse(chat.contains("<think>"), chat);
      assertFalse(chat.startsWith("user:"), chat);
    }
  }
}
