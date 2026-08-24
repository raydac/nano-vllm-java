package com.igormaznitsa.nanollvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.models.internal.CausalLMFactory;
import com.igormaznitsa.nanollvm.models.internal.ConvLayout;
import com.igormaznitsa.nanollvm.models.internal.EmbeddingEncoderFactory;
import com.igormaznitsa.nanollvm.models.internal.WeightNames;
import com.igormaznitsa.nanollvm.models.llmarch.OnnxWeightNames;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxDataTypes;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxDataTypes.Kind;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxGraphBundle;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxTensorProto;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxTransport;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxWeightReader;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class OnnxSupportUnitTest {

  private static byte[] encodeModelWithConvDilation(final String nodeName, final int dilation) {
    ByteBuffer attr = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    writeTag(attr, 1, 2);
    writeBytes(attr, "dilations".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(attr, 8, 0);
    writeVarint(attr, dilation);
    attr.flip();
    byte[] attrBytes = new byte[attr.remaining()];
    attr.get(attrBytes);

    ByteBuffer node = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN);
    writeTag(node, 1, 2);
    writeBytes(node, "x".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 1, 2);
    writeBytes(node, "W".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 3, 2);
    writeBytes(node, nodeName.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 4, 2);
    writeBytes(node, "Conv".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 5, 2);
    writeBytes(node, attrBytes);
    node.flip();
    byte[] nodeBytes = new byte[node.remaining()];
    node.get(nodeBytes);
    return encodeLengthDelimited(7, encodeLengthDelimited(1, nodeBytes));
  }

  private static byte[] encodeModelWithConvTranspose(
    final String nodeName,
    final int stride,
    final int padding,
    final int outputPadding
  ) {
    ByteBuffer strides = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    writeTag(strides, 1, 2);
    writeBytes(strides, "strides".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(strides, 8, 0);
    writeVarint(strides, stride);
    strides.flip();
    byte[] strideBytes = new byte[strides.remaining()];
    strides.get(strideBytes);

    ByteBuffer pads = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    writeTag(pads, 1, 2);
    writeBytes(pads, "pads".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(pads, 8, 0);
    writeVarint(pads, padding);
    writeTag(pads, 8, 0);
    writeVarint(pads, padding);
    pads.flip();
    byte[] padBytes = new byte[pads.remaining()];
    pads.get(padBytes);

    ByteBuffer outPad = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    writeTag(outPad, 1, 2);
    writeBytes(outPad, "output_padding".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(outPad, 8, 0);
    writeVarint(outPad, outputPadding);
    outPad.flip();
    byte[] outPadBytes = new byte[outPad.remaining()];
    outPad.get(outPadBytes);

    ByteBuffer node = ByteBuffer.allocate(512).order(ByteOrder.LITTLE_ENDIAN);
    writeTag(node, 1, 2);
    writeBytes(node, "x".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 1, 2);
    writeBytes(node, "W".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 3, 2);
    writeBytes(node, nodeName.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 4, 2);
    writeBytes(node, "ConvTranspose".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(node, 5, 2);
    writeBytes(node, strideBytes);
    writeTag(node, 5, 2);
    writeBytes(node, padBytes);
    writeTag(node, 5, 2);
    writeBytes(node, outPadBytes);
    node.flip();
    byte[] nodeBytes = new byte[node.remaining()];
    node.get(nodeBytes);
    return encodeLengthDelimited(7, encodeLengthDelimited(1, nodeBytes));
  }

  private static byte[] encodeModelWithFloatTensor(
    final String name,
    final long[] dims,
    final float[] values
  ) {
    return encodeModelWithRawField(name, dims, values, 13);
  }

  private static byte[] encodeModelWithLegacyRawData(
    final String name,
    final long[] dims,
    final float[] values
  ) {
    return encodeModelWithRawField(name, dims, values, 9);
  }

  private static byte[] encodeModelWithRawField(
    final String name,
    final long[] dims,
    final float[] values,
    final int rawField
  ) {
    byte[] tensor = encodeTensor(name, dims, values, rawField);
    byte[] graph = encodeLengthDelimited(5, tensor);
    return encodeLengthDelimited(7, graph);
  }

  private static byte[] encodeTensor(
    final String name,
    final long[] dims,
    final float[] values,
    final int rawField
  ) {
    ByteBuffer out = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN);
    for (long dim : dims) {
      writeTag(out, 1, 0);
      writeVarint(out, dim);
    }
    writeTag(out, 2, 0);
    writeVarint(out, OnnxProtoReader.FLOAT);
    writeTag(out, 8, 2);
    writeBytes(out, name.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    writeTag(out, rawField, 2);
    ByteBuffer raw = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    raw.asFloatBuffer().put(values);
    writeBytes(out, raw.array());
    out.flip();
    byte[] bytes = new byte[out.remaining()];
    out.get(bytes);
    return bytes;
  }

  private static byte[] encodeLengthDelimited(final int field, final byte[] payload) {
    ByteBuffer out = ByteBuffer.allocate(payload.length + 16).order(ByteOrder.LITTLE_ENDIAN);
    writeTag(out, field, 2);
    writeBytes(out, payload);
    out.flip();
    byte[] bytes = new byte[out.remaining()];
    out.get(bytes);
    return bytes;
  }

  private static void writeTag(final ByteBuffer out, final int field, final int wire) {
    writeVarint(out, ((long) field << 3) | wire);
  }

  private static void writeBytes(final ByteBuffer out, final byte[] bytes) {
    writeVarint(out, bytes.length);
    out.put(bytes);
  }

  private static void writeVarint(final ByteBuffer out, final long value) {
    long v = value;
    while ((v & ~0x7FL) != 0L) {
      out.put((byte) ((v & 0x7F) | 0x80));
      v >>>= 7;
    }
    out.put((byte) v);
  }

  @Test
  void detectsLlamaArchitecture() {
    Config.HfConfig config = Config.HfConfig.parse("""
      {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "attention_bias": false,
        "rope_theta": 10000.0
      }
      """);
    assertEquals(WeightNames.ARCH_LLAMA, CausalLMFactory.detect(config));
    assertFalse(EmbeddingEncoderFactory.isEmbeddingArchitecture(config));
  }

  @Test
  void normalizesChatAndBertNames() {
    assertEquals(
      "model.layers.0.self_attn.q_proj.weight",
      OnnxWeightNames.normalizeChatName("/model.layers.0.self_attn.q_proj.weight"));
    assertEquals(
      WeightNames.GGUF_TOKEN_EMBD,
      OnnxWeightNames.normalizeBertName("embeddings.word_embeddings.weight"));
    assertEquals(
      WeightNames.GGUF_TOKEN_EMBD,
      OnnxWeightNames.normalizeBertName("roberta.embeddings.word_embeddings.weight"));
    assertEquals(
      WeightNames.GGUF_TOKEN_EMBD,
      OnnxWeightNames.normalizeBertName("xlm-roberta.embeddings.word_embeddings.weight"));
    assertEquals(
      WeightNames.GGUF_TOKEN_EMBD,
      OnnxWeightNames.normalizeBertName("roberta/embeddings/word_embeddings.weight"));
    assertEquals(
      "blk.0.attn_q.weight",
      OnnxWeightNames.normalizeBertName("encoder.layer.0.attention.self.query.weight"));
    assertEquals(
      "blk.0.attn_q.weight",
      OnnxWeightNames.normalizeBertName("roberta.encoder.layer.0.attention.self.query.weight"));
    assertEquals(
      WeightNames.GGUF_TOKEN_EMBD,
      OnnxWeightNames.normalizeBertName("nomic_bert.embeddings.word_embeddings.weight"));
    assertEquals(
      "blk.2.attn_k.weight",
      OnnxWeightNames.normalizeBertName(
        "model.encoder.layer.2.attention.self.key.weight"));
    assertEquals(
      WeightNames.GGUF_TOKEN_EMBD,
      OnnxWeightNames.normalizeBertName("model.bert.embeddings.word_embeddings.weight"));
    assertEquals(
      "blk.7.attn_output.weight",
      OnnxWeightNames.normalizeBertName("encoder.layer.7.attention.output.dense.weight"));
    assertEquals(
      "blk.0.attn_output.weight",
      OnnxWeightNames.normalizeBertName(
        OnnxProtoReader.matMulNodeToWeightName("/encoder/layer.0/attention/output/dense/MatMul")));
    assertEquals(
      "blk.0.attn_q.weight",
      OnnxWeightNames.normalizeBertName(
        OnnxProtoReader.matMulNodeToWeightName("/encoder/layer.0/attention/self/query/MatMul")));
  }

  @Test
  void rejectsQuantizedOnnxNames() {
    assertTrue(OnnxTransport.isAllowedName("model.onnx"));
    assertTrue(OnnxTransport.isAllowedName("model_fp16.onnx"));
    assertFalse(OnnxTransport.isAllowedName("model_q4.onnx"));
    assertFalse(OnnxTransport.isAllowedName("decoder_with_past_model.onnx"));
  }

  @Test
  void readsEmbeddedFloatInitializer(@TempDir final Path dir) throws Exception {
    float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
    byte[] onnx =
      encodeModelWithFloatTensor("model.embed_tokens.weight", new long[] {2, 2}, values);
    Path file = dir.resolve("model.onnx");
    Files.write(file, onnx);

    List<OnnxTensorProto> tensors = OnnxProtoReader.readInitializers(file);
    assertEquals(1, tensors.size());
    assertEquals("model.embed_tokens.weight", tensors.getFirst().name());
    assertEquals(OnnxProtoReader.FLOAT, tensors.getFirst().dataType());

    Tensor decoded = OnnxWeightReader.toTensor(tensors.getFirst(), dir);
    assertEquals(2, decoded.size(0));
    assertEquals(2, decoded.size(1));
    assertEquals(1.0f, decoded.data()[decoded.offset()]);
    assertEquals(4.0f, decoded.data()[decoded.offset() + 3]);
  }

  @Test
  void readsLegacyRawDataField9() throws Exception {
    float[] values = {1.5f, 2.5f, 3.5f, 4.5f};
    byte[] onnx = encodeModelWithLegacyRawData("w", new long[] {2, 2}, values);
    List<OnnxTensorProto> tensors = OnnxProtoReader.readInitializers(
      ByteBuffer.wrap(onnx).order(ByteOrder.LITTLE_ENDIAN), "legacy");
    assertEquals(1, tensors.size());
    assertEquals(4, tensors.getFirst().rawData().length / 4);
    Tensor decoded = OnnxWeightReader.toTensor(tensors.getFirst(), Path.of("."));
    assertEquals(1.5f, decoded.data()[decoded.offset()]);
  }

  @Test
  void mapsMatMulNodeNameToHfWeight() {
    assertEquals(
      "model.layers.0.self_attn.q_proj.weight",
      OnnxProtoReader.matMulNodeToWeightName("/model/layers.0/self_attn/q_proj/MatMul"));
    assertEquals(
      "lm_head.weight",
      OnnxProtoReader.matMulNodeToWeightName("/lm_head/MatMul"));
  }

  @Test
  void mapsConvAndEmbeddingGatherNodeNamesToPytorchWeights() {
    assertEquals(
      "flow.flows.0.enc.in_layers.0.weight",
      OnnxProtoReader.convNodeToWeightName("/flow/flows.0/enc/in_layers.0/Conv"));
    assertEquals(
      "dec.ups.0.weight",
      OnnxProtoReader.convTransposeNodeToWeightName("/dec/ups.0/ConvTranspose"));
    assertEquals(
      "enc_p.emb.weight",
      OnnxProtoReader.gatherNodeToWeightName("/enc_p/emb/Gather"));
  }

  @Test
  void readsConvDilationFromNodeAttributes() throws Exception {
    byte[] onnx = encodeModelWithConvDilation("/dec/resblocks.0/convs.1/Conv", 2);
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(
      ByteBuffer.wrap(onnx).order(ByteOrder.LITTLE_ENDIAN),
      "conv-dil");
    assertEquals(2, graph.convDilations().get("dec.resblocks.0.convs.1.weight"));
    ConvLayout layout = graph.convLayouts().get("dec.resblocks.0.convs.1.weight");
    assertEquals(ConvLayout.Kind.CONV, layout.kind());
    assertEquals(2, layout.dilation());
    assertEquals(1, layout.strideOr(1));
    assertNull(layout.stride());
    assertEquals(1, layout.paddingOr(1));
  }

  @Test
  void readsConvTransposeGeometryFromNodeAttributes() throws Exception {
    byte[] onnx = encodeModelWithConvTranspose(
      "/dec/ups.0/ConvTranspose", 8, 4, 1);
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(
      ByteBuffer.wrap(onnx).order(ByteOrder.LITTLE_ENDIAN),
      "conv-transpose");
    ConvLayout layout = graph.convLayouts().get("dec.ups.0.weight");
    assertEquals(ConvLayout.Kind.CONV_TRANSPOSE, layout.kind());
    assertEquals(8, layout.stride());
    assertEquals(4, layout.padding());
    assertEquals(1, layout.outputPadding());
    assertTrue(layout.isTransposed());
  }

  @Test
  void zeroOnnxPadsDoNotOverrideFusedPaddingFallback() {
    ConvLayout omitted = new ConvLayout(ConvLayout.Kind.CONV, 1, 0, null, 1, 1);
    assertEquals(1, omitted.paddingOr(1));
    ConvLayout explicit = new ConvLayout(ConvLayout.Kind.CONV, 1, 3, null, 1, 1);
    assertEquals(3, explicit.paddingOr(1));
  }

  @Test
  void classifiesAllKnownOnnxDataTypes() {
    Map<Integer, Kind> expected = Map.ofEntries(
      Map.entry(OnnxDataTypes.UNDEFINED, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.FLOAT, Kind.LOADABLE_FLOAT),
      Map.entry(OnnxDataTypes.UINT8, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.INT8, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.UINT16, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.INT16, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.INT32, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.INT64, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.STRING, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.BOOL, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.FLOAT16, Kind.LOADABLE_FLOAT),
      Map.entry(OnnxDataTypes.DOUBLE, Kind.LOADABLE_FLOAT),
      Map.entry(OnnxDataTypes.UINT32, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.UINT64, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.COMPLEX64, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.COMPLEX128, Kind.SKIP_GRAPH_CONSTANT),
      Map.entry(OnnxDataTypes.BFLOAT16, Kind.LOADABLE_FLOAT),
      Map.entry(OnnxDataTypes.FLOAT8E4M3FN, Kind.UNSUPPORTED_WEIGHT),
      Map.entry(OnnxDataTypes.FLOAT8E4M3FNUZ, Kind.UNSUPPORTED_WEIGHT),
      Map.entry(OnnxDataTypes.FLOAT8E5M2, Kind.UNSUPPORTED_WEIGHT),
      Map.entry(OnnxDataTypes.FLOAT8E5M2FNUZ, Kind.UNSUPPORTED_WEIGHT),
      Map.entry(OnnxDataTypes.UINT4, Kind.UNSUPPORTED_WEIGHT),
      Map.entry(OnnxDataTypes.INT4, Kind.UNSUPPORTED_WEIGHT),
      Map.entry(OnnxDataTypes.FLOAT4E2M1, Kind.UNSUPPORTED_WEIGHT));

    expected.forEach((code, kind) -> assertEquals(kind, OnnxDataTypes.kind(code), "type " + code));
    assertEquals(Kind.UNSUPPORTED_WEIGHT, OnnxDataTypes.kind(999));
    assertTrue(OnnxDataTypes.name(999).contains("999"));

    IntStream.rangeClosed(0, 23).forEach(code -> {
      assertFalse(OnnxDataTypes.name(code).isBlank());
      assertTrue(
        OnnxDataTypes.isLoadableFloatingWeight(code)
          || OnnxDataTypes.isSkippableGraphConstant(code)
          || OnnxDataTypes.kind(code) == Kind.UNSUPPORTED_WEIGHT);
    });
  }

  @Test
  void skipsInt64ConstantsButRejectsFloat8Weights() {
    OnnxTensorProto int64 = new OnnxTensorProto(
      "onnx::Constant_0", OnnxDataTypes.INT64, new long[] {8}, null, null, null, Map.of());
    assertFalse(OnnxDataTypes.shouldLoadAsWeight(int64));
    OnnxDataTypes.requireHandledOrSkip(int64);

    OnnxTensorProto scalarFloat = new OnnxTensorProto(
      "scalar", OnnxDataTypes.FLOAT, new long[] {}, null, null, null, Map.of());
    assertFalse(OnnxDataTypes.shouldLoadAsWeight(scalarFloat));
    OnnxDataTypes.requireHandledOrSkip(scalarFloat);

    OnnxTensorProto float8 = new OnnxTensorProto(
      "model.layers.0.mlp.down_proj.weight",
      OnnxDataTypes.FLOAT8E4M3FN,
      new long[] {4, 4},
      null,
      null,
      null,
      Map.of());
    assertFalse(OnnxDataTypes.shouldLoadAsWeight(float8));
    assertThrows(ModelLoadException.class, () -> OnnxDataTypes.requireHandledOrSkip(float8));
  }
}
