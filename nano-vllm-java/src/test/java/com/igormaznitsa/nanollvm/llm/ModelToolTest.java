package com.igormaznitsa.nanollvm.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.chat.ChatMessage;
import com.igormaznitsa.nanollvm.chat.ChatRole;
import com.igormaznitsa.nanollvm.chat.ThinkTags;
import com.igormaznitsa.nanollvm.internal.Json;
import com.igormaznitsa.nanollvm.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ModelToolTest {

  private static final String SCHEMA = """
    {"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}""";

  private static ModelTool weatherTool() {
    return ModelTool.of("get_weather", "Look up weather", SCHEMA, args -> "sunny");
  }

  private static Tokenizer chatMlTokenizer() {
    return Tokenizer.fromJsonDocuments(
      """
        {
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "<|im_start|>": 1, "<|im_end|>": 2, "hi": 3},
            "merges": []
          },
          "added_tokens": [
            {"id": 1, "content": "<|im_start|>", "special": true},
            {"id": 2, "content": "<|im_end|>", "special": true}
          ]
        }
        """,
      """
        {
          "eos_token": "<|im_end|>",
          "pad_token": "<|im_end|>",
          "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        }
        """,
      null,
      "{\"model_type\":\"qwen3\"}");
  }

  @Test
  void rejectsANameTheDialectsCannotEmit() {
    assertThrows(IllegalArgumentException.class,
      () -> ModelTool.of("get weather", "Look up weather", SCHEMA, args -> "sunny"));
    assertThrows(IllegalArgumentException.class,
      () -> ModelTool.of("weather", "   ", SCHEMA, args -> "sunny"));
    assertThrows(IllegalArgumentException.class,
      () -> ModelTool.of("weather", "Look up weather", "[]", args -> "sunny"));
  }

  @Test
  void qwen3HermesCallIsInjectedAsAUserToolResponse() {
    ModelTool weather = weatherTool();
    ToolDialect dialect = ToolDialects.of("qwen3");
    Tokenizer tokenizer = chatMlTokenizer();
    List<ChatMessage> history = new ArrayList<>(List.of(
      ChatMessage.system("Be brief."),
      ChatMessage.user("Weather in Paris?")));

    String prompt = dialect.render(history, List.of(weather), tokenizer, false, ThinkTags.DEFAULT);
    int toolsAt = prompt.indexOf("# Tools");
    int policyAt = prompt.indexOf("Be brief.");
    assertTrue(policyAt >= 0 && toolsAt > policyAt);
    assertTrue(prompt.contains("<tools>"));
    assertTrue(prompt.contains("get_weather"));
    assertTrue(prompt.contains("<|im_start|>assistant"));

    String raw = """
      <tool_call>
      {"name": "get_weather", "arguments": {"city": "Paris"}}
      </tool_call><|im_end|>
      """;
    assertEquals("", dialect.visibleText(raw));
    List<ToolCall> calls = dialect.parseCalls(raw);
    assertEquals("get_weather", calls.getFirst().name());
    assertEquals("Paris", Json.parseObject(calls.getFirst().argumentsJson()).get("city"));

    List<ChatMessage> exchange = dialect.appendExchange(
      raw, calls, ToolExchange.execute(List.of(weather), calls));
    assertEquals(ChatRole.USER, exchange.get(1).role());
    assertTrue(exchange.get(1).content().contains("<tool_response>"));
    assertTrue(exchange.get(1).content().contains("sunny"));
    assertFalse(exchange.getFirst().content().contains("<|im_end|>"));

    history.addAll(exchange);
    String follow = dialect.render(history, List.of(weather), tokenizer, false, ThinkTags.DEFAULT);
    assertTrue(follow.contains("<tool_response>"));
    assertTrue(follow.contains("sunny"));
    assertTrue(follow.endsWith("<|im_start|>assistant\n"));
  }

  @Test
  void hermesAcceptsSeveralCallsAndAStringifiedArgumentsObject() {
    ToolDialect dialect = ToolDialects.of("qwen3");
    String raw = """
      Sure.
      <tool_call>
      {"name": "get_weather", "arguments": "{\\"city\\":\\"Paris\\"}"}
      </tool_call>
      <tool_call>
      {"name": "other", "arguments": {"n": 2}}
      </tool_call>
      """;
    List<ToolCall> calls = dialect.parseCalls(raw);
    assertEquals(2, calls.size());
    assertEquals("Paris", Json.parseObject(calls.getFirst().argumentsJson()).get("city"));
    assertEquals(2L, Json.parseObject(calls.get(1).argumentsJson()).get("n"));
    assertEquals("Sure.", dialect.visibleText(raw));
    assertEquals("os_name", dialect.parseCalls("""
      <tool_call>
      {"name": "os_ name", "arguments": {}}
      </tool_call>
      """).getFirst().name());
    String withTrailingAnswer = raw + "\nos_ name\n.EDITORCONFIG\n";
    String stored =
      dialect.appendExchange(withTrailingAnswer, calls, List.of("sunny", "no")).getFirst()
        .content();
    assertTrue(stored.contains("</tool_call>"));
    assertFalse(stored.contains("os_ name"));
    assertFalse(stored.contains(".EDITORCONFIG"));
    assertEquals("error: unknown tool other",
      ToolExchange.execute(List.of(weatherTool()), calls).get(1));
  }

  @Test
  void qwen35XmlCallKeepsMultilineParameters() {
    ToolDialect dialect = ToolDialects.of("qwen3_5");
    Tokenizer tokenizer = chatMlTokenizer();
    String prompt = dialect.render(
      List.of(ChatMessage.system("Be brief."), ChatMessage.user("hi")),
      List.of(weatherTool()),
      tokenizer,
      false,
      ThinkTags.DEFAULT);
    assertTrue(prompt.indexOf("# Tools") < prompt.indexOf("Be brief."));
    assertTrue(prompt.contains("<function=example_function_name>"));

    String raw = """
      <tool_call>
      <function=get_weather>
      <parameter=city>
      Paris
      line 2
      </parameter>
      <parameter=flag>
      true
      </parameter>
      </function>
      </tool_call>
      """;
    ToolCall call = dialect.parseCalls(raw).getFirst();
    assertEquals("get_weather", call.name());
    Map<String, Object> arguments = Json.parseObject(call.argumentsJson());
    assertEquals("Paris\nline 2", arguments.get("city"));
    assertEquals(true, arguments.get("flag"));
    assertEquals("note", dialect.visibleText("note\n" + raw));
  }

  @Test
  void lfm2PythonicCallComesBackAsAToolTurn() {
    ToolDialect dialect = ToolDialects.of("lfm2");
    Tokenizer tokenizer = chatMlTokenizer();
    String prompt = dialect.render(
      List.of(ChatMessage.user("status?")),
      List.of(weatherTool()),
      tokenizer,
      false,
      ThinkTags.DEFAULT);
    assertTrue(prompt.contains("<|tool_list_start|>"));
    assertTrue(prompt.contains("get_weather"));

    String raw =
      "<|tool_call_start|>[get_weather(city='Paris', n=2), missing(city=\"X\")]<|tool_call_end|>";
    List<ToolCall> calls = dialect.parseCalls(raw);
    assertEquals("Paris", Json.parseObject(calls.getFirst().argumentsJson()).get("city"));
    assertEquals(2L, Json.parseObject(calls.getFirst().argumentsJson()).get("n"));
    assertEquals("missing", calls.get(1).name());
    assertEquals("", dialect.visibleText(" " + raw + " "));

    List<ChatMessage> exchange = dialect.appendExchange(
      raw, calls, ToolExchange.execute(List.of(weatherTool()), calls));
    assertEquals(ChatRole.TOOL, exchange.get(1).role());
    assertTrue(exchange.get(1).content().startsWith("<|tool_response_start|>"));
    assertTrue(exchange.get(2).content().contains("unknown tool missing"));

    String follow = dialect.render(
      List.of(ChatMessage.user("status?"), exchange.getFirst(), exchange.get(1)),
      List.of(weatherTool()),
      tokenizer,
      false,
      ThinkTags.DEFAULT);
    assertTrue(follow.contains("<|im_start|>tool\n"));
    assertTrue(follow.contains("sunny"));
  }

  @Test
  void gemma4CallStaysInsideTheModelTurn() {
    ToolDialect dialect = ToolDialects.of("gemma4");
    assertEquals(List.of("<|tool_response>"), dialect.extraStopTokens());
    Tokenizer tokenizer = chatMlTokenizer();
    String prompt = dialect.render(
      List.of(ChatMessage.system("Be helpful."), ChatMessage.user("Weather?")),
      List.of(weatherTool()),
      tokenizer,
      false,
      ThinkTags.DEFAULT);
    assertTrue(prompt.contains("<|tool>declaration:get_weather"));
    assertTrue(prompt.contains("<|\"|>"));
    assertTrue(prompt.endsWith("<|turn>model\n"));

    String raw = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>";
    ToolCall call = dialect.parseCalls(raw).getFirst();
    assertEquals("get_weather", call.name());
    assertEquals("Paris", Json.parseObject(call.argumentsJson()).get("city"));
    assertEquals("", dialect.visibleText(raw));

    ChatMessage continued = dialect.appendExchange(
      raw, List.of(call), List.of("{\"ok\":true}")).getFirst();
    assertTrue(continued.content().contains(
      "<|tool_response>response:get_weather{ok:true}<tool_response|>"));
    assertFalse(continued.content().endsWith("<|tool_response>"));

    String follow = dialect.render(
      List.of(ChatMessage.user("Weather?"), continued),
      List.of(weatherTool()),
      tokenizer,
      false,
      ThinkTags.DEFAULT);
    assertTrue(follow.contains("<tool_response|>"));
    assertFalse(follow.endsWith("<|turn>model\n"));
    assertFalse(follow.endsWith("<turn|>\n"));
  }

  @Test
  void aThrowingToolIsReportedBackToTheModel() {
    ModelTool broken = ModelTool.of("get_weather", "Look up weather", SCHEMA,
      args -> {
        throw new IllegalStateException("offline");
      });
    List<String> results = ToolExchange.execute(
      List.of(broken),
      List.of(new ToolCall("get_weather", "{\"city\":\"Paris\"}")));
    assertEquals("error: offline", results.getFirst());
  }

  @Test
  void llamaAndGemma3HaveNoToolRoundTrip() {
    assertTrue(assertThrows(IllegalArgumentException.class, () -> ToolDialects.of("llama"))
      .getMessage().contains("Llama 3"));
    assertTrue(assertThrows(IllegalArgumentException.class, () -> ToolDialects.of("gemma3"))
      .getMessage().contains("Gemma 4"));
  }
}
