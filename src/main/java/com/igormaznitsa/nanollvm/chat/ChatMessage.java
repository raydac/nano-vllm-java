package com.igormaznitsa.nanollvm.chat;

import static java.util.Objects.requireNonNull;

import java.util.LinkedHashMap;
import java.util.Map;

public record ChatMessage(ChatRole role, String content) {

  public ChatMessage {
    requireNonNull(role, "role");
    content = content == null ? "" : content;
  }

  public static ChatMessage system(String content) {
    return new ChatMessage(ChatRole.SYSTEM, content);
  }

  public static ChatMessage user(String content) {
    return new ChatMessage(ChatRole.USER, content);
  }

  public static ChatMessage assistant(String content) {
    return new ChatMessage(ChatRole.ASSISTANT, content);
  }

  public static ChatMessage fromMap(Map<String, String> map) {
    requireNonNull(map, "map");
    return new ChatMessage(
        ChatRole.fromWire(map.get("role")),
        map.getOrDefault("content", ""));
  }

  public Map<String, String> toMap() {
    Map<String, String> m = new LinkedHashMap<>();
    m.put("role", this.role.wireName());
    m.put("content", this.content);
    return m;
  }
}
