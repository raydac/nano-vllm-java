package io.nanovllm.chat;

import java.util.Locale;

public enum ChatRole {
  SYSTEM("system"),
  USER("user"),
  ASSISTANT("assistant");

  private final String wireName;

  ChatRole(String wireName) {
    this.wireName = wireName;
  }

  public static ChatRole fromWire(String role) {
    if (role == null || role.isBlank()) {
      return USER;
    }
    return switch (role.strip().toLowerCase(Locale.ROOT)) {
      case "system" -> SYSTEM;
      case "assistant", "model" -> ASSISTANT;
      default -> USER;
    };
  }

  public String wireName() {
    return this.wireName;
  }
}
