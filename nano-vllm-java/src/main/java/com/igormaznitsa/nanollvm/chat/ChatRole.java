package com.igormaznitsa.nanollvm.chat;

import java.util.Locale;

/**
 * Speaker of a {@link ChatMessage}.
 *
 * <p>{@link #wireName()} is the string written into chat templates ({@code system} / {@code user} /
 * {@code assistant}). {@link #fromWire(String)} is the inverse for template maps: {@code model} is
 * treated as assistant (Gemma turn-based), anything else unknown becomes {@link #USER}.
 */
public enum ChatRole {
  SYSTEM("system"),
  USER("user"),
  ASSISTANT("assistant");

  private final String wireName;

  ChatRole(final String wireName) {
    this.wireName = wireName;
  }

  /**
   * Parses a template / JSON role string.
   *
   * @param role wire name; {@code null} / blank / unknown → {@link #USER}; {@code model} →
   *             {@link #ASSISTANT}
   * @return never {@code null}
   */
  public static ChatRole fromWire(final String role) {
    if (role == null || role.isBlank()) {
      return USER;
    }
    return switch (role.strip().toLowerCase(Locale.ROOT)) {
      case "system" -> SYSTEM;
      case "assistant", "model" -> ASSISTANT;
      default -> USER;
    };
  }

  /**
   * Canonical template role string ({@code system}, {@code user}, or {@code assistant}).
   */
  public String wireName() {
    return this.wireName;
  }
}
