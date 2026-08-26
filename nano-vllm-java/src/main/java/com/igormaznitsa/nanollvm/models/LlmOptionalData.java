package com.igormaznitsa.nanollvm.models;

import static java.util.Objects.requireNonNull;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Typed keys for load-time extras that only some checkpoints use (data directories, sidecars).
 *
 * <p>Pass values through {@link LlmModelFactory.Builder#optionalData(Key, Object)}. Look them up
 * on a loaded {@link LlmModel} with {@link LlmModel#optionalData(Key)}. Unknown keys are stored
 * and ignored by families that do not read them.
 *
 * @since 1.3.0
 */
public final class LlmOptionalData {

  /**
   * Directory of espeak-ng-data for Piper TTS. Value is a {@link Path}; a {@link CharSequence}
   * is accepted and converted. Omit the key to use {@code {model}/espeak-ng-data}. A missing
   * or incomplete folder is ignored; TTS still runs. G2P reads {@code dictsource/*_list}
   * and {@code *_rules} when present (download scripts install those next to {@code lang/}),
   * including suffix/prefix {@code S}/{@code P} rules and number fragments. If those source
   * files are missing, compiled {@code phontab} and {@code *_dict} in the same directory
   * are used for listed words.
   *
   * @since 1.3.0
   */
  public static final Key<Path> ESPEAK_DATA = Key.of("espeak.data", Path.class);

  private LlmOptionalData() {
  }

  /**
   * Converts a stored value to {@code key}'s type ({@link Path} keys also accept
   * {@link CharSequence}).
   *
   * @param key   typed key; never {@code null}
   * @param value stored object; never {@code null}
   * @return value as {@code T}
   * @throws IllegalArgumentException if the value cannot be converted
   * @since 1.3.0
   */
  public static <T> T cast(final Key<T> key, final Object value) {
    requireNonNull(key, "key");
    requireNonNull(value, "value");
    if (key.type() == Path.class) {
      return key.type().cast(asPath(value));
    }
    if (key.type().isInstance(value)) {
      return key.type().cast(value);
    }
    throw new IllegalArgumentException(
      "optionalData '" + key.id() + "' must be a " + key.type().getSimpleName()
        + ", got " + value.getClass().getName());
  }

  /**
   * Converts a stored path-like value to a normalized {@link Path}.
   *
   * @param value a {@link Path} or {@link CharSequence}; must not be {@code null}
   * @return absolute normalized path
   * @throws IllegalArgumentException if {@code value} is blank or the wrong type
   * @since 1.3.0
   */
  public static Path asPath(final Object value) {
    if (value instanceof Path path) {
      return path.toAbsolutePath().normalize();
    }
    if (value instanceof CharSequence text) {
      String raw = text.toString().strip();
      if (raw.isEmpty()) {
        throw new IllegalArgumentException("optionalData path must not be blank");
      }
      return Path.of(raw).toAbsolutePath().normalize();
    }
    throw new IllegalArgumentException(
      "optionalData path must be a Path or CharSequence, got " + value.getClass().getName());
  }

  /**
   * Identifier plus expected Java type for one optional datum.
   *
   * @param <T> value type
   * @since 1.3.0
   */
  public static final class Key<T> {

    private final String id;
    private final Class<T> type;

    private Key(final String id, final Class<T> type) {
      this.id = requireNonNull(id, "id");
      this.type = requireNonNull(type, "type");
      if (id.isBlank()) {
        throw new IllegalArgumentException("optionalData key id must not be blank");
      }
    }

    /**
     * Application or library key. Prefer the constants on {@link LlmOptionalData} when one exists.
     *
     * @param id   stable wire name; never blank
     * @param type expected value class; never {@code null}
     * @return typed key
     * @since 1.3.0
     */
    public static <T> Key<T> of(final String id, final Class<T> type) {
      return new Key<>(id, type);
    }

    /**
     * Stable map key stored in {@link LlmModel#OPTION_OPTIONAL_DATA}.
     *
     * @return non-blank id
     * @since 1.3.0
     */
    public String id() {
      return this.id;
    }

    /**
     * Expected value class.
     *
     * @return type used by {@link LlmOptionalData#cast}
     * @since 1.3.0
     */
    public Class<T> type() {
      return this.type;
    }

    @Override
    public boolean equals(final Object other) {
      return other instanceof Key<?> key && this.id.equals(key.id);
    }

    @Override
    public int hashCode() {
      return Objects.hash(this.id);
    }

    @Override
    public String toString() {
      return this.id;
    }
  }
}
