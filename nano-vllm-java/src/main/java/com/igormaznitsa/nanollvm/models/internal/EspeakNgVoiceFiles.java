package com.igormaznitsa.nanollvm.models.internal;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toUnmodifiableSet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Resolves an espeak voice name to a {@code dictsource} language id and {@code dictrules}.
 */
final class EspeakNgVoiceFiles {

  private static final Pattern LANGUAGE =
    Pattern.compile("^language\\s+(\\S+)(?:\\s+(\\d+))?\\s*$");
  private static final Pattern DICT_RULES = Pattern.compile("^dictrules\\s+(.*)$");
  private static final Pattern PHONEMES = Pattern.compile("^phonemes\\s+(\\S+)\\s*$");

  private EspeakNgVoiceFiles() {
  }

  static Optional<Profile> resolve(final Path dataDir, final String espeakVoice) {
    requireNonNull(dataDir, "dataDir");
    String voice = requireNonNull(espeakVoice, "espeakVoice").strip().toLowerCase(ROOT);
    if (voice.isEmpty()) {
      return Optional.empty();
    }
    Path langDir = dataDir.resolve("lang");
    if (!Files.isDirectory(langDir)) {
      return Optional.of(Profile.fromVoiceName(voice, Set.of()));
    }
    try (Stream<Path> files = Files.walk(langDir)) {
      return files
        .filter(Files::isRegularFile)
        .map(EspeakNgVoiceFiles::readProfile)
        .flatMap(Optional::stream)
        .filter(profile -> profile.matches(voice))
        .findFirst()
        .or(() -> Optional.of(Profile.fromVoiceName(voice, Set.of())));
    } catch (IOException | RuntimeException ignored) {
      return Optional.of(Profile.fromVoiceName(voice, Set.of()));
    }
  }

  private static Optional<Profile> readProfile(final Path file) {
    List<String> lines;
    try {
      lines = Files.readAllLines(file, UTF_8);
    } catch (IOException ignored) {
      return Optional.empty();
    }
    List<String> languageIds = new ArrayList<>();
    List<String> phonemeTables = new ArrayList<>();
    Set<Integer> dictRules = Set.of();
    for (String raw : lines) {
      int comment = raw.indexOf("//");
      String line = (comment < 0 ? raw : raw.substring(0, comment)).strip();
      Matcher language = LANGUAGE.matcher(line);
      if (language.matches()) {
        languageIds.add(language.group(1).toLowerCase(ROOT));
        continue;
      }
      Matcher phonemes = PHONEMES.matcher(line);
      if (phonemes.matches()) {
        phonemeTables.add(phonemes.group(1).toLowerCase(ROOT));
        continue;
      }
      Matcher rules = DICT_RULES.matcher(line);
      if (rules.matches()) {
        dictRules = parseDictRules(rules.group(1));
      }
    }
    if (languageIds.isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(
      new Profile(List.copyOf(languageIds), dictRules, List.copyOf(phonemeTables)));
  }

  private static Set<Integer> parseDictRules(final String spec) {
    return Stream.of(spec.split("\\s+"))
      .filter(token -> !token.isEmpty() && token.chars().allMatch(Character::isDigit))
      .map(Integer::parseInt)
      .collect(toUnmodifiableSet());
  }

  record Profile(List<String> languageIds, Set<Integer> dictRules, List<String> phonemeTables) {

    Profile {
      languageIds = List.copyOf(requireNonNull(languageIds, "languageIds"));
      dictRules = Set.copyOf(requireNonNull(dictRules, "dictRules"));
      phonemeTables = List.copyOf(requireNonNull(phonemeTables, "phonemeTables"));
    }

    static Profile fromVoiceName(final String voice, final Set<Integer> dictRules) {
      LinkedHashSet<String> ids = new LinkedHashSet<>();
      ids.add(voice);
      int hyphen = voice.indexOf('-');
      if (hyphen > 0) {
        ids.add(voice.substring(0, hyphen));
      }
      return new Profile(List.copyOf(ids), dictRules, List.copyOf(ids));
    }

    List<String> phonemeTableNames() {
      LinkedHashSet<String> names = new LinkedHashSet<>(this.phonemeTables);
      this.languageIds.forEach(names::add);
      names.add("base");
      return List.copyOf(names);
    }

    boolean matches(final String voice) {
      return this.languageIds.stream().anyMatch(id -> id.equalsIgnoreCase(voice));
    }

    List<String> dictionaryNames() {
      LinkedHashSet<String> names = new LinkedHashSet<>();
      this.languageIds.forEach(id -> this.addDictionaryNames(names, id));
      return List.copyOf(names);
    }

    private void addDictionaryNames(final LinkedHashSet<String> names, final String id) {
      names.add(id);
      int hyphen = id.indexOf('-');
      if (hyphen > 0) {
        names.add(id.substring(0, hyphen));
      }
    }

    boolean allowsCondition(final int condition, final boolean negated) {
      boolean present = this.dictRules.contains(condition);
      return negated != present;
    }
  }
}
