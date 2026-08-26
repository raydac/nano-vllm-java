package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class EspeakNgDictSourceTest {

  @Test
  void missingDictsourceIsNotLoaded(@TempDir final Path dir) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang"));
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    assertFalse(source.isLoaded());
    assertEquals(Optional.empty(), source.phonemesForWord("hello"));
  }

  @Test
  void listEntryWinsOverRules(@TempDir final Path dir) throws Exception {
    Path data = this.writeDict(dir, "hello h@l'oU\n", """
      .group h
       h x
      """);
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    assertTrue(source.isLoaded());
    assertEquals(Optional.of("h@l'oU"), source.phonemesForWord("hello"));
    assertEquals("həlˈoʊ", source.toIpa("Hello"));
  }

  @Test
  void rulesConsumeDigraphs(@TempDir final Path dir) throws Exception {
    Path data = this.writeDict(dir, "", """
      .group s
       s s
       sh S
      .group o
       o 0
      .group p
       p p
      """);
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    assertEquals(Optional.of("S0p"), source.phonemesForWord("shop"));
    assertEquals("ʃɒp", source.toIpa("shop"));
  }

  @Test
  void conditionalListUsesVoiceDictrules(@TempDir final Path dir) throws Exception {
    Path data = this.writeDict(dir, """
      hello h@lo
      ?3 hello h@l'oU
      """, ".group a\n a a\n");
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    assertEquals(Optional.of("h@l'oU"), source.phonemesForWord("hello"));
  }

  @Test
  void suffixRuleRetranslatesStem(@TempDir final Path dir) throws Exception {
    Path data = this.writeDict(dir, "", """
      .group w
       w w
      .group a
       a a
      .group l
       l l
      .group k
       k k
      .group i
       i I
       ing (_S3 IN
      .group n
       n n
      .group g
       g g
      """);
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    assertEquals(Optional.of("walkIN"), source.phonemesForWord("walking"));
  }

  @Test
  void prefixRuleRetranslatesRest(@TempDir final Path dir) throws Exception {
    Path data = this.writeDict(dir, "", """
      .group u
       _) un (P2 %Vn
      .group d
       d d
      .group o
       o 0
      """);
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    assertEquals(Optional.of("%Vnd0"), source.phonemesForWord("undo"));
  }

  @Test
  void numbersUseListFragments(@TempDir final Path dir) throws Exception {
    Path data = this.writeDict(dir, """
      _1 w'02n
      _2 t'u:
      _0C h'VndrI2d
      _2X tw'Ent2i||
      _0M1 T'aUz@nd
      """, ".group a\n a a\n");
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    String twentyOne = source.phonemesForWord("21").orElse("");
    assertTrue(twentyOne.contains("tw'Ent2i"), twentyOne);
    assertTrue(twentyOne.contains("w'02n"), twentyOne);
    String hundred = source.phonemesForWord("100").orElse("");
    assertTrue(hundred.contains("w'02n"), hundred);
    assertTrue(hundred.contains("h'VndrI2d"), hundred);
    String thousand = source.toIpa("1000");
    assertTrue(thousand.contains("w"), thousand);
  }

  @Test
  void compiledDictLooksUpHelloWhenDictsourceMissing() {
    Path data = Path.of("/usr/lib/x86_64-linux-gnu/espeak-ng-data");
    assumeTrue(
      Files.isRegularFile(data.resolve("phontab")) && Files.isRegularFile(data.resolve("en_dict")));
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "en-us");
    assertTrue(source.isLoaded());
    Optional<String> hello = source.phonemesForWord("hello");
    assertTrue(hello.isPresent(), "compiled en_dict should contain hello");
    assertTrue(hello.get().contains("h"), hello.toString());
    assertTrue(hello.get().contains("@") || hello.get().contains("e"), hello.toString());
  }

  @Test
  void compiledRussianLooksUpMoloko() {
    Path data = OptionalModelAssumptions.requirePiperRussian().resolve("espeak-ng-data");
    assumeTrue(
      Files.isRegularFile(data.resolve("ru_dict")) && Files.isRegularFile(data.resolve("phontab")),
      "copy compiled ru_dict/phontab into the voice espeak-ng-data");
    EspeakNgCompiledDict compiled = EspeakNgCompiledDict.load(
      data, List.of("ru"), List.of("ru", "base"), Set.of());
    assertTrue(compiled.isLoaded());
    Optional<EspeakNgCompiledDict.Hit> milk = compiled.lookupHit("молоко");
    assertTrue(milk.isPresent(), "ru_dict should contain молоко");
    assertEquals(3, milk.get().stressSyllable());
  }

  @Test
  void bundledDictsourcePhonemizesVoiceLanguage() {
    Path model = OptionalModelAssumptions.requirePiper();
    Path data = model.resolve("espeak-ng-data");
    assumeTrue(
      Files.isDirectory(data.resolve("dictsource")),
      "re-run models/download-piper-*.sh so espeak-ng-data includes dictsource/");
    boolean english = model.getFileName().toString().contains("-en-");
    EspeakNgDictSource source = EspeakNgDictSource.load(data, english ? "en-us" : "ru");
    assertTrue(source.isLoaded());
    String ipa = source.toIpa(english ? "Hello" : "Привет");
    assertFalse(ipa.isBlank(), () -> "empty IPA for " + model.getFileName());
    if (english) {
      assertTrue(ipa.contains("h"), ipa);
      assertTrue(ipa.contains("l"), ipa);
    }
  }

  @Test
  void bundledRussianDictsourceSoundsRussian() {
    Path model = OptionalModelAssumptions.requirePiperRussian();
    Path data = model.resolve("espeak-ng-data");
    assumeTrue(
      Files.isDirectory(data.resolve("dictsource")),
      "re-run models/download-piper-ru-irina-medium.sh so espeak-ng-data includes dictsource/");
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "ru");
    assertTrue(source.isLoaded());
    String hello = source.toIpa("Привет");
    assertTrue(hello.contains("ʲ"), hello);
    assertTrue(hello.contains("ˈ"), hello);
    assertFalse(hello.contains("y"), hello);
    String milk = source.toIpa("молоко");
    assertTrue(milk.contains("ʌ"), milk);
    String eto = source.toIpa("это");
    assertTrue(eto.contains("ˈ"), eto);
    assertFalse(eto.contains("tˈ"), eto);
    String mama = source.toIpa("мама");
    assertTrue(mama.contains("ˈ"), mama);
    assertFalse(mama.contains("mʌmˈ"), mama);
    assertFalse(mama.contains("mamˈ"), mama);
    if (Files.isRegularFile(data.resolve("ru_dict")) &&
      Files.isRegularFile(data.resolve("phontab"))) {
      assertFalse(milk.contains("mˈ") || milk.startsWith("ˈ"), milk);
    }
    String shi = source.toIpa("ши");
    assertTrue(shi.contains("y"), shi);
    String ge = source.toIpa("г");
    assertTrue(ge.contains("ɡ"), ge);
    assertFalse(ge.indexOf('g') >= 0, ge);
    String city = source.toIpa("город");
    assertTrue(city.contains("ɡ"), city);
    assertFalse(city.indexOf('g') >= 0, city);
  }

  @Test
  void russianLetterGroupsKeepIAfterSoftConsonant(@TempDir final Path dir) throws Exception {
    Path data = this.writeRuDict(dir, "", """
      .group р
       р r
      .group и
       H) и y
       и i
      .group ш
       ш S
      """);
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "ru");
    String ri = source.phonemesForWord("ри").orElse("");
    assertTrue(ri.contains("i"), ri);
    assertFalse(ri.contains("y"), ri);
    String shi = source.phonemesForWord("ши").orElse("");
    assertTrue(shi.contains("y"), shi);
  }

  @Test
  void russianYGroupPalatalizesBeforeJotated(@TempDir final Path dir) throws Exception {
    Path data = this.writeRuDict(dir, "", """
      .group б
       б (Y    b;
       б       b
      .group я
       я       V
      """);
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "ru");
    String bya = source.phonemesForWord("бя").orElse("");
    assertTrue(bya.contains(";"), bya);
  }

  @Test
  void russianListStressMarksFirstSyllable(@TempDir final Path dir) throws Exception {
    Path data = this.writeRuDict(dir, "это $1\nмама $1\n", """
      .group э
       э E2
      .group т
       т t
      .group о
       о o
      .group м
       м m
      .group а
       а V
       _) а a
       а (_ a
      """);
    EspeakNgDictSource source = EspeakNgDictSource.load(data, "ru");
    String eto = source.toIpa("это");
    assertTrue(eto.contains("ˈ"), eto);
    assertFalse(eto.contains("tˈ"), eto);
    String mama = source.toIpa("мама");
    assertTrue(mama.contains("ˈ"), mama);
    assertFalse(mama.contains("mʌmˈ"), mama);
    assertFalse(mama.contains("mamˈ"), mama);
  }

  private Path writeDict(final Path dir, final String list, final String rules) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang/gmw"));
    Files.writeString(data.resolve("lang/gmw/en-US"), """
      language en-us
      dictrules 3
      """);
    Path dictsource = data.resolve("dictsource");
    Files.createDirectories(dictsource);
    Files.writeString(dictsource.resolve("en_list"), list);
    Files.writeString(dictsource.resolve("en_rules"), rules);
    return data;
  }

  private Path writeRuDict(final Path dir, final String list, final String rules) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang/zle"));
    Files.writeString(data.resolve("lang/zle/ru"), """
      language ru
      """);
    Path dictsource = data.resolve("dictsource");
    Files.createDirectories(dictsource);
    Files.writeString(dictsource.resolve("ru_list"), list);
    Files.writeString(dictsource.resolve("ru_rules"), rules);
    return data;
  }
}
