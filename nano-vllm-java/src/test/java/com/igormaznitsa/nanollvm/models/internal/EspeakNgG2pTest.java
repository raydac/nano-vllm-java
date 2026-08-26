package com.igormaznitsa.nanollvm.models.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.llm.Config;
import com.igormaznitsa.nanollvm.testsupport.OptionalModelAssumptions;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class EspeakNgG2pTest {

  @Test
  void russianLettersMapToPhonemeIds(@TempDir final Path dir) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang"));
    Map<String, List<Integer>> ids = Map.of(
      "^", List.of(1),
      "$", List.of(2),
      "_", List.of(0),
      "m", List.of(10),
      "a", List.of(11));
    assertEquals(
      List.of(1, 0, 10, 0, 11, 0, 10, 0, 11, 0, 2),
      new EspeakNgG2p(ids, data).phonemeIds("мама"));
  }

  @Test
  void zhAndShUseEspeakPhones(@TempDir final Path dir) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang"));
    Map<String, List<Integer>> ids = Map.ofEntries(
      Map.entry("^", List.of(1)),
      Map.entry("$", List.of(2)),
      Map.entry("_", List.of(0)),
      Map.entry(" ", List.of(3)),
      Map.entry("o", List.of(14)),
      Map.entry("d", List.of(17)),
      Map.entry("n", List.of(26)),
      Map.entry("a", List.of(11)),
      Map.entry("ʒ", List.of(108)),
      Map.entry("ʐ", List.of(106)),
      Map.entry("y", List.of(37)),
      Map.entry("ɨ", List.of(73)),
      Map.entry("v", List.of(34)),
      Map.entry("ʃ", List.of(96)),
      Map.entry("ʂ", List.of(95)),
      Map.entry("e", List.of(18)),
      Map.entry("l", List.of(24)),
      Map.entry("ʲ", List.of(119)));
    EspeakNgG2p g2p = new EspeakNgG2p(ids, data);
    List<Integer> once = g2p.phonemeIds("однажды");
    List<Integer> left = g2p.phonemeIds("вышел");
    assertTrue(once.contains(108), once::toString);
    assertFalse(once.contains(106), once::toString);
    assertTrue(left.contains(96), left::toString);
    assertFalse(left.contains(95), left::toString);
    assertTrue(left.contains(37), left::toString);
    assertFalse(left.contains(73), left::toString);
    assertFalse(left.contains(119), left::toString);
  }

  @Test
  void russianPhonologyFollowsEspeakRules(@TempDir final Path dir) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang"));
    Map<String, List<Integer>> ids = Map.ofEntries(
      Map.entry("^", List.of(1)),
      Map.entry("$", List.of(2)),
      Map.entry("_", List.of(0)),
      Map.entry(" ", List.of(3)),
      Map.entry("a", List.of(11)),
      Map.entry("d", List.of(17)),
      Map.entry("e", List.of(18)),
      Map.entry("f", List.of(19)),
      Map.entry("g", List.of(20)),
      Map.entry("i", List.of(21)),
      Map.entry("l", List.of(24)),
      Map.entry("m", List.of(10)),
      Map.entry("o", List.of(14)),
      Map.entry("r", List.of(28)),
      Map.entry("s", List.of(29)),
      Map.entry("t", List.of(30)),
      Map.entry("v", List.of(34)),
      Map.entry("y", List.of(37)),
      Map.entry("z", List.of(38)),
      Map.entry("ʃ", List.of(96)),
      Map.entry("ʒ", List.of(108)));
    EspeakNgG2p g2p = new EspeakNgG2p(ids, data);
    List<Integer> frost = g2p.phonemeIds("мороз");
    assertTrue(frost.contains(29), frost::toString);
    assertFalse(frost.contains(38), frost::toString);
    List<Integer> inGarden = g2p.phonemeIds("в саду");
    assertTrue(inGarden.contains(19), inGarden::toString);
    List<Integer> inHouse = g2p.phonemeIds("в доме");
    assertTrue(inHouse.contains(34), inHouse::toString);
    assertFalse(inHouse.contains(19), inHouse::toString);
    List<Integer> lived = g2p.phonemeIds("жил");
    assertTrue(lived.contains(37), lived::toString);
    assertFalse(lived.contains(21), lived::toString);
    List<Integer> his = g2p.phonemeIds("его");
    assertTrue(his.contains(34), his::toString);
    assertFalse(his.contains(20), his::toString);
    List<Integer> what = g2p.phonemeIds("что");
    assertTrue(what.contains(96), what::toString);
  }

  @Test
  void englishHelloUsesEspeakPhones(@TempDir final Path dir) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang"));
    Map<String, List<Integer>> ids = Map.ofEntries(
      Map.entry("^", List.of(1)),
      Map.entry("$", List.of(2)),
      Map.entry("_", List.of(0)),
      Map.entry(" ", List.of(3)),
      Map.entry("h", List.of(40)),
      Map.entry("ə", List.of(41)),
      Map.entry("l", List.of(24)),
      Map.entry("ˈ", List.of(42)),
      Map.entry("oʊ", List.of(43)),
      Map.entry("o", List.of(14)),
      Map.entry("w", List.of(44)),
      Map.entry("ɝ", List.of(45)),
      Map.entry("d", List.of(17)),
      Map.entry("θ", List.of(46)),
      Map.entry("ʃ", List.of(96)));
    EspeakNgG2p g2p = new EspeakNgG2p(ids, data, "en-us");
    List<Integer> hello = g2p.phonemeIds("Hello world");
    assertTrue(hello.contains(40), hello::toString);
    assertTrue(hello.contains(24), hello::toString);
    assertTrue(hello.contains(43) || hello.contains(14), hello::toString);
    List<Integer> shop = g2p.phonemeIds("shop");
    assertTrue(shop.contains(96), shop::toString);
    assertFalse(shop.contains(40), shop::toString);
  }

  @Test
  void missingDataDirIsIgnored(@TempDir final Path dir) {
    Map<String, List<Integer>> ids = Map.of("^", List.of(1), "$", List.of(2), "_", List.of(0));
    EspeakNgG2p g2p = new EspeakNgG2p(ids, dir.resolve("missing"));
    assertFalse(g2p.hasEspeakData());
    assertEquals(List.of(1, 0, 2), g2p.phonemeIds("а"));
  }

  @Test
  void russianGeUsesIpaVelar(@TempDir final Path dir) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang"));
    Map<String, List<Integer>> ids = Map.ofEntries(
      Map.entry("^", List.of(1)),
      Map.entry("$", List.of(2)),
      Map.entry("_", List.of(0)),
      Map.entry("a", List.of(14)),
      Map.entry("ɡ", List.of(66)),
      Map.entry("g", List.of(20)));
    EspeakNgG2p g2p = new EspeakNgG2p(ids, data);
    List<Integer> ga = g2p.phonemeIds("га");
    assertTrue(ga.contains(66), ga::toString);
    assertFalse(ga.contains(20), ga::toString);
  }

  @Test
  void irinaAlphabetKeepsVelarG() throws Exception {
    Path model = OptionalModelAssumptions.requirePiperRussian();
    Path json;
    try (Stream<Path> files = Files.list(model)) {
      json = files
        .filter(path -> path.getFileName().toString().endsWith(".onnx.json"))
        .findFirst()
        .orElseThrow();
    }
    Map<String, List<Integer>> ids =
      Config.HfConfig.fromPiperJson(Files.readString(json)).piper().phonemeIdMap();
    EspeakNgG2p g2p = new EspeakNgG2p(ids, model.resolve("espeak-ng-data"), "ru");
    List<Integer> ge = g2p.phonemeIds("Г");
    assertTrue(ge.contains(66), ge::toString);
    List<Integer> alphabet = g2p.phonemeIds(
      "А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я");
    assertTrue(alphabet.contains(66), alphabet::toString);
  }

  @Test
  void lessacPangramKeepsVelarAndDropsDigitTwo() throws Exception {
    Path model = OptionalModelAssumptions.requirePiper();
    assumeTrue(
      model.getFileName().toString().contains("-en-"),
      "needs models/piper-en-lessac-medium");
    Path json;
    try (Stream<Path> files = Files.list(model)) {
      json = files
        .filter(path -> path.getFileName().toString().endsWith(".onnx.json"))
        .findFirst()
        .orElseThrow();
    }
    Map<String, List<Integer>> ids =
      Config.HfConfig.fromPiperJson(Files.readString(json)).piper().phonemeIdMap();
    EspeakNgG2p g2p = new EspeakNgG2p(ids, model.resolve("espeak-ng-data"), "en-us");
    List<Integer> pangram = g2p.phonemeIds("The quick brown fox jumps over the lazy dog");
    assertTrue(pangram.contains(66), pangram::toString);
    assertFalse(pangram.contains(132), pangram::toString);
    assertTrue(pangram.contains(88), pangram::toString);
    assertTrue(pangram.contains(60) || pangram.contains(62), pangram::toString);
    assertTrue(pangram.contains(120), pangram::toString);
  }

  @Test
  void dictsourceListAndRulesMapToPhonemeIds(@TempDir final Path dir) throws Exception {
    Path data = this.writeMiniDict(dir);
    Map<String, List<Integer>> ids = Map.ofEntries(
      Map.entry("^", List.of(1)),
      Map.entry("$", List.of(2)),
      Map.entry("_", List.of(0)),
      Map.entry("h", List.of(40)),
      Map.entry("ə", List.of(41)),
      Map.entry("l", List.of(24)),
      Map.entry("ˈ", List.of(42)),
      Map.entry("o", List.of(14)),
      Map.entry("ʊ", List.of(47)),
      Map.entry("n", List.of(26)),
      Map.entry("a", List.of(11)),
      Map.entry("ɪ", List.of(48)),
      Map.entry("t", List.of(30)),
      Map.entry("ʃ", List.of(96)),
      Map.entry("p", List.of(28)));
    EspeakNgG2p g2p = new EspeakNgG2p(ids, data, "en-us");
    List<Integer> hello = g2p.phonemeIds("Hello");
    assertTrue(hello.contains(40), hello::toString);
    assertTrue(hello.contains(41), hello::toString);
    assertTrue(hello.contains(42), hello::toString);
    List<Integer> shop = g2p.phonemeIds("shop");
    assertTrue(shop.contains(96), shop::toString);
    assertTrue(shop.contains(28), shop::toString);
    List<Integer> knight = g2p.phonemeIds("knight");
    assertTrue(knight.contains(26), knight::toString);
    assertFalse(knight.contains(40), knight::toString);
  }

  @Test
  void numbersMapToPhonemeIds(@TempDir final Path dir) throws Exception {
    Path data = this.writeMiniDict(dir);
    Map<String, List<Integer>> ids = Map.ofEntries(
      Map.entry("^", List.of(1)),
      Map.entry("$", List.of(2)),
      Map.entry("_", List.of(0)),
      Map.entry(" ", List.of(3)),
      Map.entry("t", List.of(30)),
      Map.entry("w", List.of(44)),
      Map.entry("ˈ", List.of(42)),
      Map.entry("ɛ", List.of(49)),
      Map.entry("n", List.of(26)),
      Map.entry("i", List.of(21)),
      Map.entry("ʌ", List.of(50)),
      Map.entry("u", List.of(32)),
      Map.entry("ː", List.of(51)));
    EspeakNgG2p g2p = new EspeakNgG2p(ids, data, "en-us");
    List<Integer> twentyOne = g2p.phonemeIds("21");
    assertTrue(twentyOne.contains(30) || twentyOne.contains(44), twentyOne::toString);
  }

  private Path writeMiniDict(final Path dir) throws Exception {
    Path data = dir.resolve("espeak-ng-data");
    Files.createDirectories(data.resolve("lang/gmw"));
    Files.writeString(data.resolve("lang/gmw/en-US"), """
      name English (America)
      language en-us
      dictrules 3
      """);
    Path dictsource = data.resolve("dictsource");
    Files.createDirectories(dictsource);
    Files.writeString(dictsource.resolve("en_list"), """
      hello h@l'oU
      knight n'aIt
      _1 w'02n
      _2 t'u:
      _2X tw'Ent2i||
      """);
    Files.writeString(dictsource.resolve("en_rules"), """
      .group h
       h h
      .group e
       e E
      .group l
       l l
      .group o
       o 0
       oU oU
      .group s
       s s
       sh S
      .group p
       p p
      .group t
       t t
       th T
      """);
    return data;
  }
}
