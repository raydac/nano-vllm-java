package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.igormaznitsa.nanollvm.llm.LlmAdvisor;
import com.igormaznitsa.nanollvm.samples.Example.ModelChoice;
import com.igormaznitsa.nanollvm.samples.Example.RagMode;
import com.igormaznitsa.nanollvm.samples.utils.OrderedConsole;
import com.igormaznitsa.nanollvm.samples.utils.SampleModelAssumptions;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ExampleModelMenuTest {

  private static int menuIndexOf(final Path path) {
    List<ModelChoice> menu = Example.menuModels();
    for (int i = 0; i < menu.size(); i++) {
      if (menu.get(i).path().filter(path::equals).isPresent()) {
        return i;
      }
    }
    throw new AssertionError("model not in menu: " + path);
  }

  private static OrderedConsole silentConsole() {
    PrintStream sink = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
    return new OrderedConsole(sink, sink);
  }

  @Test
  void enterSelectsFirstAvailableModel() throws Exception {
    Optional<ModelChoice> first = Example.menuModels().stream()
      .filter(choice -> !choice.missing())
      .findFirst();
    assumeTrue(first.isPresent(), "no bundled model downloaded");
    Path chosen = Example.resolveModel(new String[0], new BufferedReader(new StringReader("\n")));
    assertEquals(first.get().requirePath(), chosen);
  }

  @Test
  void modelMenuSelectsQwenWhenPresent() throws Exception {
    Path qwen = SampleModelAssumptions.requireQwen3();
    Path chosen = Example.resolveModel(
      new String[0],
      new BufferedReader(new StringReader((menuIndexOf(qwen) + 1) + "\n")));
    assertEquals(qwen, chosen);
  }

  @Test
  void skipsMenuWhenCliPathGiven() throws Exception {
    Path qwen = SampleModelAssumptions.requireQwen3();
    Path chosen = Example.resolveModel(
      new String[] {qwen.toString()},
      new BufferedReader(new StringReader("2\n")));
    assertEquals(qwen.toAbsolutePath().normalize(), chosen.toAbsolutePath().normalize());
  }

  @Test
  void modelMenuExit() throws Exception {
    int exit = Example.menuModels().size() + 1;
    assertTrue(
      Example.resolveModel(
        new String[0],
        new BufferedReader(new StringReader(exit + "\n"))) == null);
  }

  @Test
  void modelMenuSelectsGteWhenPresent() throws Exception {
    Path gte = SampleModelAssumptions.requireGteSmallGguf();
    Path chosen = Example.resolveModel(
      new String[0],
      new BufferedReader(new StringReader((menuIndexOf(gte) + 1) + "\n")));
    assertEquals(gte, chosen);
  }

  @Test
  void missingExplicitPathExits() throws Exception {
    assertTrue(
      Example.resolveModel(
        new String[] {"models/definitely-missing-nano-vllm"},
        new BufferedReader(new StringReader(""))) == null);
  }

  @Test
  void debugFlagIsNotTreatedAsModelPath() throws Exception {
    int exit = Example.menuModels().size() + 1;
    assertTrue(
      Example.resolveModel(
        new String[] {"--debug"},
        new BufferedReader(new StringReader(exit + "\n"))) == null);
  }

  @Test
  void debugFlagDoesNotHideExplicitModelPath() throws Exception {
    Path qwen = SampleModelAssumptions.requireQwen3();
    Path chosen = Example.resolveModel(
      new String[] {"--debug", qwen.toString()},
      new BufferedReader(new StringReader("2\n")));
    assertEquals(qwen.toAbsolutePath().normalize(), chosen.toAbsolutePath().normalize());

    Path chosenAfter = Example.resolveModel(
      new String[] {qwen.toString(), "--debug"},
      new BufferedReader(new StringReader("2\n")));
    assertEquals(qwen.toAbsolutePath().normalize(), chosenAfter.toAbsolutePath().normalize());
  }

  @Test
  void ragMenuSelectsNone() throws Exception {
    Optional<RagMode> mode = Example.selectRagMode(
      new BufferedReader(new StringReader("1\n")),
      silentConsole());
    assertEquals(Optional.of(RagMode.NONE), mode);
  }

  @Test
  void ragMenuEnterSelectsFirst() throws Exception {
    Optional<RagMode> mode = Example.selectRagMode(
      new BufferedReader(new StringReader("\n")),
      silentConsole());
    assertEquals(Optional.of(RagMode.NONE), mode);
  }

  @Test
  void ragMenuExit() throws Exception {
    Optional<RagMode> mode = Example.selectRagMode(
      new BufferedReader(new StringReader("5\n")),
      silentConsole());
    assertTrue(mode.isEmpty());
  }

  @Test
  void ragMenuSelectsBm25WhenCorpusPresent() throws Exception {
    SampleModelAssumptions.requireRag();
    Optional<RagMode> mode = Example.selectRagMode(
      new BufferedReader(new StringReader("2\n")),
      silentConsole());
    assertEquals(Optional.of(RagMode.BM25), mode);
  }

  @Test
  void advisorMenuSelectsNoneAndThree() throws Exception {
    assertEquals(
      Optional.of(0),
      Example.selectAdvisorCount(new BufferedReader(new StringReader("0\n")), silentConsole()));
    assertEquals(
      Optional.of(3),
      Example.selectAdvisorCount(new BufferedReader(new StringReader("3\n")), silentConsole()));
  }

  @Test
  void advisorMenuEnterSelectsNone() throws Exception {
    assertEquals(
      Optional.of(0),
      Example.selectAdvisorCount(new BufferedReader(new StringReader("\n")), silentConsole()));
  }

  @Test
  void advisorMenuExit() throws Exception {
    assertTrue(
      Example.selectAdvisorCount(new BufferedReader(new StringReader("4\n")), silentConsole())
        .isEmpty());
  }

  @Test
  void advisorsForCountTakesRolesInOrder() {
    assertTrue(Example.advisorsForCount(0).isEmpty());
    assertEquals(List.of("Practical"),
      Example.advisorsForCount(1).stream().map(LlmAdvisor::name).toList());
    assertEquals(List.of("Practical", "Abstract", "Consequence"),
      Example.advisorsForCount(3).stream().map(LlmAdvisor::name).toList());
  }
}
