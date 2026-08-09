package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.BundledModelAssumptions;
import java.io.BufferedReader;
import java.io.StringReader;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class ExampleModelMenuTest {

  @Test
  void modelMenuSelectsQwenWhenPresent() throws Exception {
    Path qwen = BundledModelAssumptions.requireQwen3();
    Path chosen = Example.resolveModel(new String[0], new BufferedReader(new StringReader("1\n")));
    assertEquals(qwen, chosen);
  }

  @Test
  void modelMenuExit() throws Exception {
    assertTrue(
      Example.resolveModel(new String[0], new BufferedReader(new StringReader("5\n"))) == null);
  }

  @Test
  void modelMenuSelectsGteWhenPresent() throws Exception {
    Path gte = BundledModelAssumptions.requireGteSmallGguf();
    Path chosen = Example.resolveModel(new String[0], new BufferedReader(new StringReader("4\n")));
    assertEquals(gte, chosen);
  }

  @Test
  void skipsMenuWhenCliPathGiven() throws Exception {
    Path qwen = BundledModelAssumptions.requireQwen3();
    Path chosen = Example.resolveModel(
      new String[] {qwen.toString()},
      new BufferedReader(new StringReader("2\n")));
    assertEquals(qwen.toAbsolutePath().normalize(), chosen.toAbsolutePath().normalize());
  }
}
