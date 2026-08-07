package com.igormaznitsa.nanollvm.samples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.io.BufferedReader;
import java.io.StringReader;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class ExampleModelMenuTest {

  @Test
  void modelMenuSelectsBundledPaths() throws Exception {
    var qwen = BundledModels.find(BundledModels.QWEN3_0_6B);
    Assumptions.assumeTrue(qwen.isPresent());

    Path chosen = Example.resolveModel(new String[0], new BufferedReader(new StringReader("1\n")));
    assertEquals(qwen.get(), chosen);

    assertTrue(
      Example.resolveModel(new String[0], new BufferedReader(new StringReader("4\n"))) == null);
  }

  @Test
  void skipsMenuWhenCliPathGiven() throws Exception {
    var qwen = BundledModels.require(BundledModels.QWEN3_0_6B);
    Path chosen = Example.resolveModel(
      new String[] {qwen.toString()},
      new BufferedReader(new StringReader("2\n")));
    assertEquals(qwen.toAbsolutePath().normalize(), chosen.toAbsolutePath().normalize());
  }
}
