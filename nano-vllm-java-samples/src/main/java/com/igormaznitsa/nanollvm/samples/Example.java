package com.igormaznitsa.nanollvm.samples;

import com.igormaznitsa.nanollvm.samples.utils.OrderedConsole;
import java.io.BufferedReader;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Interactive sample: Lanterna TUI (default) or classic console ({@code --cli}) for causal chat /
 * {@link com.igormaznitsa.nanollvm.rag.RagSession}, or an embedding REPL for BERT GGUF.
 */
public final class Example {

  private Example() {
  }

  public static void main(final String[] args) throws Exception {
    boolean cli = hasFlag(args, "--cli");
    String[] rest = stripFlag(args, "--cli");
    if (cli) {
      ExampleCli.run(rest);
      return;
    }
    try {
      ExampleTui.run(rest);
    } catch (Exception ex) {
      System.err.println(
        "Lanterna TUI failed (" + ex.getMessage() + "). Falling back to --cli console.");
      ExampleCli.run(rest);
    }
  }

  static Path resolveModel(final String[] args, final BufferedReader in) throws Exception {
    return ExampleCli.resolveModel(args, in, new OrderedConsole(System.out, System.err));
  }

  private static boolean hasFlag(final String[] args, final String flag) {
    if (args == null) {
      return false;
    }
    return Arrays.stream(args).anyMatch(arg -> flag.equalsIgnoreCase(arg));
  }

  private static String[] stripFlag(final String[] args, final String flag) {
    if (args == null || args.length == 0) {
      return new String[0];
    }
    return Arrays.stream(args)
      .filter(arg -> arg == null || !flag.equalsIgnoreCase(arg))
      .toArray(String[]::new);
  }
}
