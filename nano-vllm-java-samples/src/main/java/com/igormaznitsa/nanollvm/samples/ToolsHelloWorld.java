package com.igormaznitsa.nanollvm.samples;

import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.ChatSession;
import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.llm.ModelTool;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.samples.utils.BundledModels;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Collator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * Asks a tool-capable chat model for the OS name, then for the names in one directory.
 *
 * <p>The model does not see the disk until it calls {@code os_name} or {@code list_files}.
 * Default checkpoint is Qwen3-0.6B (Hermes tool tokens). Gemma 3 cannot take these tools.
 * Args: optional model directory, then an optional directory to list (default: the working
 * directory). From the repository root:
 * {@code mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.ToolsHelloWorld}
 */
public final class ToolsHelloWorld {

  static final int MAX_LISTED = 100;

  private static final String EMPTY_SCHEMA = "{\"type\":\"object\",\"properties\":{}}";
  private static final String LIST_SCHEMA = """
    {"type":"object","properties":{"directory":{"type":"string","description":"Directory to list. Omit it to use the current working directory."}}}""";

  private ToolsHelloWorld() {
  }

  public static void main(final String[] args) {
    Path modelDir = args != null && args.length > 0 && args[0] != null && !args[0].isBlank()
      ? Path.of(args[0]).toAbsolutePath().normalize()
      : BundledModels.require(BundledModels.QWEN3_0_6B);
    String directory = args != null && args.length > 1 && args[1] != null && !args[1].isBlank()
      ? args[1].strip()
      : "";

    System.out.println("Loading model from " + modelDir);

    try (LlmModel model = LlmModelFactory.make(modelDir);
         LLM llm = LLM.builder(model)
           .systemPrompt("""
             Call the tool named in the question before you answer.
             Do not invent an operating system name or file names.""")
           .tool(osNameTool(), listFilesTool())
           .build()) {
      ChatSession chat = llm.chat()
        .enableThinking(false)
        .maxTokens(256)
        .deterministic();
      System.out.println();
      System.out.println(chat.send(
        "What operating system is this?").answer());
      System.out.println();
      String files = directory.isEmpty()
        ?
        "Which files and folders are in the current working directory?"
        : "Which files and folders are in " + directory + "?";
      System.out.println(chat.send(files).answer());
    }
  }

  /**
   * Tool whose result is {@link #operatingSystem()}. It takes no arguments.
   *
   * @return the {@code os_name} function registered on the engine
   */
  static ModelTool osNameTool() {
    return ModelTool.of(
      "os_name",
      "Operating system name of this machine.",
      EMPTY_SCHEMA,
      arguments -> {
        System.err.println("Requested OS name: " + arguments);
        return report("os_name", operatingSystem());
      });
  }

  /**
   * Tool whose result is {@link #listFiles(String)}. The optional {@code directory} argument
   * selects the folder; omitting it lists the working directory.
   *
   * @return the {@code list_files} function registered on the engine
   */
  static ModelTool listFilesTool() {
    return ModelTool.of(
      "list_files",
      "Names of files and folders in one directory. Folders end with a slash. Does not read file contents.",
      LIST_SCHEMA,
      arguments -> {
        System.err.println("Requested list files: " + arguments);
        return report("list_files", listFiles(arguments));
      });
  }

  /**
   * Operating-system name from the runtime, with the CPU architecture in parentheses when that
   * property is set.
   *
   * @return {@code os.name}, then {@code (os.arch)} when {@code os.arch} is non-blank
   */
  static String operatingSystem() {
    String name = System.getProperty("os.name", "unknown").strip();
    String arch = System.getProperty("os.arch", "").strip();
    return arch.isEmpty() ? name : name + " (" + arch + ")";
  }

  /**
   * Names in one directory, one per line. A folder name ends with {@code /}. File contents are
   * not read. At most {@link #MAX_LISTED} names are returned; a longer directory ends with
   * {@code … more}.
   *
   * @param argumentsJson JSON object; a missing or blank {@code directory} is the working directory
   * @return the listing, {@code empty} when the directory has no entries, or {@code error: ...}
   * when the path is not a directory or cannot be listed
   */
  static String listFiles(final String argumentsJson) {
    Path directory = resolveDirectory(argumentsJson);
    if (!Files.isDirectory(directory)) {
      return "error: not a directory: " + directory;
    }

    try (Stream<Path> entries = Files.list(directory)) {
      List<String> names = entries
        .map(ToolsHelloWorld::entryName)
        .sorted(Collator.getInstance(Locale.ROOT))
        .limit(MAX_LISTED + 1L)
        .toList();
      if (names.isEmpty()) {
        return "empty";
      }
      if (names.size() <= MAX_LISTED) {
        return String.join("\n", names);
      }
      return String.join("\n", names.subList(0, MAX_LISTED)) + "\n… more";
    } catch (IOException ex) {
      String message = ex.getMessage();
      return "error: " + (message == null || message.isBlank() ? "cannot list directory" : message);
    }
  }

  /**
   * Absolute directory named by the {@code directory} string in {@code argumentsJson}. A null,
   * blank, or non-string value is the working directory. A relative path resolves against that
   * directory.
   *
   * @param argumentsJson JSON object text from the model; may be {@code null}
   * @return absolute normalized path; the directory is not required to exist
   */
  static Path resolveDirectory(final String argumentsJson) {
    String raw = directoryArgument(argumentsJson).strip();
    Path path =
      raw.isEmpty() ? Path.of(requireNonNull(System.getProperty("user.dir"))) : Path.of(raw);
    if (!path.isAbsolute()) {
      path = Path.of(requireNonNull(System.getProperty("user.dir"))).resolve(path);
    }
    return path.toAbsolutePath().normalize();
  }

  /**
   * Prints the tool name and its result on the standard error stream, then returns that result
   * so the model receives the same text.
   *
   * @param tool  registered tool name
   * @param value text returned to the model
   * @return {@code value}
   */
  private static String report(final String tool, final String value) {
    System.err.println("[tool] " + tool);
    System.err.println(value);
    return value;
  }

  /**
   * Last path segment of {@code path}. A directory gets a trailing slash so the model can tell
   * folders from files without another call.
   *
   * @param path one directory entry
   * @return the file name, or the file name plus {@code /} when {@code path} is a directory
   */
  private static String entryName(final Path path) {
    String name = path.getFileName() == null ? path.toString() : path.getFileName().toString();
    return Files.isDirectory(path) ? name + "/" : name;
  }

  /**
   * Value of the {@code directory} string property. Absent, non-string, and unreadable values
   * are {@code ""}, which {@link #resolveDirectory(String)} treats as the working directory.
   *
   * @param argumentsJson JSON object text; may be {@code null}
   * @return the raw directory string, without surrounding quotes
   */
  private static String directoryArgument(final String argumentsJson) {
    if (argumentsJson == null || argumentsJson.isBlank()) {
      return "";
    }
    String key = "\"directory\"";
    int keyAt = argumentsJson.indexOf(key);
    if (keyAt < 0) {
      return "";
    }
    int colon = argumentsJson.indexOf(':', keyAt + key.length());
    if (colon < 0) {
      return "";
    }
    int cursor = colon + 1;
    while (cursor < argumentsJson.length() &&
      Character.isWhitespace(argumentsJson.charAt(cursor))) {
      cursor++;
    }
    if (cursor >= argumentsJson.length() || argumentsJson.charAt(cursor) != '"') {
      return "";
    }
    return jsonString(argumentsJson, cursor + 1);
  }

  /**
   * JSON string contents beginning at {@code start}, which is the index after the opening quote.
   * A string that never closes returns the characters collected so far. Recognizes {@code \n},
   * {@code \r}, {@code \t}, {@code \"}, {@code \\}, and {@code \/}.
   *
   * @param text  arguments JSON
   * @param start index of the first character inside the string
   * @return decoded string value
   */
  private static String jsonString(final String text, final int start) {
    StringBuilder value = new StringBuilder();
    for (int i = start; i < text.length(); i++) {
      char ch = text.charAt(i);
      if (ch == '"') {
        return value.toString();
      }
      if (ch == '\\' && i + 1 < text.length()) {
        char escaped = text.charAt(++i);
        value.append(switch (escaped) {
          case 'n' -> '\n';
          case 'r' -> '\r';
          case 't' -> '\t';
          case '"', '\\', '/' -> escaped;
          default -> escaped;
        });
        continue;
      }
      value.append(ch);
    }
    return value.toString();
  }
}
