package com.igormaznitsa.nanollvm.models.internal.yolo;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.chat.LlmListener;
import com.igormaznitsa.nanollvm.chat.LlmListeners;
import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxDataTypes;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxGraphBundle;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxProtoReader.OnnxTensorProto;
import com.igormaznitsa.nanollvm.models.llmcontainer.OnnxWeightReader;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Loads a fused YOLO26 ONNX file and the class names beside it.
 */
public final class Yolo26Loader {

  private Yolo26Loader() {
  }

  public static Yolo26Model load(final Path onnxFile, final LlmListener io) throws IOException {
    Path file = requireNonNull(onnxFile, "onnxFile").toAbsolutePath().normalize();
    Path fileName = file.getFileName();
    if (fileName == null) {
      throw new ModelLoadException("not a YOLO26 ONNX file: " + file);
    }
    YoloTask task = YoloTask.fromFileName(fileName.toString());
    if (task == null) {
      throw new ModelLoadException("not a YOLO26 ONNX file: " + file);
    }
    return load(file, task, io);
  }

  public static Yolo26Model load(
      final Path onnxFile,
      final YoloTask task,
      final LlmListener io
  ) throws IOException {
    Path file = requireNonNull(onnxFile, "onnxFile").toAbsolutePath().normalize();
    requireNonNull(task, "task");
    LlmListener streams = io == null ? LlmListeners.silent() : io;
    LlmListeners.info(streams, null, "Loading YOLO26 " + task.wireName() + " weights…");
    Map<String, Tensor> named = readInitializers(file);
    YoloTensors tensors = YoloTensors.of(named);
    List<String> names = classNames(file.getParent(), task, classCount(tensors, task));
    return Yolo26Model.load(tensors, task, names);
  }

  private static Map<String, Tensor> readInitializers(final Path file) throws IOException {
    OnnxGraphBundle graph = OnnxProtoReader.readGraph(file);
    Map<String, Tensor> named = new LinkedHashMap<>();
    for (OnnxTensorProto proto : graph.initializers()) {
      if (!OnnxDataTypes.shouldLoadAsWeight(proto)) {
        OnnxDataTypes.requireHandledOrSkip(proto);
        continue;
      }
      String name = proto.name() == null ? "" : proto.name().replace('/', '.');
      named.put(name, OnnxWeightReader.toTensor(proto, file.getParent()));
    }
    if (named.isEmpty()) {
      throw new ModelLoadException("YOLO26 ONNX has no floating initializers: " + file);
    }
    return named;
  }

  static List<String> classNames(final Path folder, final YoloTask task, final int classes)
    throws IOException {
    if (task == YoloTask.DEPTH) {
      return List.of();
    }
    Path sidecar = folder == null ? null : folder.resolve("classes.txt");
    if (sidecar == null || !Files.isRegularFile(sidecar)) {
      throw new ModelLoadException(
        "YOLO26 " + task.wireName() + " needs classes.txt beside the ONNX file ("
          + classes + " names, one per line).");
    }
    List<String> names = new ArrayList<>();
    for (String line : Files.readString(sidecar, UTF_8).split("\\R")) {
      if (!line.isBlank()) {
        names.add(line.strip());
      }
    }
    if (names.isEmpty()) {
      throw new ModelLoadException("classes.txt is empty: " + sidecar);
    }
    return List.copyOf(names);
  }

  private static int classCount(final YoloTensors tensors, final YoloTask task) {
    String classWeight = switch (task) {
      case SEMANTIC -> "model.17.classifier.1.weight";
      case CLASSIFY -> "model.10.linear.weight";
      case DEPTH -> null;
      default -> "model.23.one2one_cv3.0.2.weight";
    };
    if (classWeight == null || !tensors.has(classWeight)) {
      return task.classes();
    }
    return tensors.shape(classWeight)[0];
  }

  public static Path resolveOnnx(final Path path) throws IOException {
    return resolveOnnx(path, true);
  }

  public static Path resolveOnnx(final Path path, final boolean requireYoloName)
      throws IOException {
    Path file = requireNonNull(path, "path").toAbsolutePath().normalize();
    if (Files.isRegularFile(file)) {
      return file;
    }
    if (!Files.isDirectory(file)) {
      return null;
    }
    try (Stream<Path> stream = Files.list(file)) {
      List<Path> onnx = stream
        .filter(Files::isRegularFile)
          .filter(candidate -> isOnnx(candidate, requireYoloName))
        .sorted()
        .toList();
      if (onnx.size() == 1) {
        return onnx.getFirst();
      }
      if (!requireYoloName && onnx.size() > 1) {
        throw new ModelLoadException(
            "folder has " + onnx.size() + " ONNX files; point open() at one file: " + file);
      }
    }
    return null;
  }

  private static boolean isOnnx(final Path candidate, final boolean requireYoloName) {
    Path fileName = candidate.getFileName();
    if (fileName == null) {
      return false;
    }
    String name = fileName.toString().toLowerCase(Locale.ROOT);
    return name.endsWith(".onnx") && (!requireYoloName || name.contains("yolo26"));
  }
}
