package com.igormaznitsa.nanollvm.models.internal.yolo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.exceptions.ModelLoadException;
import com.igormaznitsa.nanollvm.exceptions.UnsupportedModelException;
import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.Box;
import com.igormaznitsa.nanollvm.models.LlmInImage;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutImage;
import com.igormaznitsa.nanollvm.models.LlmOutLabels;
import com.igormaznitsa.nanollvm.models.LlmOutObjects;
import com.igormaznitsa.nanollvm.models.LlmOutput;
import com.igormaznitsa.nanollvm.models.Point2D;
import com.igormaznitsa.nanollvm.models.VisionHit;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

final class Yolo26Test {

  static boolean detectOnnxPresent() {
    return Files.isRegularFile(detectOnnx());
  }

  private static void fill(
    final boolean[] mask,
    final int width,
    final int x0,
    final int y0,
    final int x1,
    final int y1
  ) {
    for (int y = y0; y < y1; y++) {
      for (int x = x0; x < x1; x++) {
        mask[y * width + x] = true;
      }
    }
  }

  private static int chebyshev(final Point2D a, final Point2D b) {
    return (int) Math.max(Math.abs(a.x() - b.x()), Math.abs(a.y() - b.y()));
  }

  private static boolean onBorder(final boolean[] mask, final int width, final Point2D point) {
    int x = Math.round(point.x());
    int y = Math.round(point.y());
    int[] dx = {1, -1, 0, 0};
    int[] dy = {0, 0, 1, -1};
    for (int k = 0; k < 4; k++) {
      int nx = x + dx[k];
      int ny = y + dy[k];
      if (nx < 0 || ny < 0 || nx >= width || ny >= mask.length / width || !mask[ny * width + nx]) {
        return true;
      }
    }
    return false;
  }

  private static Path detectOnnx() {
    Path local = Path.of("models/YOLO26n/detect/yolo26n.onnx");
    if (Files.isRegularFile(local)) {
      return local;
    }
    return Path.of("..").resolve(local);
  }

  @Test
  void fileNameSelectsTheNanoTask() {
    assertEquals(YoloTask.DETECT, YoloTask.fromFileName("yolo26n.onnx"));
    assertEquals(YoloTask.DETECT, YoloTask.fromFileName("yolo26s.onnx"));
    assertEquals(YoloTask.SEGMENT, YoloTask.fromFileName("yolo26n-seg.onnx"));
    assertEquals(YoloTask.SEMANTIC, YoloTask.fromFileName("yolo26n-sem.onnx"));
    assertEquals(YoloTask.DEPTH, YoloTask.fromFileName("yolo26n-depth.onnx"));
    assertEquals(YoloTask.CLASSIFY, YoloTask.fromFileName("YOLO26n-cls.onnx"));
    assertEquals(YoloTask.POSE, YoloTask.fromFileName("yolo26n-pose.onnx"));
    assertEquals(YoloTask.OBB, YoloTask.fromFileName("yolo26n-obb.onnx"));
    assertNull(YoloTask.fromFileName("whisper.onnx"));
  }

  @Test
  void rejectsWideScalesPickleAndYoloe(@TempDir final Path dir) throws Exception {
    for (String name : List.of(
      "yolo26m.onnx", "yolo26l.onnx", "yolo26x.onnx", "yolo26n.pt",
      "yolo26n-p2.onnx", "yolo26n-p6.onnx", "yoloe-26n.onnx")) {
      Path file = dir.resolve(name);
      Files.writeString(file, "not-an-onnx");
      assertThrows(UnsupportedModelException.class, () -> LlmModelFactory.make(file));
    }
  }

  @Test
  void argbImageRejectsAShortBuffer() {
    assertThrows(IllegalArgumentException.class, () -> new ArgbImage(2, 2, new int[3]));
  }

  @Test
  void letterboxKeepsAspectAndMapsBack() {
    YoloLetterbox letter = YoloLetterbox.of(new ArgbImage(320, 160, new int[320 * 160]));

    assertEquals(3, letter.tensor.size(0));
    assertEquals(YoloLetterbox.SIZE, letter.tensor.size(1));
    assertEquals(YoloLetterbox.SIZE, letter.tensor.size(2));
    assertEquals(160f, letter.toSource(320f, 320f).x(), 1e-3f);
    assertEquals(80f, letter.toSource(320f, 320f).y(), 1e-3f);
  }

  @Test
  void nearestTwiceDoublesSpatialSize() {
    Tensor scaled = YoloMath.nearestTwice(Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 2, 2));

    assertEquals(4, scaled.size(1));
    assertEquals(4, scaled.size(2));
    assertEquals(1f, scaled.data()[0], 1e-6f);
    assertEquals(1f, scaled.data()[1], 1e-6f);
  }

  @Test
  void everyNanoTaskBuildsOnZeroWeights() {
    for (YoloTask task : YoloTask.values()) {
      Yolo26Model model = Yolo26Model.synthetic(task);
      assertTrue(model.architectureName().startsWith("yolo26n-"));
      assertEquals(task, model.task());
    }
  }

  @Test
  void zeroForwardReturnsTheTaskResult() {
    Tensor pixels = Tensor.zeros(3, 64, 64);
    YoloLetterbox letter = YoloLetterbox.direct(pixels, 64, 64);
    for (YoloTask task : YoloTask.values()) {
      Yolo26Model model = Yolo26Model.synthetic(task);
      LlmOutput output = model.forwardLetter(letter, null);
      switch (task) {
        case CLASSIFY -> assertInstanceOf(LlmOutLabels.class, output);
        case SEMANTIC, DEPTH -> {
          LlmOutImage image = assertInstanceOf(LlmOutImage.class, output);
          assertEquals(64, image.image().width());
          assertEquals(64, image.image().height());
        }
        default -> {
          LlmOutObjects found = assertInstanceOf(LlmOutObjects.class, output);
          assertTrue(found.objects().size() <= 300);
        }
      }
    }
  }

  @Test
  void zeroDetectForwardReturnsTopBoxes() {
    Yolo26Model model = Yolo26Model.synthetic(YoloTask.DETECT);
    LlmOutObjects found = (LlmOutObjects) model.forwardLetter(
      YoloLetterbox.direct(Tensor.zeros(3, 64, 64), 64, 64), null);

    assertEquals(300, found.objects().size());
    VisionHit first = found.objects().getFirst();
    assertEquals("person", first.className());
    assertEquals(0.5f, first.probability(), 1e-5f);
    Box box = assertInstanceOf(Box.class, first.shape());
    assertTrue(box.left() >= 0f && box.right() <= 64f);
    assertTrue(box.top() >= 0f && box.bottom() <= 64f);
    assertTrue(first.keypoints().isEmpty());
  }

  @Test
  void maskOutlineWalksTheBorderAndIgnoresASmallerBlob() {
    boolean[] mask = new boolean[16 * 16];
    fill(mask, 16, 1, 1, 4, 4);
    fill(mask, 16, 6, 5, 14, 13);

    List<Point2D> outline = YoloHead.orderedMaskOutline(mask, 16, 16);

    assertTrue(outline.size() >= 8);
    for (int i = 0; i < outline.size(); i++) {
      Point2D point = outline.get(i);
      Point2D next = outline.get((i + 1) % outline.size());
      assertTrue(point.x() >= 6f && point.x() <= 13f && point.y() >= 5f && point.y() <= 12f);
      assertTrue(chebyshev(point, next) == 1);
      assertTrue(onBorder(mask, 16, point));
    }
  }

  @Test
  void obbAndPoseUseBuiltInNames() throws Exception {
    assertEquals("small vehicle", Yolo26Loader.classNames(Path.of("."), YoloTask.OBB, 15).get(10));
    assertEquals("large vehicle", Yolo26Loader.classNames(Path.of("."), YoloTask.OBB, 15).get(9));
    assertEquals(List.of("person"), Yolo26Loader.classNames(Path.of("."), YoloTask.POSE, 1));
  }

  @Test
  void semanticWithoutNamesFails() {
    assertThrows(ModelLoadException.class,
      () -> Yolo26Loader.classNames(Path.of("."), YoloTask.SEMANTIC, 19));
  }

  @Test
  @EnabledIf("detectOnnxPresent")
  void detectOnnxRunsWhenPresent() throws Exception {
    try (LlmModel model = LlmModelFactory.make(detectOnnx())) {
      ArgbImage image = new ArgbImage(64, 64, new int[64 * 64]);
      LlmOutObjects found = model.generate(new LlmInImage(image), LlmModality.IMAGE);
      assertTrue(found.objects().size() <= 300);
    }
  }
}
