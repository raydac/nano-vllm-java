package com.igormaznitsa.nanollvm.samples;

import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.Box;
import com.igormaznitsa.nanollvm.models.Keypoint;
import com.igormaznitsa.nanollvm.models.LlmLabelScore;
import com.igormaznitsa.nanollvm.models.Point2D;
import com.igormaznitsa.nanollvm.models.Polygon;
import com.igormaznitsa.nanollvm.models.VisionHit;
import com.igormaznitsa.nanollvm.samples.YoloStreetHelloWorld.Selection;
import com.igormaznitsa.nanollvm.samples.YoloStreetHelloWorld.StreetHead;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

class YoloStreetHelloWorldTest {

  private static final List<String> NAMES = List.of(
    "person", "bicycle", "car", "truck", "traffic light", "small vehicle", "Bicycle");

  private static Selection select(final String argument, final String lines) throws IOException {
    String[] args = argument.isEmpty() ? new String[0] : new String[] {argument};
    return YoloStreetHelloWorld.select(
      args,
      new BufferedReader(new StringReader(lines)),
      new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
  }

  private static VisionHit hit(final String className, final float probability, final int top) {
    int classId = NAMES.indexOf(className);
    if (classId < 0) {
      throw new IllegalArgumentException(className);
    }
    return new VisionHit(classId, probability, new Box(16, top, 90, top + 40));
  }

  private static int classId(final String className) {
    return NAMES.indexOf(className);
  }

  private static boolean contains(final BufferedImage image, final Color color) {
    return count(image, color) > 0;
  }

  private static int count(final BufferedImage image, final Color color) {
    int rgb = color.getRGB();
    int found = 0;
    for (int y = 0; y < image.getHeight(); y++) {
      for (int x = 0; x < image.getWidth(); x++) {
        if (image.getRGB(x, y) == rgb) {
          found++;
        }
      }
    }
    return found;
  }

  @Test
  void bufferedImageRoundTripKeepsArgb() {
    BufferedImage source = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
    source.setRGB(0, 0, 0xff112233);
    source.setRGB(1, 0, 0xff445566);

    ArgbImage raster = YoloStreetHelloWorld.fromBufferedImage(source);
    BufferedImage back = YoloStreetHelloWorld.toBufferedImage(raster);

    assertEquals(2, raster.width());
    assertEquals(1, raster.height());
    assertEquals(0xff112233, back.getRGB(0, 0));
    assertEquals(0xff445566, back.getRGB(1, 0));
  }

  @Test
  void streetPhotoIsTheBundledCrossing() throws IOException {
    BufferedImage photo = YoloStreetHelloWorld.streetPhoto();

    assertEquals(800, photo.getWidth());
    assertEquals(544, photo.getHeight());
  }

  @Test
  void streetObjectsKeepPeopleBicyclesAndCars() {
    List<VisionHit> kept = YoloStreetHelloWorld.streetObjects(List.of(
      hit("person", 0.91f, 10),
      hit("truck", 0.80f, 20),
      hit("Bicycle", 0.70f, 30),
      hit("car", 0.66f, 40),
      hit("traffic light", 0.50f, 50)), NAMES);

    assertEquals(
      List.of("person", "Bicycle", "car"),
      kept.stream().map(found -> YoloStreetHelloWorld.className(NAMES, found.classId())).toList());
    assertEquals(
      List.of("small vehicle"),
      YoloStreetHelloWorld.streetObjects(List.of(hit("small vehicle", 0.8f, 10)), NAMES)
        .stream().map(found -> YoloStreetHelloWorld.className(NAMES, found.classId())).toList());
  }

  private static int blend(final int channel) {
    return (channel * 128 + 255 * 127 + 127) / 255;
  }

  private static int sceneOnWhite(final int red, final int green, final int blue) {
    return 0xff000000
      | (blend(red) << 16)
      | (blend(green) << 8)
      | blend(blue);
  }

  @Test
  void depthTintPaintsNearRedAndFarGreenOverThePhoto() {
    BufferedImage photo = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
    photo.setRGB(0, 0, 0xffffff);
    photo.setRGB(1, 0, 0xffffff);
    ArgbImage depth = new ArgbImage(2, 1, new int[] {0xff000000, 0xffffffff});

    BufferedImage tinted = YoloStreetHelloWorld.tintDepth(photo, depth);

    assertEquals(0xffFF7F7F, tinted.getRGB(0, 0));
    assertEquals(0xff7FFF7F, tinted.getRGB(1, 0));
  }

  @Test
  void obbUsesTheParkingLotAndDetectUsesTheStreet() throws IOException {
    BufferedImage parking = YoloStreetHelloWorld.photoFor(StreetHead.OBB);
    BufferedImage street = YoloStreetHelloWorld.photoFor(StreetHead.DETECT);

    assertEquals(1171, parking.getWidth());
    assertEquals(780, parking.getHeight());
    assertEquals(800, street.getWidth());
    assertEquals(544, street.getHeight());
    assertEquals(800, YoloStreetHelloWorld.photoFor(StreetHead.DEPTH).getWidth());
    assertEquals(544, YoloStreetHelloWorld.photoFor(StreetHead.DEPTH).getHeight());
    assertEquals(800, YoloStreetHelloWorld.photoFor(StreetHead.SEMANTIC).getWidth());
    assertEquals(544, YoloStreetHelloWorld.photoFor(StreetHead.SEMANTIC).getHeight());
    assertEquals(800, YoloStreetHelloWorld.photoFor(StreetHead.CLASSIFY).getWidth());
    assertEquals(544, YoloStreetHelloWorld.photoFor(StreetHead.CLASSIFY).getHeight());
  }

  @Test
  void sceneTintPaintsRoadAndSkyOverThePhoto() {
    BufferedImage photo = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
    photo.setRGB(0, 0, 0xffffff);
    photo.setRGB(1, 0, 0xffffff);
    ArgbImage classes = new ArgbImage(2, 1, new int[] {0xff000000, 0xff0a0a0a});

    BufferedImage tinted = YoloStreetHelloWorld.tintScene(photo, classes);

    assertEquals(sceneOnWhite(128, 64, 128), tinted.getRGB(0, 0));
    assertEquals(sceneOnWhite(70, 130, 180), tinted.getRGB(1, 0));
    assertEquals("road, sky", YoloStreetHelloWorld.sceneClasses(classes));
  }

  @Test
  void markLabelsWritesTheTopNameOnThePhoto() {
    BufferedImage photo = new BufferedImage(220, 80, BufferedImage.TYPE_INT_RGB);
    int[] white = new int[220 * 80];
    Arrays.fill(white, 0xffffff);
    photo.setRGB(0, 0, 220, 80, white, 0, 220);

    BufferedImage marked = YoloStreetHelloWorld.markLabels(
      photo, List.of(new LlmLabelScore("street sign", 0.906f)));

    assertEquals("street sign 0.91",
      YoloStreetHelloWorld.label(new LlmLabelScore("street sign", 0.906f)));
    assertEquals(new Color(20, 20, 20).getRGB(), marked.getRGB(10, 10));
  }

  @Test
  void secondsUseThreeDecimals() {
    assertEquals("1.250 s", YoloStreetHelloWorld.seconds(1_250_000_000L));
  }

  @Test
  void labelPrintsTheScoreBesideTheName() {
    assertEquals("person 0.91", YoloStreetHelloWorld.label(hit("person", 0.906f, 0), NAMES));
  }

  @Test
  void blankLineSelectsDetect() throws IOException {
    Selection selection = select("", "\n");

    assertEquals(StreetHead.DETECT, selection.head());
    assertEquals(Path.of("models/YOLO26n/detect/yolo26n.onnx"), selection.model());
  }

  @Test
  void menuNumberAndAliasSelectTheHead() throws IOException {
    assertEquals(StreetHead.SEGMENT, select("", "2\n").head());
    assertEquals(StreetHead.POSE, YoloStreetHelloWorld.parseHead("pose"));
    assertEquals(
      Path.of("models/YOLO26n/pose/yolo26n-pose.onnx"),
      StreetHead.POSE.model());
    assertEquals(StreetHead.OBB, YoloStreetHelloWorld.parseHead("oriented"));
    assertEquals(StreetHead.SEMANTIC, select("", "3\n").head());
    assertEquals(StreetHead.SEMANTIC, YoloStreetHelloWorld.parseHead("semantic"));
    assertEquals(
      Path.of("models/YOLO26n/sem/yolo26n-sem.onnx"),
      StreetHead.SEMANTIC.model());
    assertEquals(StreetHead.CLASSIFY, select("", "7\n").head());
    assertEquals(StreetHead.CLASSIFY, YoloStreetHelloWorld.parseHead("classification"));
    assertEquals(
      Path.of("models/YOLO26n/cls/yolo26n-cls.onnx"),
      StreetHead.CLASSIFY.model());
    assertEquals(StreetHead.DEPTH, select("", "6\n").head());
    assertEquals(StreetHead.DEPTH, YoloStreetHelloWorld.parseHead("depth"));
    assertEquals(
      Path.of("models/YOLO26n/depth/yolo26n-depth.onnx"),
      StreetHead.DEPTH.model());
    assertEquals(Path.of("yolo-street-seg.png"),
      YoloStreetHelloWorld.outputFile(StreetHead.SEGMENT));
    assertEquals(Path.of("yolo-street-depth.png"),
      YoloStreetHelloWorld.outputFile(StreetHead.DEPTH));
    assertEquals(Path.of("yolo-street-sem.png"),
      YoloStreetHelloWorld.outputFile(StreetHead.SEMANTIC));
    assertEquals(Path.of("yolo-street-cls.png"),
      YoloStreetHelloWorld.outputFile(StreetHead.CLASSIFY));
  }

  @Test
  void onnxPathSelectsItsHead() throws IOException {
    Path file = createTempDirectory("yolo-street").resolve("yolo26n-pose.onnx");
    writeString(file, "not-weights");

    Selection selection = select(file.toString(), "");

    assertEquals(StreetHead.POSE, selection.head());
    assertEquals(file.toAbsolutePath().normalize(), selection.model().toAbsolutePath().normalize());
  }

  @Test
  void unknownHeadsAreRefused() {
    assertEquals(StreetHead.SEMANTIC, YoloStreetHelloWorld.parseHead("sem"));
    assertEquals(StreetHead.SEMANTIC, YoloStreetHelloWorld.parseHead("yolo26n-sem.onnx"));
    assertEquals(StreetHead.CLASSIFY, YoloStreetHelloWorld.parseHead("cls"));
    assertEquals(StreetHead.CLASSIFY, YoloStreetHelloWorld.parseHead("yolo26n-cls.onnx"));
    assertThrows(IllegalArgumentException.class, () -> YoloStreetHelloWorld.parseHead("house"));
    assertEquals(StreetHead.DETECT, YoloStreetHelloWorld.parseHead("yolo26m.onnx"));
    assertEquals(StreetHead.POSE, YoloStreetHelloWorld.parseHead("yolo26x-pose.onnx"));
    assertEquals(StreetHead.DEPTH, YoloStreetHelloWorld.parseHead("yolo26n-depth.onnx"));
  }

  @Test
  void markPaintsEachStreetClassAndThePngKeepsTheFrame() throws IOException {
    BufferedImage source = new BufferedImage(220, 140, BufferedImage.TYPE_INT_RGB);
    List<VisionHit> hits = List.of(
      hit("person", 0.91f, 20),
      new VisionHit(classId("bicycle"), 0.80f, new Polygon(List.of(
        new Point2D(120, 40), new Point2D(180, 50), new Point2D(150, 110)))),
      hit("car", 0.77f, 70));

    BufferedImage marked = YoloStreetHelloWorld.mark(source, hits, NAMES);
    assertTrue(contains(marked, new Color(220, 40, 40)));
    assertTrue(contains(marked, new Color(30, 140, 220)));
    assertTrue(contains(marked, new Color(30, 160, 60)));

    BufferedImage crowded = YoloStreetHelloWorld.mark(source, List.of(
      hit("person", 0.90f, 30),
      hit("person", 0.40f, 30)), NAMES);
    int oneCaption = count(marked, new Color(220, 40, 40));
    assertTrue(count(crowded, new Color(220, 40, 40)) > oneCaption);

    Path png = createTempDirectory("yolo-street-png").resolve("marked.png");
    BufferedImage reread =
      ImageIO.read(YoloStreetHelloWorld.writeMarked(source, hits, NAMES, png).toFile());
    assertEquals(220, reread.getWidth());
    assertEquals(140, reread.getHeight());
  }

  @Test
  void visibleHitsKeepALargeQuietCarAndDropASpeck() {
    VisionHit strong = new VisionHit(classId("car"), 0.80f, new Box(0, 0, 30, 30));
    VisionHit quiet = new VisionHit(classId("car"), 0.012f, new Box(50, 50, 99, 99));
    VisionHit speck = new VisionHit(classId("car"), 0.02f, new Box(0, 80, 8, 88));
    VisionHit covered = new VisionHit(classId("car"), 0.011f, new Box(52, 52, 97, 97));

    assertEquals(
      List.of(strong, quiet),
      YoloStreetHelloWorld.visibleHits(
        List.of(speck, quiet, strong, hit("truck", 0.9f, 0)), NAMES, 100, 100));
    assertEquals(
      List.of(quiet),
      YoloStreetHelloWorld.visibleHits(List.of(quiet, covered), NAMES, 100, 100));
  }

  @Test
  void poseDrawsAGreenLimbBetweenVisibleJoints() {
    BufferedImage source = new BufferedImage(80, 80, BufferedImage.TYPE_INT_RGB);
    VisionHit person = new VisionHit(
      classId("person"),
      0.9f,
      new Box(8, 8, 70, 70),
      List.of(new Keypoint(5, 20, 40), new Keypoint(7, 60, 40)));

    BufferedImage marked = YoloStreetHelloWorld.mark(source, List.of(person), NAMES);

    assertEquals(new Color(0, 200, 70).getRGB(), marked.getRGB(40, 40));
    assertEquals(new Color(220, 40, 40).getRGB(), marked.getRGB(20, 40));
  }

  @Test
  void segmentationFillsTheOutlineAtHalfStrength() {
    int[] white = new int[120 * 100];
    Arrays.fill(white, 0xffffff);
    BufferedImage source = new BufferedImage(120, 100, BufferedImage.TYPE_INT_RGB);
    source.setRGB(0, 0, 120, 100, white, 0, 120);
    VisionHit person = new VisionHit(classId("person"), 0.9f, new Polygon(List.of(
      new Point2D(30, 20), new Point2D(90, 20), new Point2D(60, 80))));

    BufferedImage filled = YoloStreetHelloWorld.mark(
      source, List.of(person), NAMES, StreetHead.SEGMENT);
    BufferedImage outlined = YoloStreetHelloWorld.mark(
      source, List.of(person), NAMES, StreetHead.OBB);

    assertEquals(new Color(blend(220), blend(40), blend(40)).getRGB(), filled.getRGB(60, 40));
    assertEquals(0xffffffff, outlined.getRGB(60, 40));
  }
}
