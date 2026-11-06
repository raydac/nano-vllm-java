package com.igormaznitsa.nanollvm.samples;

import static java.nio.file.Files.createTempDirectory;
import static java.nio.file.Files.writeString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.Box;
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
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

class YoloStreetHelloWorldTest {

  private static Selection select(final String argument, final String lines) throws IOException {
    String[] args = argument.isEmpty() ? new String[0] : new String[] {argument};
    return YoloStreetHelloWorld.select(
      args,
      new BufferedReader(new StringReader(lines)),
      new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
  }

  private static VisionHit hit(final String className, final float probability, final int top) {
    return new VisionHit(className, probability, new Box(16, top, 90, top + 40));
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
      hit("traffic light", 0.50f, 50)));

    assertEquals(
      List.of("person", "Bicycle", "car"),
      kept.stream().map(VisionHit::className).toList());
    assertEquals(
      List.of("small vehicle"),
      YoloStreetHelloWorld.streetObjects(List.of(hit("small vehicle", 0.8f, 10)))
        .stream().map(VisionHit::className).toList());
  }

  @Test
  void obbUsesTheParkingLotAndDetectUsesTheStreet() throws IOException {
    BufferedImage parking = YoloStreetHelloWorld.photoFor(StreetHead.OBB);
    BufferedImage street = YoloStreetHelloWorld.photoFor(StreetHead.DETECT);

    assertEquals(1171, parking.getWidth());
    assertEquals(780, parking.getHeight());
    assertEquals(800, street.getWidth());
    assertEquals(544, street.getHeight());
  }

  @Test
  void secondsUseThreeDecimals() {
    assertEquals("1.250 s", YoloStreetHelloWorld.seconds(1_250_000_000L));
  }

  @Test
  void labelPrintsTheScoreBesideTheName() {
    assertEquals("person 0.91", YoloStreetHelloWorld.label(hit("person", 0.906f, 0)));
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
    assertEquals(Path.of("yolo-street-seg.png"),
      YoloStreetHelloWorld.outputFile(StreetHead.SEGMENT));
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
  void semanticDepthAndUnknownHeadsAreRefused() {
    assertThrows(IllegalArgumentException.class, () -> YoloStreetHelloWorld.parseHead("sem"));
    assertThrows(IllegalArgumentException.class, () -> YoloStreetHelloWorld.parseHead("depth"));
    assertThrows(IllegalArgumentException.class, () -> YoloStreetHelloWorld.parseHead("cls"));
    assertThrows(IllegalArgumentException.class, () -> YoloStreetHelloWorld.parseHead("house"));
    assertThrows(IllegalArgumentException.class,
      () -> YoloStreetHelloWorld.parseHead("yolo26m.onnx"));
  }

  @Test
  void markPaintsEachStreetClassAndThePngKeepsTheFrame() throws IOException {
    BufferedImage source = new BufferedImage(220, 140, BufferedImage.TYPE_INT_RGB);
    List<VisionHit> hits = List.of(
      hit("person", 0.91f, 20),
      new VisionHit("bicycle", 0.80f, new Polygon(List.of(
        new Point2D(120, 40), new Point2D(180, 50), new Point2D(150, 110)))),
      hit("car", 0.77f, 70));

    BufferedImage marked = YoloStreetHelloWorld.mark(source, hits);
    assertTrue(contains(marked, new Color(220, 40, 40)));
    assertTrue(contains(marked, new Color(30, 140, 220)));
    assertTrue(contains(marked, new Color(30, 160, 60)));

    BufferedImage crowded = YoloStreetHelloWorld.mark(source, List.of(
      hit("person", 0.90f, 30),
      hit("person", 0.40f, 30)));
    int oneCaption = count(marked, new Color(220, 40, 40));
    assertTrue(count(crowded, new Color(220, 40, 40)) > oneCaption);

    Path png = createTempDirectory("yolo-street-png").resolve("marked.png");
    BufferedImage reread =
      ImageIO.read(YoloStreetHelloWorld.writeMarked(source, hits, png).toFile());
    assertEquals(220, reread.getWidth());
    assertEquals(140, reread.getHeight());
  }
}
