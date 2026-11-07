package com.igormaznitsa.nanollvm.samples;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.Box;
import com.igormaznitsa.nanollvm.models.LlmInImage;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutObjects;
import com.igormaznitsa.nanollvm.models.Point2D;
import com.igormaznitsa.nanollvm.models.Polygon;
import com.igormaznitsa.nanollvm.models.Shape2D;
import com.igormaznitsa.nanollvm.models.VisionHit;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import javax.imageio.ImageIO;

/**
 * Marks people, bicycles, and cars on a street photo and writes a PNG with each score.
 *
 * <p>Detect, segment, and pose use {@code images/yolo-street.jpg}. Oriented boxes use
 * {@code images/yolo-obb.jpg}, an overhead parking lot. With no argument the sample asks for a
 * head: detect, seg, pose, or obb. Enter selects detect. A head name or a YOLO26 ONNX path skips
 * the menu. Weights are {@code models/YOLO26n} ({@code ./models/download-yolo26n.sh}). The marked
 * picture is {@code yolo-street-<head>.png} in the working directory. Recognition runs on every
 * processor. The log gives the wall-clock time to load the weights and to recognize the picture.
 * Run from the repository root.
 *
 * @since 1.5.0
 */
public final class YoloStreetHelloWorld {

  static final String STREET_PHOTO = "/images/yolo-street.jpg";
  static final String OBB_PHOTO = "/images/yolo-obb.jpg";

  private static final Set<String> STREET_CLASSES = Set.of(
    "person", "bicycle", "car", "small vehicle", "large vehicle");

  private YoloStreetHelloWorld() {
  }

  public static void main(final String[] args) throws Exception {
    Selection selection = select(
      args,
      new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)),
      System.out);
    requireWeights(selection.model());
    BufferedImage photo = photoFor(selection.head());

    System.out.println("Loading " + selection.model().toAbsolutePath());
    long started = System.nanoTime();
    try (LlmModel model = LlmModelFactory.make(selection.model());
         LLM llm = LLM.builder(model).allCpuThreads().build()) {
      long loaded = System.nanoTime();
      System.out.println("loaded in " + seconds(loaded - started));
      LlmOutObjects found = llm.generate(
        new LlmInImage(fromBufferedImage(photo)), LlmModality.IMAGE);
      long recognized = System.nanoTime();
      List<VisionHit> street = streetObjects(found.objects());
      Path written = writeMarked(photo, street, outputFile(selection.head()));
      System.out.println("recognized in " + seconds(recognized - loaded));
      System.out.println(street.size() + " objects, wrote " + written);
      for (VisionHit hit : street) {
        System.out.println("  " + label(hit));
      }
    }
  }

  static Selection select(
    final String[] args,
    final BufferedReader in,
    final PrintStream out
  ) throws IOException {
    requireNonNull(in, "in");
    requireNonNull(out, "out");
    if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
      Path file = Path.of(args[0]);
      if (Files.isRegularFile(file)) {
        return selectionFromFile(file);
      }
      StreetHead head = parseHead(args[0]);
      return new Selection(head, head.model());
    }

    printMenu(out);
    String line = in.readLine();
    StreetHead head = line == null || line.isBlank() ? StreetHead.DETECT : parseHead(line);
    return new Selection(head, head.model());
  }

  static StreetHead parseHead(final String token) {
    String text = requireNonNull(token, "token").strip().toLowerCase(ROOT);
    if (text.isEmpty()) {
      throw unknown(token);
    }
    StreetHead numbered = numbered(text);
    if (numbered != null) {
      return numbered;
    }
    StreetHead alias = aliased(text);
    if (alias != null) {
      return alias;
    }
    if (!text.contains("yolo26")) {
      throw unknown(text);
    }
    return headOfFileName(text);
  }

  static Path outputFile(final StreetHead head) {
    return Path.of("yolo-street-" + requireNonNull(head, "head").wireName() + ".png");
  }

  /**
   * Packed {@code 0xAARRGGBB} raster for {@link LlmInImage}.
   *
   * @param image source picture; must be non-empty
   * @return a copy of the pixels in row-major order
   */
  public static ArgbImage fromBufferedImage(final BufferedImage image) {
    requireNonNull(image, "image");
    int width = image.getWidth();
    int height = image.getHeight();
    return new ArgbImage(width, height, image.getRGB(0, 0, width, height, null, 0, width));
  }

  /**
   * Picture for drawing or writing a PNG from an {@link ArgbImage}.
   *
   * @param image packed raster
   * @return an {@code ARGB} buffered image with those pixels
   */
  public static BufferedImage toBufferedImage(final ArgbImage image) {
    requireNonNull(image, "image");
    BufferedImage buffered = new BufferedImage(
      image.width(), image.height(), BufferedImage.TYPE_INT_ARGB);
    buffered.setRGB(0, 0, image.width(), image.height(), image.pixels(), 0, image.width());
    return buffered;
  }

  static BufferedImage streetPhoto() throws IOException {
    return readPhoto(STREET_PHOTO);
  }

  static BufferedImage photoFor(final StreetHead head) throws IOException {
    return readPhoto(requireNonNull(head, "head") == StreetHead.OBB ? OBB_PHOTO : STREET_PHOTO);
  }

  static String seconds(final long nanos) {
    return String.format(ROOT, "%.3f s", nanos / 1e9);
  }

  private static BufferedImage readPhoto(final String resource) throws IOException {
    try (InputStream in = YoloStreetHelloWorld.class.getResourceAsStream(resource)) {
      if (in == null) {
        throw new IllegalStateException("missing photo " + resource);
      }
      BufferedImage photo = ImageIO.read(in);
      if (photo == null) {
        throw new IllegalStateException("could not read photo " + resource);
      }
      return photo;
    }
  }

  static List<VisionHit> streetObjects(final List<VisionHit> hits) {
    requireNonNull(hits, "hits");
    return hits.stream()
      .filter(hit -> STREET_CLASSES.contains(hit.className().toLowerCase(ROOT)))
      .toList();
  }

  static String label(final VisionHit hit) {
    requireNonNull(hit, "hit");
    return "%s %s".formatted(hit.className(), String.format(ROOT, "%.2f", hit.probability()));
  }

  static BufferedImage mark(final BufferedImage source, final List<VisionHit> hits) {
    requireNonNull(source, "source");
    requireNonNull(hits, "hits");
    BufferedImage canvas = new BufferedImage(
      source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_RGB);
    Graphics2D pen = canvas.createGraphics();
    pen.drawImage(source, 0, 0, null);
    pen.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    pen.setRenderingHint(
      RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
    pen.setStroke(new BasicStroke(2f));
    pen.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
    List<VisionHit> ranked = hits.stream()
      .sorted(Comparator.comparingDouble(VisionHit::probability).reversed())
      .toList();
    for (VisionHit hit : ranked) {
      paintShape(pen, hit);
    }
    List<Rectangle> occupied = new ArrayList<>();
    for (VisionHit hit : ranked) {
      paintLabel(pen, hit, canvas.getWidth(), canvas.getHeight(), occupied);
    }
    pen.dispose();
    return canvas;
  }

  static Path writeMarked(
    final BufferedImage source,
    final List<VisionHit> hits,
    final Path out
  ) throws IOException {
    requireNonNull(out, "out");
    Path parent = out.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    if (!ImageIO.write(mark(source, hits), "png", out.toFile())) {
      throw new IllegalStateException("PNG writer missing for " + out);
    }
    return out.toAbsolutePath();
  }

  private static void printMenu(final PrintStream out) {
    out.println("Select a YOLO26 head. People, bicycles, and cars are marked.");
    out.println("Oriented boxes use the parking-lot photo. The other heads use the street.");
    StreetHead[] heads = StreetHead.values();
    for (int i = 0; i < heads.length; i++) {
      out.printf(ROOT, "  %d. %s — %s%n", i + 1, heads[i].wireName(), heads[i].summary());
    }
    out.println("Enter selects detect.");
  }

  private static Selection selectionFromFile(final Path file) {
    Path name = file.getFileName();
    if (name == null) {
      throw unknown(file.toString());
    }
    StreetHead head = headOfFileName(name.toString());
    return new Selection(head, file);
  }

  private static StreetHead numbered(final String text) {
    if (text.length() != 1 || !Character.isDigit(text.charAt(0))) {
      return null;
    }
    int index = text.charAt(0) - '1';
    StreetHead[] heads = StreetHead.values();
    if (index < 0 || index >= heads.length) {
      throw unknown(text);
    }
    return heads[index];
  }

  private static StreetHead aliased(final String text) {
    return switch (text) {
      case "detect", "detection" -> StreetHead.DETECT;
      case "seg", "segment", "segmentation" -> StreetHead.SEGMENT;
      case "pose" -> StreetHead.POSE;
      case "obb", "oriented" -> StreetHead.OBB;
      default -> null;
    };
  }

  private static StreetHead headOfFileName(final String fileName) {
    String name = fileName.toLowerCase(ROOT);
    if (!name.contains("yolo26")) {
      throw unknown(fileName);
    }
    if (name.contains("-sem") || name.contains("-depth") || name.contains("-cls")) {
      throw new IllegalArgumentException(
        "this sample marks people, bicycles, and cars. Choose detect, seg, pose, or obb.");
    }
    if (name.contains("-seg")) {
      return StreetHead.SEGMENT;
    }
    if (name.contains("-pose")) {
      return StreetHead.POSE;
    }
    if (name.contains("-obb")) {
      return StreetHead.OBB;
    }
    if (name.contains("yolo26n") || name.contains("yolo26s") || name.contains("yolo26m")
      || name.contains("yolo26l") || name.contains("yolo26x")) {
      return StreetHead.DETECT;
    }
    throw unknown(fileName);
  }

  private static void requireWeights(final Path model) {
    if (!Files.isRegularFile(model)) {
      throw new IllegalStateException(
        "YOLO26 ONNX not found: " + model.toAbsolutePath()
          + ". Run ./models/download-yolo26n.sh");
    }
  }

  private static void paintShape(final Graphics2D pen, final VisionHit hit) {
    pen.setColor(ink(hit.className()));
    drawShape(pen, hit.shape());
    for (Point2D point : hit.keypoints()) {
      pen.fillOval(Math.round(point.x()) - 3, Math.round(point.y()) - 3, 6, 6);
    }
  }

  private static void drawShape(final Graphics2D pen, final Shape2D shape) {
    switch (shape) {
      case Box box -> pen.drawRect(
        Math.round(box.left()),
        Math.round(box.top()),
        Math.max(1, Math.round(box.right() - box.left())),
        Math.max(1, Math.round(box.bottom() - box.top())));
      case Polygon polygon -> {
        int[] xs = polygon.points().stream().mapToInt(point -> Math.round(point.x())).toArray();
        int[] ys = polygon.points().stream().mapToInt(point -> Math.round(point.y())).toArray();
        pen.drawPolygon(xs, ys, xs.length);
      }
    }
  }

  private static void paintLabel(
    final Graphics2D pen,
    final VisionHit hit,
    final int width,
    final int height,
    final List<Rectangle> occupied
  ) {
    String text = label(hit);
    FontMetrics metrics = pen.getFontMetrics();
    int textWidth = metrics.stringWidth(text);
    int textHeight = metrics.getHeight();
    Box box = bounds(hit.shape());
    int x = Math.clamp(Math.round(box.left()), 0, Math.max(0, width - textWidth - 4));
    int y = Math.round(box.top()) - textHeight;
    if (y < 0) {
      y = Math.clamp(Math.round(box.top()), 0, Math.max(0, height - textHeight));
    }
    Rectangle spot = freeSpot(x, y, textWidth + 4, textHeight, height, occupied);
    occupied.add(spot);

    pen.setColor(ink(hit.className()));
    pen.fillRect(spot.x, spot.y, spot.width, spot.height);
    pen.setColor(Color.WHITE);
    pen.drawString(text, spot.x + 2, spot.y + metrics.getAscent());
  }

  private static Rectangle freeSpot(
    final int x,
    final int y,
    final int width,
    final int height,
    final int imageHeight,
    final List<Rectangle> occupied
  ) {
    int top = y;
    for (int step = 0; step < 16; step++) {
      Rectangle spot = new Rectangle(x, top, width, height);
      Rectangle gap = new Rectangle(x - 3, top - 3, width + 6, height + 6);
      if (occupied.stream().noneMatch(gap::intersects)) {
        return spot;
      }
      int next = top + height + 4;
      if (next + height > imageHeight) {
        return spot;
      }
      top = next;
    }
    return new Rectangle(x, top, width, height);
  }

  private static Box bounds(final Shape2D shape) {
    if (shape instanceof Box box) {
      return box;
    }
    float left = Float.POSITIVE_INFINITY;
    float top = Float.POSITIVE_INFINITY;
    float right = Float.NEGATIVE_INFINITY;
    float bottom = Float.NEGATIVE_INFINITY;
    for (Point2D point : ((Polygon) shape).points()) {
      left = Math.min(left, point.x());
      top = Math.min(top, point.y());
      right = Math.max(right, point.x());
      bottom = Math.max(bottom, point.y());
    }
    return new Box(left, top, right, bottom);
  }

  private static Color ink(final String className) {
    return switch (className.toLowerCase(ROOT)) {
      case "person" -> new Color(220, 40, 40);
      case "bicycle" -> new Color(30, 140, 220);
      case "car" -> new Color(30, 160, 60);
      default -> new Color(240, 180, 20);
    };
  }

  private static IllegalArgumentException unknown(final String text) {
    return new IllegalArgumentException(
      "unknown YOLO26 head: " + text + ". Choose detect, seg, pose, or obb.");
  }

  enum StreetHead {
    DETECT("detect", "models/YOLO26n/detect/yolo26n.onnx", "a box on each object"),
    SEGMENT("seg", "models/YOLO26n/seg/yolo26n-seg.onnx", "the outline of each object"),
    POSE("pose", "models/YOLO26n/pose/yolo26n-pose.onnx", "a box and body points"),
    OBB("obb", "models/YOLO26n/obb/yolo26n-obb.onnx", "rotated corners");

    private final String wireName;
    private final Path model;
    private final String summary;

    StreetHead(final String wireName, final String model, final String summary) {
      this.wireName = wireName;
      this.model = Path.of(model);
      this.summary = summary;
    }

    String wireName() {
      return this.wireName;
    }

    Path model() {
      return this.model;
    }

    String summary() {
      return this.summary;
    }
  }

  record Selection(StreetHead head, Path model) {
    Selection {
      requireNonNull(head, "head");
      requireNonNull(model, "model");
    }
  }
}
