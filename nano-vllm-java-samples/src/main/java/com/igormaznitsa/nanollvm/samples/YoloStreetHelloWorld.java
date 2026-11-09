package com.igormaznitsa.nanollvm.samples;

import static java.util.Locale.ROOT;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.igormaznitsa.nanollvm.llm.LLM;
import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.Box;
import com.igormaznitsa.nanollvm.models.Keypoint;
import com.igormaznitsa.nanollvm.models.LlmInImage;
import com.igormaznitsa.nanollvm.models.LlmLabelScore;
import com.igormaznitsa.nanollvm.models.LlmModality;
import com.igormaznitsa.nanollvm.models.LlmModel;
import com.igormaznitsa.nanollvm.models.LlmModelFactory;
import com.igormaznitsa.nanollvm.models.LlmOutImage;
import com.igormaznitsa.nanollvm.models.LlmOutLabels;
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
import java.awt.Stroke;
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
import java.util.stream.IntStream;
import javax.imageio.ImageIO;

/**
 * Marks people, bicycles, and cars on a street photo and writes a PNG with each score.
 * Pose draws the body in green between those points. Segmentation fills each outline
 * at half strength. A box covering at least two percent of the picture is kept down to a
 * score of {@code 0.01}, unless a stronger box of the same class already covers it.
 *
 * <p>Detect, segment, semantic, pose, depth, and classification use {@code images/yolo-street.jpg}.
 * Oriented boxes use {@code images/yolo-obb.jpg}, an overhead parking lot. Semantic paints each
 * Cityscapes surface in its own color, half transparent. Depth paints the street from red (near)
 * to green (far), half transparent, and prints the nearest and farthest meters. Classification
 * writes the top five ImageNet labels on the photo. With no argument the sample asks for a head:
 * detect, seg, sem, pose, obb, depth, or cls. Enter selects detect. A head name or a YOLO26 ONNX
 * path skips the menu. Weights are {@code models/YOLO26n} ({@code ./models/download-yolo26n.sh}).
 * The marked picture is {@code yolo-street-<head>.png} in the working directory. Recognition runs
 * on every processor. The log gives the wall-clock time to load the weights and to recognize the
 * picture. Run from the repository root.
 *
 * @since 1.5.0
 */
public final class YoloStreetHelloWorld {

  static final String STREET_PHOTO = "/images/yolo-street.jpg";
  static final String OBB_PHOTO = "/images/yolo-obb.jpg";

  private static final Set<String> STREET_CLASSES = Set.of(
    "person", "bicycle", "car", "small vehicle", "large vehicle");

  static final float QUIET_SCORE = 0.01f;
  private static final float QUIET_AREA_FRACTION = 0.02f;

  private static final Color LIMB = new Color(0, 200, 70);
  private static final int MASK_ALPHA = 128;
  private static final int DEPTH_ALPHA = 128;

  private static final List<SceneClass> SCENE = List.of(
    new SceneClass("road", 0x804080),
    new SceneClass("sidewalk", 0xf423e8),
    new SceneClass("building", 0x464646),
    new SceneClass("wall", 0x66669c),
    new SceneClass("fence", 0xbe9999),
    new SceneClass("pole", 0x999999),
    new SceneClass("traffic light", 0xfaaa1e),
    new SceneClass("traffic sign", 0xdcdc00),
    new SceneClass("vegetation", 0x6b8e23),
    new SceneClass("terrain", 0x98fb98),
    new SceneClass("sky", 0x4682b4),
    new SceneClass("person", 0xdc143c),
    new SceneClass("rider", 0xff0000),
    new SceneClass("car", 0x00008e),
    new SceneClass("truck", 0x000046),
    new SceneClass("bus", 0x003c64),
    new SceneClass("train", 0x005064),
    new SceneClass("motorcycle", 0x0000e6),
    new SceneClass("bicycle", 0x770b20));

  private static final List<Limb> LIMBS = List.of(
    new Limb(15, 13), new Limb(13, 11), new Limb(16, 14), new Limb(14, 12),
    new Limb(11, 12), new Limb(5, 11), new Limb(6, 12), new Limb(5, 6),
    new Limb(5, 7), new Limb(6, 8), new Limb(7, 9), new Limb(8, 10),
    new Limb(1, 2), new Limb(0, 1), new Limb(0, 2), new Limb(1, 3),
    new Limb(2, 4), new Limb(3, 5), new Limb(4, 6));

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
      switch (selection.head()) {
        case DEPTH -> writeDepth(llm, photo, loaded);
        case SEMANTIC -> writeScene(llm, photo, loaded);
        case CLASSIFY -> writeLabels(llm, photo, loaded);
        default -> writeObjects(llm, photo, selection.head(), loaded);
      }
    }
  }

  private static void writeObjects(
    final LLM llm,
    final BufferedImage photo,
    final StreetHead head,
    final long loaded
  ) throws IOException {
    LlmOutObjects found = llm.generate(
      new LlmInImage(fromBufferedImage(photo), QUIET_SCORE), LlmModality.IMAGE);
    long recognized = System.nanoTime();
    List<String> classNames = llm.model().classNames();
    List<VisionHit> street = visibleHits(
      found.objects(), classNames, photo.getWidth(), photo.getHeight());
    Path written = writeMarked(photo, street, classNames, outputFile(head), head);
    System.out.println("recognized in " + seconds(recognized - loaded));
    System.out.println(street.size() + " objects, wrote " + written);
    for (VisionHit hit : street) {
      System.out.println("  " + label(hit, classNames));
    }
  }

  private static void writeLabels(
    final LLM llm,
    final BufferedImage photo,
    final long loaded
  ) throws IOException {
    LlmOutLabels found = llm.generate(
      new LlmInImage(fromBufferedImage(photo)), LlmModality.LABELS);
    long recognized = System.nanoTime();
    Path written = writePng(markLabels(photo, found.labels()), outputFile(StreetHead.CLASSIFY));
    System.out.println("recognized in " + seconds(recognized - loaded));
    System.out.println("wrote " + written);
    for (LlmLabelScore score : found.labels()) {
      System.out.println("  " + label(score));
    }
  }

  static BufferedImage markLabels(
    final BufferedImage photo,
    final List<LlmLabelScore> labels
  ) {
    requireNonNull(photo, "photo");
    requireNonNull(labels, "labels");
    if (labels.isEmpty()) {
      throw new IllegalArgumentException("labels must not be empty");
    }
    BufferedImage canvas = new BufferedImage(
      photo.getWidth(), photo.getHeight(), BufferedImage.TYPE_INT_RGB);
    Graphics2D pen = canvas.createGraphics();
    pen.drawImage(photo, 0, 0, null);
    pen.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
    FontMetrics metrics = pen.getFontMetrics();
    int top = 8;
    for (LlmLabelScore score : labels) {
      String text = label(score);
      int barWidth = metrics.stringWidth(text) + 8;
      int barHeight = metrics.getHeight();
      pen.setColor(new Color(20, 20, 20));
      pen.fillRect(8, top, barWidth, barHeight);
      pen.setColor(Color.WHITE);
      pen.drawString(text, 12, top + metrics.getAscent());
      top += barHeight + 4;
    }
    pen.dispose();
    return canvas;
  }

  private static void writeDepth(
    final LLM llm,
    final BufferedImage photo,
    final long loaded
  ) throws IOException {
    LlmOutImage depth = llm.generate(new LlmInImage(fromBufferedImage(photo)), LlmModality.IMAGE);
    long recognized = System.nanoTime();
    Path written = writePng(tintDepth(photo, depth.image()), outputFile(StreetHead.DEPTH));
    System.out.println("recognized in " + seconds(recognized - loaded));
    System.out.printf(
      ROOT,
      "nearest %.1f m, farthest %.1f m, wrote %s%n",
      depth.minimum(),
      depth.maximum(),
      written);
  }

  private static void writeScene(
    final LLM llm,
    final BufferedImage photo,
    final long loaded
  ) throws IOException {
    LlmOutImage scene = llm.generate(new LlmInImage(fromBufferedImage(photo)), LlmModality.IMAGE);
    long recognized = System.nanoTime();
    Path written = writePng(tintScene(photo, scene.image()), outputFile(StreetHead.SEMANTIC));
    System.out.println("recognized in " + seconds(recognized - loaded));
    System.out.println(sceneClasses(scene.image()) + ", wrote " + written);
  }

  static BufferedImage tintScene(final BufferedImage photo, final ArgbImage classes) {
    requireNonNull(photo, "photo");
    requireNonNull(classes, "classes");
    int width = photo.getWidth();
    int height = photo.getHeight();
    if (classes.width() != width || classes.height() != height) {
      throw new IllegalArgumentException("class raster must match the photo");
    }
    BufferedImage canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    int[] pixels = photo.getRGB(0, 0, width, height, null, 0, width);
    int[] indexes = classes.pixels();
    for (int i = 0; i < pixels.length; i++) {
      pixels[i] = blend(pixels[i], sceneTint(indexes[i] & 0xff));
    }
    canvas.setRGB(0, 0, width, height, pixels, 0, width);
    return canvas;
  }

  static String sceneClasses(final ArgbImage classes) {
    requireNonNull(classes, "classes");
    return IntStream.of(classes.pixels())
      .map(pixel -> pixel & 0xff)
      .filter(index -> index < SCENE.size())
      .distinct()
      .sorted()
      .mapToObj(index -> SCENE.get(index).name())
      .collect(joining(", "));
  }

  private static int sceneTint(final int index) {
    int rgb = index < SCENE.size()
      ? SCENE.get(index).rgb()
      : Color.HSBtoRGB((index % SCENE.size()) / (float) SCENE.size(), 0.7f, 0.9f) & 0xffffff;
    return (MASK_ALPHA << 24) | rgb;
  }

  static BufferedImage tintDepth(final BufferedImage photo, final ArgbImage depth) {
    requireNonNull(photo, "photo");
    requireNonNull(depth, "depth");
    int width = photo.getWidth();
    int height = photo.getHeight();
    if (depth.width() != width || depth.height() != height) {
      throw new IllegalArgumentException("depth raster must match the photo");
    }
    BufferedImage canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    int[] pixels = photo.getRGB(0, 0, width, height, null, 0, width);
    int[] shades = depth.pixels();
    for (int i = 0; i < pixels.length; i++) {
      pixels[i] = blend(pixels[i], depthTint(shades[i] & 0xff));
    }
    canvas.setRGB(0, 0, width, height, pixels, 0, width);
    return canvas;
  }

  private static int depthTint(final int gray) {
    int rgb = Color.HSBtoRGB(gray / (255f * 3f), 1f, 1f);
    return (DEPTH_ALPHA << 24) | (rgb & 0xffffff);
  }

  private static int blend(final int under, final int over) {
    int alpha = (over >>> 24) & 0xff;
    int rest = 255 - alpha;
    int red = (((over >> 16) & 0xff) * alpha + ((under >> 16) & 0xff) * rest + 127) / 255;
    int green = (((over >> 8) & 0xff) * alpha + ((under >> 8) & 0xff) * rest + 127) / 255;
    int blue = ((over & 0xff) * alpha + (under & 0xff) * rest + 127) / 255;
    return 0xff000000 | (red << 16) | (green << 8) | blue;
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

  static List<VisionHit> streetObjects(
    final List<VisionHit> hits,
    final List<String> classNames
  ) {
    requireNonNull(hits, "hits");
    requireNonNull(classNames, "classNames");
    return hits.stream()
      .filter(hit -> isStreetClass(classNames, hit.classId()))
      .toList();
  }

  static List<VisionHit> visibleHits(
    final List<VisionHit> hits,
    final List<String> classNames,
    final int width,
    final int height
  ) {
    requireNonNull(hits, "hits");
    requireNonNull(classNames, "classNames");
    if (width < 1 || height < 1) {
      throw new IllegalArgumentException("picture must be non-empty");
    }
    float quietArea = width * (float) height * QUIET_AREA_FRACTION;
    List<VisionHit> ranked = streetObjects(hits, classNames).stream()
      .filter(hit -> isVisible(hit, quietArea))
      .sorted(Comparator.comparingDouble(VisionHit::probability).reversed())
      .toList();
    List<VisionHit> kept = new ArrayList<>();
    for (VisionHit hit : ranked) {
      if (hit.probability() >= LlmInImage.DEFAULT_MIN_PROBABILITY ||
        !coveredByStronger(hit, kept)) {
        kept.add(hit);
      }
    }
    return List.copyOf(kept);
  }

  private static boolean isVisible(final VisionHit hit, final float quietArea) {
    if (hit.probability() >= LlmInImage.DEFAULT_MIN_PROBABILITY) {
      return true;
    }
    return hit.probability() >= QUIET_SCORE && area(bounds(hit.shape())) >= quietArea;
  }

  private static boolean isStreetClass(final List<String> classNames, final int classId) {
    return classId < classNames.size()
      && STREET_CLASSES.contains(classNames.get(classId).toLowerCase(ROOT));
  }

  private static boolean coveredByStronger(final VisionHit hit, final List<VisionHit> kept) {
    Box box = bounds(hit.shape());
    int classId = hit.classId();
    return kept.stream().anyMatch(stronger ->
      stronger.classId() == classId
        && overlapOfSmaller(box, bounds(stronger.shape())) > 0.5f);
  }

  private static float overlapOfSmaller(final Box box, final Box other) {
    float width = Math.min(box.right(), other.right()) - Math.max(box.left(), other.left());
    float height = Math.min(box.bottom(), other.bottom()) - Math.max(box.top(), other.top());
    if (width <= 0f || height <= 0f) {
      return 0f;
    }
    float smaller = Math.min(area(box), area(other));
    return smaller <= 0f ? 1f : width * height / smaller;
  }

  private static float area(final Box box) {
    return Math.max(0f, box.right() - box.left()) * Math.max(0f, box.bottom() - box.top());
  }

  static String className(final List<String> classNames, final int classId) {
    requireNonNull(classNames, "classNames");
    return classId >= 0 && classId < classNames.size()
      ? classNames.get(classId)
      : "class-" + classId;
  }

  static String label(final VisionHit hit, final List<String> classNames) {
    requireNonNull(hit, "hit");
    return "%s %s".formatted(
      className(classNames, hit.classId()), String.format(ROOT, "%.2f", hit.probability()));
  }

  static String label(final LlmLabelScore score) {
    requireNonNull(score, "score");
    return "%s %s".formatted(score.label(), String.format(ROOT, "%.2f", score.score()));
  }

  static BufferedImage mark(
    final BufferedImage source,
    final List<VisionHit> hits,
    final List<String> classNames
  ) {
    return mark(source, hits, classNames, StreetHead.DETECT);
  }

  static BufferedImage mark(
    final BufferedImage source,
    final List<VisionHit> hits,
    final List<String> classNames,
    final StreetHead head
  ) {
    requireNonNull(source, "source");
    requireNonNull(hits, "hits");
    requireNonNull(classNames, "classNames");
    requireNonNull(head, "head");
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
    boolean fillMask = head == StreetHead.SEGMENT;
    for (VisionHit hit : ranked) {
      paintShape(pen, hit, classNames, fillMask);
    }
    List<Rectangle> occupied = new ArrayList<>();
    for (VisionHit hit : ranked) {
      paintLabel(pen, hit, classNames, canvas.getWidth(), canvas.getHeight(), occupied);
    }
    pen.dispose();
    return canvas;
  }

  static Path writeMarked(
    final BufferedImage source,
    final List<VisionHit> hits,
    final List<String> classNames,
    final Path out
  ) throws IOException {
    return writeMarked(source, hits, classNames, out, StreetHead.DETECT);
  }

  static Path writeMarked(
    final BufferedImage source,
    final List<VisionHit> hits,
    final List<String> classNames,
    final Path out,
    final StreetHead head
  ) throws IOException {
    return writePng(mark(source, hits, classNames, head), out);
  }

  private static Path writePng(final BufferedImage image, final Path out) throws IOException {
    requireNonNull(out, "out");
    Path parent = out.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    if (!ImageIO.write(image, "png", out.toFile())) {
      throw new IllegalStateException("PNG writer missing for " + out);
    }
    return out.toAbsolutePath();
  }

  private static void printMenu(final PrintStream out) {
    out.println("Select a YOLO26 head. People, bicycles, and cars are marked.");
    out.println("Semantic paints each surface in its own color.");
    out.println("Depth paints the street from red (near) to green (far).");
    out.println("Classification writes the top labels on the street.");
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
      case "sem", "semantic" -> StreetHead.SEMANTIC;
      case "pose" -> StreetHead.POSE;
      case "obb", "oriented" -> StreetHead.OBB;
      case "depth" -> StreetHead.DEPTH;
      case "cls", "classify", "classification" -> StreetHead.CLASSIFY;
      default -> null;
    };
  }

  private static StreetHead headOfFileName(final String fileName) {
    String name = fileName.toLowerCase(ROOT);
    if (!name.contains("yolo26")) {
      throw unknown(fileName);
    }
    if (name.contains("-depth")) {
      return StreetHead.DEPTH;
    }
    if (name.contains("-sem")) {
      return StreetHead.SEMANTIC;
    }
    if (name.contains("-cls")) {
      return StreetHead.CLASSIFY;
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

  private static void paintShape(
    final Graphics2D pen,
    final VisionHit hit,
    final List<String> classNames,
    final boolean fillMask
  ) {
    Color color = ink(className(classNames, hit.classId()));
    pen.setColor(color);
    drawShape(pen, hit.shape(), fillMask);
    paintLimbs(pen, hit.keypoints());
    pen.setColor(color);
    for (Keypoint joint : hit.keypoints()) {
      pen.fillOval(Math.round(joint.x()) - 3, Math.round(joint.y()) - 3, 6, 6);
    }
  }

  private static void paintLimbs(final Graphics2D pen, final List<Keypoint> joints) {
    if (joints.isEmpty()) {
      return;
    }
    Keypoint[] placed = new Keypoint[Keypoint.COUNT];
    for (Keypoint joint : joints) {
      placed[joint.joint()] = joint;
    }

    Stroke stroke = pen.getStroke();
    pen.setStroke(new BasicStroke(3f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
    pen.setColor(LIMB);
    for (Limb limb : LIMBS) {
      drawLimb(pen, placed, limb);
    }
    pen.setStroke(stroke);
  }

  private static void drawLimb(
    final Graphics2D pen,
    final Keypoint[] placed,
    final Limb limb
  ) {
    Keypoint from = placed[limb.from()];
    Keypoint to = placed[limb.to()];
    if (from == null || to == null) {
      return;
    }
    pen.drawLine(
      Math.round(from.x()), Math.round(from.y()),
      Math.round(to.x()), Math.round(to.y()));
  }

  private static void drawShape(
    final Graphics2D pen,
    final Shape2D shape,
    final boolean fillMask
  ) {
    switch (shape) {
      case Box box -> pen.drawRect(
        Math.round(box.left()),
        Math.round(box.top()),
        Math.max(1, Math.round(box.right() - box.left())),
        Math.max(1, Math.round(box.bottom() - box.top())));
      case Polygon polygon -> paintPolygon(pen, polygon, fillMask);
    }
  }

  private static void paintPolygon(
    final Graphics2D pen,
    final Polygon polygon,
    final boolean fillMask
  ) {
    int[] xs = polygon.points().stream().mapToInt(point -> Math.round(point.x())).toArray();
    int[] ys = polygon.points().stream().mapToInt(point -> Math.round(point.y())).toArray();
    if (fillMask) {
      Color ink = pen.getColor();
      pen.setColor(translucent(ink));
      pen.fillPolygon(xs, ys, xs.length);
      pen.setColor(ink);
    }
    pen.drawPolygon(xs, ys, xs.length);
  }

  private static Color translucent(final Color color) {
    return new Color(color.getRed(), color.getGreen(), color.getBlue(), MASK_ALPHA);
  }

  private static void paintLabel(
    final Graphics2D pen,
    final VisionHit hit,
    final List<String> classNames,
    final int width,
    final int height,
    final List<Rectangle> occupied
  ) {
    String text = label(hit, classNames);
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

    pen.setColor(ink(className(classNames, hit.classId())));
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
      "unknown YOLO26 head: " + text + ". Choose detect, seg, sem, pose, obb, depth, or cls.");
  }

  enum StreetHead {
    DETECT("detect", "models/YOLO26n/detect/yolo26n.onnx", "a box on each object"),
    SEGMENT("seg", "models/YOLO26n/seg/yolo26n-seg.onnx", "a tinted outline of each object"),
    SEMANTIC("sem", "models/YOLO26n/sem/yolo26n-sem.onnx", "each surface in its own color"),
    POSE("pose", "models/YOLO26n/pose/yolo26n-pose.onnx", "a box and the body"),
    OBB("obb", "models/YOLO26n/obb/yolo26n-obb.onnx", "rotated corners"),
    DEPTH("depth", "models/YOLO26n/depth/yolo26n-depth.onnx", "red near, green far"),
    CLASSIFY("cls", "models/YOLO26n/cls/yolo26n-cls.onnx", "the top labels");

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

  private record SceneClass(String name, int rgb) {
  }

  private record Limb(int from, int to) {
  }

  record Selection(StreetHead head, Path model) {
    Selection {
      requireNonNull(head, "head");
      requireNonNull(model, "model");
    }
  }
}
