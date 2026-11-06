package com.igormaznitsa.nanollvm.models.internal.yolo;

import com.igormaznitsa.nanollvm.layers.Conv2d;
import com.igormaznitsa.nanollvm.layers.ConvTranspose2d;
import com.igormaznitsa.nanollvm.models.ArgbImage;
import com.igormaznitsa.nanollvm.models.Box;
import com.igormaznitsa.nanollvm.models.LlmLabelScore;
import com.igormaznitsa.nanollvm.models.LlmOutImage;
import com.igormaznitsa.nanollvm.models.LlmOutLabels;
import com.igormaznitsa.nanollvm.models.LlmOutObjects;
import com.igormaznitsa.nanollvm.models.LlmOutput;
import com.igormaznitsa.nanollvm.models.Point2D;
import com.igormaznitsa.nanollvm.models.Polygon;
import com.igormaznitsa.nanollvm.models.Shape2D;
import com.igormaznitsa.nanollvm.models.VisionHit;
import com.igormaznitsa.nanollvm.tensor.MatmulRuntime;
import com.igormaznitsa.nanollvm.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Task head on top of {@link YoloTrunk}. Detection uses the one-to-one end-to-end branch.
 */
final class YoloHead {

  private static final int POSE_POINTS = 17;
  private static final int POSE_DIMS = 3;
  private final YoloTask task;
  private final Level[] levels;
  private final Proto proto;
  private final Semantic semantic;
  private final Depth depth;
  private final Classify classify;
  private final int classes;

  private YoloHead(
    final YoloTask task,
    final Level[] levels,
    final Proto proto,
    final Semantic semantic,
    final Depth depth,
    final Classify classify,
    final int classes
  ) {
    this.task = task;
    this.levels = levels;
    this.proto = proto;
    this.semantic = semantic;
    this.depth = depth;
    this.classify = classify;
    this.classes = classes;
  }

  static YoloHead load(final YoloTensors weights, final YoloWidths width, final YoloTask task) {
    int classes = declaredClasses(weights, task);
    return switch (task) {
      case CLASSIFY -> new YoloHead(
        task, null, null, null, null, Classify.load(weights, width.c1024(), classes), classes);
      case SEMANTIC -> new YoloHead(
        task, null, null, Semantic.load(weights, width.c256(), classes), null, null, classes);
      case DEPTH -> new YoloHead(
        task, null, null, null,
        Depth.load(weights, width.c256(), width.c512(), width.c1024()), null, 0);
      case DETECT, SEGMENT, POSE, OBB -> pyramid(weights, width, task, classes);
    };
  }

  private static Box box(final Raw raw, final YoloLetterbox letter) {
    Point2D a = letter.toSource(raw.x1, raw.y1);
    Point2D b = letter.toSource(raw.x2, raw.y2);
    float left = Math.min(a.x(), b.x());
    float top = Math.min(a.y(), b.y());
    float right = Math.max(a.x(), b.x());
    float bottom = Math.max(a.y(), b.y());
    return new Box(left, top, Math.max(right, left), Math.max(bottom, top));
  }

  private static Polygon oriented(final Raw raw, final YoloLetterbox letter) {
    float cos = (float) Math.cos(raw.angle);
    float sin = (float) Math.sin(raw.angle);
    float dx = raw.width / 2f;
    float dy = raw.height / 2f;
    float vx = dx * cos;
    float vy = dx * sin;
    float ux = -dy * sin;
    float uy = dy * cos;
    return new Polygon(List.of(
      letter.toSource(raw.cx + vx + ux, raw.cy + vy + uy),
      letter.toSource(raw.cx + vx - ux, raw.cy + vy - uy),
      letter.toSource(raw.cx - vx - ux, raw.cy - vy - uy),
      letter.toSource(raw.cx - vx + ux, raw.cy - vy + uy)));
  }

  private static List<Point2D> keypoints(final Raw raw, final YoloLetterbox letter) {
    List<Point2D> points = new ArrayList<>();
    for (int i = 0; i < raw.extra.length; i += 3) {
      float visible = 1f / (1f + (float) Math.exp(-raw.extra[i + 2]));
      if (visible < 0.5f) {
        continue;
      }
      float x = (raw.extra[i] + raw.anchorX) * raw.stride;
      float y = (raw.extra[i + 1] + raw.anchorY) * raw.stride;
      points.add(letter.toSource(x, y));
    }
    return List.copyOf(points);
  }

  private static Polygon outline(final Raw raw, final Tensor proto, final YoloLetterbox letter) {
    Box bounds = box(raw, letter);
    if (proto == null || raw.extra.length == 0 || proto.shape().length < 3) {
      return rectangle(bounds);
    }
    int channels = Math.min(proto.size(0), raw.extra.length);
    int height = proto.size(1);
    int width = proto.size(2);
    int x0 = protoPixel(raw.x1, width, letter.tensor.size(2), false);
    int y0 = protoPixel(raw.y1, height, letter.tensor.size(1), false);
    int x1 = protoPixel(raw.x2, width, letter.tensor.size(2), true);
    int y1 = protoPixel(raw.y2, height, letter.tensor.size(1), true);
    if (x1 - x0 < 1 || y1 - y0 < 1) {
      return rectangle(bounds);
    }
    boolean[] mask = maskCrop(proto, raw.extra, channels, x0, y0, x1, y1, width);
    List<Point2D> local = orderedMaskOutline(mask, x1 - x0, y1 - y0);
    if (local.size() < 3) {
      return rectangle(bounds);
    }
    return new Polygon(sourceOutline(thin(local, 72), x0, y0, width, height, letter));
  }

  static List<Point2D> orderedMaskOutline(final boolean[] mask, final int width, final int height) {
    if (width < 1 || height < 1 || mask.length != width * height) {
      throw new IllegalArgumentException("mask length must be width * height");
    }
    boolean[] component = largestForeground(mask, width, height);
    int start = -1;
    for (int i = 0; i < component.length; i++) {
      if (component[i]) {
        start = i;
        break;
      }
    }
    if (start < 0) {
      return List.of();
    }
    int[] dx = {1, 1, 0, -1, -1, -1, 0, 1};
    int[] dy = {0, 1, 1, 1, 0, -1, -1, -1};
    int x = start % width;
    int y = start / width;
    int originX = x;
    int originY = y;
    int search = 4;
    List<Point2D> path = new ArrayList<>();
    for (int step = 0; step < mask.length; step++) {
      path.add(new Point2D(x, y));
      boolean found = false;
      for (int turn = 0; turn < 8; turn++) {
        int dir = (search + turn) % 8;
        int nx = x + dx[dir];
        int ny = y + dy[dir];
        if (nx < 0 || ny < 0 || nx >= width || ny >= height || !component[ny * width + nx]) {
          continue;
        }
        search = (dir + 5) % 8;
        x = nx;
        y = ny;
        found = true;
        break;
      }
      if (!found || (x == originX && y == originY)) {
        break;
      }
    }
    return path;
  }

  private static int protoPixel(
    final float letterCoord,
    final int protoSize,
    final int letterSize,
    final boolean upper
  ) {
    float scaled = letterCoord * protoSize / letterSize;
    int pixel = upper ? (int) Math.ceil(scaled) : (int) Math.floor(scaled);
    return Math.clamp(pixel, 0, protoSize);
  }

  private static boolean[] maskCrop(
    final Tensor proto,
    final float[] coeff,
    final int channels,
    final int x0,
    final int y0,
    final int x1,
    final int y1,
    final int width
  ) {
    int cropWidth = x1 - x0;
    int cropHeight = y1 - y0;
    int protoHeight = proto.size(1);
    boolean[] mask = new boolean[cropWidth * cropHeight];
    float[] data = proto.data();
    int off = proto.offset();
    for (int y = y0; y < y1; y++) {
      for (int x = x0; x < x1; x++) {
        float dot = 0f;
        for (int c = 0; c < channels; c++) {
          dot += coeff[c] * data[off + (c * protoHeight + y) * width + x];
        }
        if (dot >= 0f) {
          mask[(y - y0) * cropWidth + (x - x0)] = true;
        }
      }
    }
    return mask;
  }

  private static boolean[] largestForeground(final boolean[] mask, final int width,
                                             final int height) {
    boolean[] seen = new boolean[mask.length];
    int[] stack = new int[mask.length];
    int bestSeed = -1;
    int bestSize = 0;
    for (int i = 0; i < mask.length; i++) {
      if (!mask[i] || seen[i]) {
        continue;
      }
      int size = flood(mask, seen, null, width, height, i, stack);
      if (size > bestSize) {
        bestSize = size;
        bestSeed = i;
      }
    }
    if (bestSeed < 0) {
      return mask;
    }
    boolean[] keep = new boolean[mask.length];
    flood(mask, new boolean[mask.length], keep, width, height, bestSeed, stack);
    return keep;
  }

  private static int flood(
    final boolean[] mask,
    final boolean[] seen,
    final boolean[] keep,
    final int width,
    final int height,
    final int seed,
    final int[] stack
  ) {
    int top = 0;
    stack[top++] = seed;
    seen[seed] = true;
    if (keep != null) {
      keep[seed] = true;
    }
    int size = 0;
    int[] dx = {1, 1, 0, -1, -1, -1, 0, 1};
    int[] dy = {0, 1, 1, 1, 0, -1, -1, -1};
    while (top > 0) {
      int at = stack[--top];
      size++;
      int x = at % width;
      int y = at / width;
      for (int k = 0; k < 8; k++) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
          continue;
        }
        int next = ny * width + nx;
        if (!mask[next] || seen[next]) {
          continue;
        }
        seen[next] = true;
        if (keep != null) {
          keep[next] = true;
        }
        stack[top++] = next;
      }
    }
    return size;
  }

  private static List<Point2D> thin(final List<Point2D> path, final int limit) {
    if (path.size() <= limit) {
      return path;
    }
    double step = path.size() / (double) limit;
    List<Point2D> sampled = new ArrayList<>(limit);
    for (int i = 0; i < limit; i++) {
      sampled.add(path.get((int) (i * step)));
    }
    return sampled;
  }

  private static List<Point2D> sourceOutline(
    final List<Point2D> local,
    final int x0,
    final int y0,
    final int width,
    final int height,
    final YoloLetterbox letter
  ) {
    int letterWidth = letter.tensor.size(2);
    int letterHeight = letter.tensor.size(1);
    return local.stream()
      .map(point -> letter.toSource(
        (x0 + point.x() + 0.5f) * letterWidth / width,
        (y0 + point.y() + 0.5f) * letterHeight / height))
      .toList();
  }

  private static Polygon rectangle(final Box box) {
    float right = box.right() > box.left() ? box.right() : box.left() + 1f;
    float bottom = box.bottom() > box.top() ? box.bottom() : box.top() + 1f;
    return new Polygon(List.of(
      new Point2D(box.left(), box.top()),
      new Point2D(right, box.top()),
      new Point2D(right, bottom),
      new Point2D(box.left(), bottom)));
  }

  private static List<Pick> topPicks(final List<Raw> raws) {
    Integer[] order = new Integer[raws.size()];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    Arrays.sort(order, Comparator.comparingDouble((Integer i) -> raws.get(i).best).reversed());
    int anchors = Math.min(Yolo26Model.MAX_DET, order.length);
    List<Pick> ranked = new ArrayList<>();
    for (int i = 0; i < anchors; i++) {
      Raw raw = raws.get(order[i]);
      for (int cls = 0; cls < raw.scores.length; cls++) {
        ranked.add(new Pick(order[i], cls, raw.scores[cls]));
      }
    }
    ranked.sort(Comparator.comparingDouble(Pick::score).reversed());
    return ranked.subList(0, Math.min(Yolo26Model.MAX_DET, ranked.size()));
  }

  private static YoloHead pyramid(
    final YoloTensors weights,
    final YoloWidths width,
    final YoloTask task,
    final int classes
  ) {
    int[] channels = {width.c256(), width.c512(), width.c1024()};
    int[] strides = {8, 16, 32};
    int c2 = Math.max(16, Math.max(channels[0] / 4, 4));
    int c3 = Math.clamp(classes, channels[0], 100);
    int extra = extraChannels(task);
    int c4 = sideChannels(channels[0], extra, task);
    Level[] levels = new Level[3];
    for (int i = 0; i < levels.length; i++) {
      levels[i] = Level.load(
        weights, "model.23", i, channels[i], c2, c3, c4, classes, extra, strides[i], task);
    }
    int masks = task == YoloTask.SEGMENT ? 32 : 0;
    int protos = task == YoloTask.SEGMENT
      ? YoloMath.channels(256, width.scale.width(), width.scale.maxChannels())
      : 0;
    Proto proto = masks == 0 ? null : Proto.load(weights, channels, protos, masks);
    return new YoloHead(task, levels, proto, null, null, null, classes);
  }

  private static int declaredClasses(final YoloTensors weights, final YoloTask task) {
    if (task == YoloTask.CLASSIFY || task == YoloTask.SEMANTIC || task == YoloTask.DEPTH) {
      return task.classes();
    }
    String classWeight = "model.23.one2one_cv3.0.2.weight";
    if (!weights.has(classWeight)) {
      return task.classes();
    }
    return weights.shape(classWeight)[0];
  }

  private static int extraChannels(final YoloTask task) {
    return switch (task) {
      case SEGMENT -> 32;
      case POSE -> POSE_POINTS * POSE_DIMS;
      case OBB -> 1;
      default -> 0;
    };
  }

  private static int sideChannels(final int stem, final int extra, final YoloTask task) {
    if (extra == 0) {
      return 0;
    }
    int hidden = task == YoloTask.POSE ? POSE_POINTS * (POSE_DIMS + 2) : extra;
    return Math.max(stem / 4, hidden);
  }

  LlmOutput read(
    final List<Tensor> layers,
    final YoloLetterbox letter,
    final List<String> names,
    final float minProbability,
    final MatmulRuntime runtime
  ) {
    if (this.classify != null) {
      return this.classify.read(layers.getLast(), names);
    }
    if (this.semantic != null) {
      return this.semantic.read(layers.get(16), letter);
    }
    if (this.depth != null) {
      return this.depth.read(List.of(layers.get(16), layers.get(19), layers.get(22)), letter,
        runtime);
    }
    return this.objects(layers, letter, names, minProbability, runtime);
  }

  private LlmOutput objects(
    final List<Tensor> layers,
    final YoloLetterbox letter,
    final List<String> names,
    final float minProbability,
    final MatmulRuntime runtime
  ) {
    Tensor[] feats = {layers.get(16), layers.get(19), layers.get(22)};
    List<Raw> raws = new ArrayList<>();
    for (int level = 0; level < this.levels.length; level++) {
      this.levels[level].collect(feats[level], runtime, raws, this.task);
    }
    List<Pick> picks = topPicks(raws);
    Tensor masks = this.proto == null ? null : this.proto.forward(feats, runtime);
    List<VisionHit> hits = new ArrayList<>();
    for (Pick pick : picks) {
      if (pick.score < minProbability) {
        continue;
      }
      hits.add(this.hit(raws.get(pick.anchor), pick, masks, letter, names));
    }
    return new LlmOutObjects(hits);
  }

  private VisionHit hit(
    final Raw raw,
    final Pick pick,
    final Tensor proto,
    final YoloLetterbox letter,
    final List<String> names
  ) {
    String name = pick.classId < names.size() ? names.get(pick.classId) : "class-" + pick.classId;
    Shape2D shape = switch (this.task) {
      case OBB -> oriented(raw, letter);
      case SEGMENT -> outline(raw, proto, letter);
      default -> box(raw, letter);
    };
    List<Point2D> keypoints = this.task == YoloTask.POSE ? keypoints(raw, letter) : List.of();
    return new VisionHit(name, pick.score, shape, keypoints);
  }

  private record Pick(int anchor, int classId, float score) {
  }

  private record Level(YoloConv box0, YoloConv box1, Conv2d boxOut, YoloConv clsDw0,
                       YoloConv clsPw0, YoloConv clsDw1, YoloConv clsPw1, Conv2d clsOut,
                       YoloConv extra0, YoloConv extra1, Conv2d extraOut, int stride,
                       YoloTask task) {

    static Level load(
      final YoloTensors weights,
      final String prefix,
      final int index,
      final int channels,
      final int c2,
      final int c3,
      final int c4,
      final int classes,
      final int extra,
      final int stride,
      final YoloTask task
    ) {
      String box = prefix + ".one2one_cv2." + index;
      String cls = prefix + ".one2one_cv3." + index;
      String side = prefix + ".one2one_cv4." + index;
      YoloConv extra0 = null;
      YoloConv extra1 = null;
      Conv2d extraOut = null;
      if (extra > 0 && task != YoloTask.POSE) {
        extra0 = YoloConv.load(weights, side + ".0", channels, c4, 3, 1, 1, true);
        extra1 = YoloConv.load(weights, side + ".1", c4, c4, 3, 1, 1, true);
        extraOut = YoloConv.loadBare(weights, side + ".2", c4, extra, 1, 1, 1);
      }
      if (task == YoloTask.POSE) {
        extra0 = YoloConv.load(weights, side + ".0", channels, c4, 3, 1, 1, true);
        extra1 = YoloConv.load(weights, side + ".1", c4, c4, 3, 1, 1, true);
        extraOut = YoloConv.loadBare(
          weights, prefix + ".one2one_cv4_kpts." + index, c4, extra, 1, 1, 1);
      }
      return new Level(
        YoloConv.load(weights, box + ".0", channels, c2, 3, 1, 1, true),
        YoloConv.load(weights, box + ".1", c2, c2, 3, 1, 1, true),
        YoloConv.loadBare(weights, box + ".2", c2, 4, 1, 1, 1),
        YoloConv.load(weights, cls + ".0.0", channels, channels, 3, 1, channels, true),
        YoloConv.load(weights, cls + ".0.1", channels, c3, 1, 1, 1, true),
        YoloConv.load(weights, cls + ".1.0", c3, c3, 3, 1, c3, true),
        YoloConv.load(weights, cls + ".1.1", c3, c3, 1, 1, 1, true),
        YoloConv.loadBare(weights, cls + ".2", c3, classes, 1, 1, 1),
        extra0,
        extra1,
        extraOut,
        stride,
        task);
    }

    void collect(
      final Tensor feature,
      final MatmulRuntime runtime,
      final List<Raw> raws,
      final YoloTask task
    ) {
      Tensor boxes = this.boxOut.forward(
        this.box1.forward(this.box0.forward(feature, runtime), runtime), runtime);
      Tensor scores = this.clsOut.forward(this.clsPw1.forward(this.clsDw1.forward(
          this.clsPw0.forward(this.clsDw0.forward(feature, runtime), runtime), runtime), runtime),
        runtime);
      Tensor extra = this.extraOut == null
        ? null
        : this.extraOut.forward(
        this.extra1.forward(this.extra0.forward(feature, runtime), runtime), runtime);
      int height = feature.size(1);
      int width = feature.size(2);
      int classes = scores.size(0);
      int extraChannels = extra == null ? 0 : extra.size(0);
      float[] boxData = boxes.data();
      float[] scoreData = scores.data();
      float[] extraData = extra == null ? null : extra.data();
      int boxOff = boxes.offset();
      int scoreOff = scores.offset();
      int extraOff = extra == null ? 0 : extra.offset();
      int plane = height * width;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int at = y * width + x;
          float anchorX = x + 0.5f;
          float anchorY = y + 0.5f;
          float left = boxData[boxOff + at];
          float top = boxData[boxOff + plane + at];
          float right = boxData[boxOff + 2 * plane + at];
          float bottom = boxData[boxOff + 3 * plane + at];
          float[] cls = new float[classes];
          float best = 0f;
          for (int c = 0; c < classes; c++) {
            float logit = scoreData[scoreOff + c * plane + at];
            cls[c] = 1f / (1f + (float) Math.exp(-logit));
            best = Math.max(best, cls[c]);
          }
          float[] side = new float[extraChannels];
          for (int c = 0; c < extraChannels; c++) {
            side[c] = extraData[extraOff + c * plane + at];
          }
          raws.add(Raw.of(
            task, anchorX, anchorY, this.stride, left, top, right, bottom, cls, best, side));
        }
      }
    }
  }

  private record Raw(float x1, float y1, float x2, float y2, float cx, float cy, float width,
                     float height, float angle, float anchorX, float anchorY, float stride,
                     float[] scores, float[] extra, float best) {

    static Raw of(
      final YoloTask task,
      final float anchorX,
      final float anchorY,
      final int stride,
      final float left,
      final float top,
      final float right,
      final float bottom,
      final float[] scores,
      final float best,
      final float[] extra
    ) {
      if (task == YoloTask.OBB) {
        float angle = extra.length == 0 ? 0f : extra[0];
        float cos = (float) Math.cos(angle);
        float sin = (float) Math.sin(angle);
        float xf = (right - left) / 2f;
        float yf = (bottom - top) / 2f;
        float cx = (anchorX + xf * cos - yf * sin) * stride;
        float cy = (anchorY + xf * sin + yf * cos) * stride;
        return new Raw(
          0f, 0f, 0f, 0f, cx, cy, (left + right) * stride, (top + bottom) * stride, angle,
          anchorX, anchorY, stride, scores, extra, best);
      }
      return new Raw(
        (anchorX - left) * stride,
        (anchorY - top) * stride,
        (anchorX + right) * stride,
        (anchorY + bottom) * stride,
        0f, 0f, 0f, 0f, 0f,
        anchorX, anchorY, stride, scores, extra, best);
    }
  }

  private record Proto(YoloConv refine4, YoloConv refine5, YoloConv fuse, YoloConv cv1,
                       ConvTranspose2d upsample, YoloConv cv2, YoloConv cv3) {

    static Proto load(
      final YoloTensors weights,
      final int[] channels,
      final int protos,
      final int masks
    ) {
      String prefix = "model.23.proto";
      return new Proto(
        YoloConv.load(weights, prefix + ".feat_refine.0", channels[1], channels[0], 1, 1, 1, true),
        YoloConv.load(weights, prefix + ".feat_refine.1", channels[2], channels[0], 1, 1, 1, true),
        YoloConv.load(weights, prefix + ".feat_fuse", channels[0], protos, 3, 1, 1, true),
        YoloConv.load(weights, prefix + ".cv1", protos, protos, 3, 1, 1, true),
        YoloConv.loadTranspose(weights, prefix + ".upsample", protos, 2, 2),
        YoloConv.load(weights, prefix + ".cv2", protos, protos, 3, 1, 1, true),
        YoloConv.load(weights, prefix + ".cv3", protos, masks, 1, 1, 1, true));
    }

    Tensor forward(final Tensor[] features, final MatmulRuntime runtime) {
      Tensor feat = YoloMath.add(
        features[0], YoloMath.nearestScale(this.refine4.forward(features[1], runtime), 2));
      feat =
        YoloMath.add(feat, YoloMath.nearestScale(this.refine5.forward(features[2], runtime), 4));
      Tensor fused = this.fuse.forward(feat, runtime);
      return this.cv3.forward(this.cv2.forward(
        this.upsample.forward(this.cv1.forward(fused, runtime)), runtime), runtime);
    }
  }

  private static int opaqueGray(final int level) {
    int gray = Math.clamp(level, 0, 255);
    return (0xff << 24) | (gray << 16) | (gray << 8) | gray;
  }

  private record Semantic(YoloConv conv, Conv2d project, int classes) {

    static Semantic load(final YoloTensors weights, final int channels, final int classes) {
      return new Semantic(
        YoloConv.load(weights, "model.17.classifier.0", channels, channels, 3, 1, 1, true),
        YoloConv.loadBare(weights, "model.17.classifier.1", channels, classes, 1, 1, 1),
        classes);
    }

    LlmOutImage read(final Tensor feature, final YoloLetterbox letter) {
      Tensor logits = this.project.forward(this.conv.forward(feature, MatmulRuntime.sequential()));
      int height = logits.size(1);
      int width = logits.size(2);
      float[] data = logits.data();
      int off = logits.offset();
      int[] pixels = new int[letter.sourceWidth * letter.sourceHeight];
      for (int y = 0; y < letter.sourceHeight; y++) {
        int srcY = Math.min(height - 1, y * height / letter.sourceHeight);
        for (int x = 0; x < letter.sourceWidth; x++) {
          int srcX = Math.min(width - 1, x * width / letter.sourceWidth);
          int best = 0;
          float bestValue = Float.NEGATIVE_INFINITY;
          for (int c = 0; c < this.classes; c++) {
            float value = data[off + (c * height + srcY) * width + srcX];
            if (value > bestValue) {
              bestValue = value;
              best = c;
            }
          }
          pixels[y * letter.sourceWidth + x] = opaqueGray(best);
        }
      }
      return new LlmOutImage(
        new ArgbImage(letter.sourceWidth, letter.sourceHeight, pixels),
        0f,
        Math.max(0, this.classes - 1));
    }
  }

  private record Depth(YoloConv[] project, YoloConv[] refineA, YoloConv[] refineB, YoloConv head0,
                       ConvTranspose2d head1, YoloConv head2, Conv2d head3, float scale,
                       float shift) {

    static Depth load(
      final YoloTensors weights,
      final int p3,
      final int p4,
      final int p5
    ) {
      int mid = 256;
      Tensor calA = weights.optional("model.23.cal_a", 1);
      Tensor calB = weights.optional("model.23.cal_b", 1);
      return new Depth(
        new YoloConv[] {
          YoloConv.load(weights, "model.23.proj.0", p3, mid, 1, 1, 1, true),
          YoloConv.load(weights, "model.23.proj.1", p4, mid, 1, 1, 1, true),
          YoloConv.load(weights, "model.23.proj.2", p5, mid, 1, 1, 1, true)
        },
        new YoloConv[] {
          YoloConv.load(weights, "model.23.refine.0.0", mid, mid, 3, 1, 1, true),
          YoloConv.load(weights, "model.23.refine.1.0", mid, mid, 3, 1, 1, true)
        },
        new YoloConv[] {
          YoloConv.load(weights, "model.23.refine.0.1", mid, mid, 3, 1, 1, true),
          YoloConv.load(weights, "model.23.refine.1.1", mid, mid, 3, 1, 1, true)
        },
        YoloConv.load(weights, "model.23.head.0", mid, mid / 2, 3, 1, 1, true),
        YoloConv.loadTranspose(weights, "model.23.head.1", mid / 2, 2, 2),
        YoloConv.load(weights, "model.23.head.2", mid / 2, mid / 4, 3, 1, 1, true),
        YoloConv.loadBare(weights, "model.23.head.3", mid / 4, 1, 1, 1, 1),
        calA == null ? 1f : calA.data()[calA.offset()],
        calB == null ? 0f : calB.data()[calB.offset()]);
    }

    LlmOutImage read(final List<Tensor> features, final YoloLetterbox letter,
                     final MatmulRuntime runtime) {
      Tensor[] projected = new Tensor[this.project.length];
      for (int i = 0; i < projected.length; i++) {
        projected[i] = this.project[i].forward(features.get(i), runtime);
      }
      Tensor out = projected[projected.length - 1];
      for (int i = projected.length - 2; i >= 0; i--) {
        Tensor up = YoloMath.bilinear(out, projected[i].size(1), projected[i].size(2), true);
        out = this.refineB[i].forward(
          this.refineA[i].forward(YoloMath.add(up, projected[i]), runtime), runtime);
      }
      Tensor logDepth = this.head3.forward(this.head2.forward(
        this.head1.forward(this.head0.forward(out, runtime)), runtime), runtime);
      int height = logDepth.size(1);
      int width = logDepth.size(2);
      float[] data = logDepth.data();
      int off = logDepth.offset();
      float min = Float.POSITIVE_INFINITY;
      float max = Float.NEGATIVE_INFINITY;
      float[] meters = new float[height * width];
      for (int i = 0; i < meters.length; i++) {
        float clamped = Math.clamp(data[off + i], -4f, 5f);
        meters[i] = (float) Math.exp(this.scale * clamped + this.shift);
        min = Math.min(min, meters[i]);
        max = Math.max(max, meters[i]);
      }
      int[] pixels = new int[letter.sourceWidth * letter.sourceHeight];
      float span = Math.max(max - min, 1e-6f);
      for (int y = 0; y < letter.sourceHeight; y++) {
        float srcY = letter.sourceHeight == 1 ? 0f : y * (height - 1f) / (letter.sourceHeight - 1f);
        int y0 = Math.clamp((int) Math.floor(srcY), 0, height - 1);
        int y1 = Math.min(y0 + 1, height - 1);
        float fy = srcY - y0;
        for (int x = 0; x < letter.sourceWidth; x++) {
          float srcX = letter.sourceWidth == 1 ? 0f : x * (width - 1f) / (letter.sourceWidth - 1f);
          int x0 = Math.clamp((int) Math.floor(srcX), 0, width - 1);
          int x1 = Math.min(x0 + 1, width - 1);
          float fx = srcX - x0;
          float value = meters[y0 * width + x0] * (1f - fx) * (1f - fy)
            + meters[y0 * width + x1] * fx * (1f - fy)
            + meters[y1 * width + x0] * (1f - fx) * fy
            + meters[y1 * width + x1] * fx * fy;
          int gray = Math.clamp(Math.round((value - min) / span * 255f), 0, 255);
          pixels[y * letter.sourceWidth + x] = opaqueGray(gray);
        }
      }
      return new LlmOutImage(new ArgbImage(letter.sourceWidth, letter.sourceHeight, pixels), min,
        max);
    }
  }

  private record Classify(YoloConv conv, Tensor weight, Tensor bias) {

    static Classify load(final YoloTensors weights, final int channels, final int classes) {
      return new Classify(
        YoloConv.load(weights, "model.10.conv", channels, 1280, 1, 1, 1, true),
        weights.take("model.10.linear.weight", classes, 1280),
        weights.take("model.10.linear.bias", classes));
    }

    LlmOutLabels read(final Tensor feature, final List<String> names) {
      Tensor conv = this.conv.forward(feature, MatmulRuntime.sequential());
      int channels = conv.size(0);
      int plane = conv.size(1) * conv.size(2);
      float[] pooled = new float[channels];
      float[] data = conv.data();
      int off = conv.offset();
      for (int c = 0; c < channels; c++) {
        float sum = 0f;
        int base = off + c * plane;
        for (int i = 0; i < plane; i++) {
          sum += data[base + i];
        }
        pooled[c] = sum / plane;
      }
      int classes = this.bias.numel();
      float[] logits = new float[classes];
      float[] matrix = this.weight.data();
      int weightOff = this.weight.offset();
      float[] bias = this.bias.data();
      int biasOff = this.bias.offset();
      float max = Float.NEGATIVE_INFINITY;
      for (int row = 0; row < classes; row++) {
        float sum = bias[biasOff + row];
        int base = weightOff + row * channels;
        for (int c = 0; c < channels; c++) {
          sum += matrix[base + c] * pooled[c];
        }
        logits[row] = sum;
        max = Math.max(max, sum);
      }
      float normalizer = 0f;
      for (int row = 0; row < classes; row++) {
        logits[row] = (float) Math.exp(logits[row] - max);
        normalizer += logits[row];
      }
      Integer[] order = new Integer[classes];
      for (int i = 0; i < classes; i++) {
        order[i] = i;
      }
      Arrays.sort(order, Comparator.comparingDouble((Integer i) -> logits[i]).reversed());
      int keep = Math.min(5, classes);
      List<LlmLabelScore> labels = new ArrayList<>();
      for (int i = 0; i < keep; i++) {
        int id = order[i];
        String name = id < names.size() ? names.get(id) : "class-" + id;
        labels.add(new LlmLabelScore(name, logits[id] / normalizer));
      }
      return new LlmOutLabels(labels);
    }
  }
}

