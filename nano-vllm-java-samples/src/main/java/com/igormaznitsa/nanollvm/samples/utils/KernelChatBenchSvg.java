package com.igormaznitsa.nanollvm.samples.utils;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.Locale;

/**
 * Well-formed SVG 1.1 bar chart for {@code KernelChatBench} tok/s results. JDK-only XML text.
 */
public final class KernelChatBenchSvg {

  private static final int WIDTH = 980;
  private static final int PLOT_LEFT = 200;
  private static final int PLOT_RIGHT = 620;
  private static final int PLOT_TOP = 80;
  private static final int ROW_HEIGHT = 56;
  private static final int BAR_HEIGHT = 28;
  private static final int SECTION_GAP = 36;
  private static final int BOTTOM_PAD = 32;

  private KernelChatBenchSvg() {
  }

  /**
   * SVG 1.1 document (XML declaration + namespaced root) for {@code bars}.
   *
   * @param title    chart title
   * @param subtitle model / budget line
   * @param bars     one bar per kernel; must not be empty
   * @return UTF-8 XML
   */
  public static String render(final String title, final String subtitle, final List<Bar> bars) {
    return render(title, subtitle, "tok/s", bars);
  }

  /**
   * SVG 1.1 document. {@code unit} is the label after the bar value, such as {@code tok/s} or
   * {@code img/s}.
   *
   * @param title    chart title
   * @param subtitle model / budget line
   * @param unit     rate unit printed beside each bar
   * @param bars     one bar per kernel; must not be empty
   * @return UTF-8 XML
   */
  public static String render(
    final String title, final String subtitle, final String unit, final List<Bar> bars
  ) {
    return renderSections(title, subtitle, List.of(new Section("", unit, bars)));
  }

  /**
   * Two or more charts in one document. Each section has its own scale and unit, so chat tok/s
   * and YOLO images/s are not drawn on the same axis.
   *
   * @param title    document title
   * @param subtitle shared subtitle
   * @param sections one heading, unit, and bar list each; must not be empty
   * @return UTF-8 XML
   */
  public static String renderSections(
    final String title, final String subtitle, final List<Section> sections
  ) {
    requireNonNull(title, "title");
    requireNonNull(subtitle, "subtitle");
    requireNonNull(sections, "sections");
    if (sections.isEmpty() || sections.stream().anyMatch(section -> section.bars().isEmpty())) {
      throw new IllegalArgumentException("each section needs at least one bar");
    }

    int body = sections.stream()
      .mapToInt(section -> sectionHeading(section) + section.bars().size() * ROW_HEIGHT)
      .sum();
    int height = PLOT_TOP + body + BOTTOM_PAD;
    StringBuilder svg = new StringBuilder();
    svg.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.append("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"")
      .append(WIDTH)
      .append("\" height=\"")
      .append(height)
      .append("\" viewBox=\"0 0 ")
      .append(WIDTH)
      .append(' ')
      .append(height)
      .append("\">\n");
    svg.append("  <title>").append(escape(title)).append("</title>\n");
    svg.append("  <desc>").append(escape(subtitle)).append("</desc>\n");
    svg.append("  <rect width=\"").append(WIDTH).append("\" height=\"").append(height)
      .append("\" fill=\"#ffffff\"/>\n");
    svg.append(
        "  <text x=\"24\" y=\"32\" font-family=\"sans-serif\" font-size=\"20\" fill=\"#111827\">")
      .append(escape(title))
      .append("</text>\n");
    svg.append(
        "  <text x=\"24\" y=\"54\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#4b5563\">")
      .append(escape(subtitle))
      .append("</text>\n");

    int cursor = PLOT_TOP;
    for (Section section : sections) {
      if (!section.heading().isBlank()) {
        svg.append(
            "  <text x=\"24\" y=\"").append(cursor + 18)
          .append(
            "\" font-family=\"sans-serif\" font-size=\"16\" fill=\"#111827\">")
          .append(escape(section.heading()))
          .append("</text>\n");
        cursor += SECTION_GAP;
      }
      double maxRate = section.bars().stream().mapToDouble(Bar::tokensPerSecond).max().orElse(1d);
      if (maxRate <= 0d) {
        maxRate = 1d;
      }
      svg.append("  <line x1=\"").append(PLOT_LEFT).append("\" y1=\"").append(cursor - 8)
        .append("\" x2=\"").append(PLOT_LEFT).append("\" y2=\"")
        .append(cursor + section.bars().size() * ROW_HEIGHT)
        .append("\" stroke=\"#d1d5db\" stroke-width=\"1\"/>\n");
      for (int index = 0; index < section.bars().size(); index++) {
        appendBar(
          svg, section.bars().get(index), cursor + index * ROW_HEIGHT, maxRate, section.unit());
      }
      cursor += section.bars().size() * ROW_HEIGHT;
    }

    svg.append("</svg>\n");
    return svg.toString();
  }

  private static int sectionHeading(final Section section) {
    return section.heading().isBlank() ? 0 : SECTION_GAP;
  }

  static String escape(final String text) {
    requireNonNull(text, "text");
    return text.replace("&", "&amp;")
      .replace("<", "&lt;")
      .replace(">", "&gt;")
      .replace("\"", "&quot;")
      .replace("'", "&apos;");
  }

  private static void appendBar(
    final StringBuilder svg,
    final Bar bar,
    final int rowTop,
    final double maxRate,
    final String unit
  ) {
    int plotWidth = PLOT_RIGHT - PLOT_LEFT;
    int barWidth = Math.max(1, (int) Math.round(plotWidth * (bar.tokensPerSecond() / maxRate)));
    int barY = rowTop + (ROW_HEIGHT - BAR_HEIGHT) / 2;
    String value = bar.caption().isBlank()
      ? String.format(
      Locale.ROOT, "%.2f %s (%.2fx)", bar.tokensPerSecond(), unit, bar.speedupVersusScalar())
      : bar.caption();

    svg.append("  <text x=\"").append(PLOT_LEFT - 12).append("\" y=\"").append(barY + 20)
      .append(
        "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#111827\">")
      .append(escape(bar.label()))
      .append("</text>\n");
    svg.append("  <rect x=\"").append(PLOT_LEFT).append("\" y=\"").append(barY)
      .append("\" width=\"").append(barWidth).append("\" height=\"").append(BAR_HEIGHT)
      .append("\" rx=\"4\" fill=\"").append(fillFor(bar.mode())).append("\"/>\n");
    svg.append("  <text x=\"").append(PLOT_RIGHT + 12).append("\" y=\"").append(barY + 20)
      .append("\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#1f2937\">")
      .append(escape(value))
      .append("</text>\n");
  }

  private static String fillFor(final String mode) {
    return switch (mode) {
      case "scalar" -> "#6b7280";
      case "vector" -> "#2563eb";
      case "tornado" -> "#d97706";
      default -> "#4b5563";
    };
  }

  /**
   * One horizontal bar. {@code caption} replaces the default rate text when it is not blank.
   * A longer bar is a higher rate. The number is drawn in a fixed column past the plot.
   */
  public record Bar(
    String mode,
    String label,
    double tokensPerSecond,
    double speedupVersusScalar,
    String caption
  ) {
    public Bar(
      final String mode, final String label, final double tokensPerSecond,
      final double speedupVersusScalar
    ) {
      this(mode, label, tokensPerSecond, speedupVersusScalar, "");
    }

    public Bar {
      requireNonNull(mode, "mode");
      requireNonNull(label, "label");
      caption = caption == null ? "" : caption;
    }
  }

  /**
   * One independently scaled group of bars.
   *
   * @param heading blank for a chart that already has a document title
   * @param unit    rate unit used when a bar has no caption
   * @param bars    must not be empty
   */
  public record Section(String heading, String unit, List<Bar> bars) {
    public Section {
      requireNonNull(heading, "heading");
      requireNonNull(unit, "unit");
      requireNonNull(bars, "bars");
    }
  }
}
