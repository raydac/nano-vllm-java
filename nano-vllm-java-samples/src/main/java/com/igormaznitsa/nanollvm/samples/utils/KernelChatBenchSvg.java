package com.igormaznitsa.nanollvm.samples.utils;

import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.Locale;

/**
 * Well-formed SVG 1.1 bar chart for {@code KernelChatBench} tok/s results. JDK-only XML text.
 */
public final class KernelChatBenchSvg {

  private static final int WIDTH = 720;
  private static final int PLOT_LEFT = 200;
  private static final int PLOT_RIGHT = 680;
  private static final int PLOT_TOP = 80;
  private static final int ROW_HEIGHT = 56;
  private static final int BAR_HEIGHT = 28;
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
    requireNonNull(title, "title");
    requireNonNull(subtitle, "subtitle");
    requireNonNull(bars, "bars");
    if (bars.isEmpty()) {
      throw new IllegalArgumentException("bars must not be empty");
    }

    int height = PLOT_TOP + bars.size() * ROW_HEIGHT + BOTTOM_PAD;
    double maxTok = bars.stream().mapToDouble(Bar::tokensPerSecond).max().orElse(1d);
    if (maxTok <= 0d) {
      maxTok = 1d;
    }

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
    svg.append("  <line x1=\"").append(PLOT_LEFT).append("\" y1=\"").append(PLOT_TOP - 8)
      .append("\" x2=\"").append(PLOT_LEFT).append("\" y2=\"")
      .append(PLOT_TOP + bars.size() * ROW_HEIGHT)
      .append("\" stroke=\"#d1d5db\" stroke-width=\"1\"/>\n");

    for (int index = 0; index < bars.size(); index++) {
      appendBar(svg, bars.get(index), index, maxTok);
    }

    svg.append("</svg>\n");
    return svg.toString();
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
    final int index,
    final double maxTok
  ) {
    int plotWidth = PLOT_RIGHT - PLOT_LEFT;
    int barWidth = Math.max(1, (int) Math.round(plotWidth * (bar.tokensPerSecond() / maxTok)));
    int y = PLOT_TOP + index * ROW_HEIGHT;
    int barY = y + (ROW_HEIGHT - BAR_HEIGHT) / 2;
    String value = String.format(
      Locale.ROOT,
      "%.2f tok/s (%.2fx)",
      bar.tokensPerSecond(),
      bar.speedupVersusScalar());

    svg.append("  <text x=\"").append(PLOT_LEFT - 12).append("\" y=\"").append(barY + 20)
      .append(
        "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#111827\">")
      .append(escape(bar.label()))
      .append("</text>\n");
    svg.append("  <rect x=\"").append(PLOT_LEFT).append("\" y=\"").append(barY)
      .append("\" width=\"").append(barWidth).append("\" height=\"").append(BAR_HEIGHT)
      .append("\" rx=\"4\" fill=\"").append(fillFor(bar.mode())).append("\"/>\n");
    svg.append("  <text x=\"").append(PLOT_LEFT + barWidth + 8).append("\" y=\"").append(barY + 20)
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
   * One horizontal bar in comparison order.
   */
  public record Bar(
    String mode,
    String label,
    double tokensPerSecond,
    double speedupVersusScalar
  ) {
    public Bar {
      requireNonNull(mode, "mode");
      requireNonNull(label, "label");
    }
  }
}
