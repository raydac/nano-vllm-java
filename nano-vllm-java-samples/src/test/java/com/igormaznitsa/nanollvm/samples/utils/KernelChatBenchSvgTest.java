package com.igormaznitsa.nanollvm.samples.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.StringReader;
import java.util.List;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.jupiter.api.Test;
import org.w3c.dom.Element;
import org.xml.sax.InputSource;

class KernelChatBenchSvgTest {

  @Test
  void renderProducesNamespacedSvgWithEscapedText() throws Exception {
    List<KernelChatBenchSvg.Bar> bars = List.of(
      new KernelChatBenchSvg.Bar("scalar", "Scalar Java", 6.4d, 1.0d),
      new KernelChatBenchSvg.Bar("vector", "Vector API (SIMD)", 16.0d, 2.5d));
    String svg = KernelChatBenchSvg.render(
      "Kernel chat throughput",
      "Gemma3-270M & <bench> · maxTokens=64",
      bars);

    assertTrue(svg.startsWith("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assertTrue(svg.contains("xmlns=\"http://www.w3.org/2000/svg\""));
    assertTrue(svg.contains("Gemma3-270M &amp; &lt;bench&gt;"));
    assertFalse(svg.contains("Gemma3-270M & <bench>"));

    DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
    factory.setNamespaceAware(true);
    factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
    Element root = factory.newDocumentBuilder()
      .parse(new InputSource(new StringReader(svg)))
      .getDocumentElement();
    assertEquals("svg", root.getLocalName());
    assertEquals("http://www.w3.org/2000/svg", root.getNamespaceURI());
    assertEquals("1.1", root.getAttribute("version"));
  }

  @Test
  void renderRejectsEmptyBars() {
    assertThrows(
      IllegalArgumentException.class,
      () -> KernelChatBenchSvg.render("title", "sub", List.of()));
  }

  @Test
  void escapeCoversXmlSpecials() {
    assertEquals("&amp;&lt;&gt;&quot;&apos;", KernelChatBenchSvg.escape("&<>\"'"));
  }
}
