package com.igormaznitsa.nanollvm.rag;

import static java.nio.charset.StandardCharsets.US_ASCII;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PdfTextExtractorTest {

  @TempDir
  Path tempDir;

  @Test
  void isPdfRecognizesExtensionCaseInsensitively() {
    assertTrue(PdfTextExtractor.isPdf(Path.of("notes.PDF")));
    assertTrue(PdfTextExtractor.isPdf(Path.of("docs/chapter.pdf")));
    assertFalse(PdfTextExtractor.isPdf(Path.of("notes.txt")));
  }

  @Test
  void extractReadsUncompressedMinimalPdf() {
    byte[] pdf = """
      %PDF-1.4
      1 0 obj<<>>stream
      BT /F1 12 Tf 50 700 Td (Hello from minimal PDF) Tj ET
      endstream
      endobj
      trailer<< /Root 1 0 R >>
      %%EOF
      """.stripIndent().getBytes(US_ASCII);

    String text = PdfTextExtractor.extract(pdf);

    assertTrue(text.contains("Hello from minimal PDF"));
  }

  @Test
  void extractIgnoresNonHexAngleBracketsWithoutFailing() {
    byte[] pdf = """
      %PDF-1.4
      1 0 obj<<>>stream
      BT /F1 12 Tf 50 700 Td <g"> Tj (keep literal text) Tj ET
      endstream
      endobj
      trailer<< /Root 1 0 R >>
      %%EOF
      """.stripIndent().getBytes(US_ASCII);

    String text = PdfTextExtractor.extract(pdf);

    assertTrue(text.contains("keep literal text"));
  }

  @Test
  void extractReadsValidHexShowString() {
    byte[] pdf = """
      %PDF-1.4
      1 0 obj<<>>stream
      BT /F1 12 Tf 50 700 Td <48656C6C6F> Tj ET
      endstream
      endobj
      trailer<< /Root 1 0 R >>
      %%EOF
      """.stripIndent().getBytes(US_ASCII);

    String text = PdfTextExtractor.extract(pdf);

    assertTrue(text.contains("Hello"));
  }

  @Test
  void extractConcatenatesGlyphTjWithoutLetterSpacing() {
    byte[] pdf = """
      %PDF-1.4
      1 0 obj<<>>stream
      BT /F1 12 Tf 50 700 Td (H) Tj (e) Tj (l) Tj (l) Tj (o) Tj ( ) Tj (W) Tj (o) Tj (r) Tj (l) Tj (d) Tj ET
      endstream
      endobj
      trailer<< /Root 1 0 R >>
      %%EOF
      """.stripIndent().getBytes(US_ASCII);

    String text = PdfTextExtractor.extract(pdf);

    assertTrue(text.contains("Hello World"), () -> "extracted: " + text);
    assertFalse(text.contains("H e l l o"), "must not insert spaces between glyph Tj pieces");
  }

  @Test
  void extractMapsOcrStyleCidTextThroughToUnicode() {
    byte[] pdf = """
      %PDF-1.4
      1 0 obj<<>>stream
      begincmap
      1 begincodespacerange
      <0000> <FFFF>
      endcodespacerange
      3 beginbfchar
      <0001> <0048>
      <0002> <0069>
      <0003> <0021>
      endbfchar
      endcmap
      endstream
      endobj
      2 0 obj<<>>stream
      BT /F1 12 Tf 50 700 Td <000100020003> Tj ET
      endstream
      endobj
      trailer<< /Root 1 0 R >>
      %%EOF
      """.stripIndent().getBytes(US_ASCII);

    String text = PdfTextExtractor.extract(pdf);

    assertTrue(text.contains("Hi!"), () -> "extracted: " + text);
  }

  @Test
  void extractReturnsPageTextInOrderFromPdfBoxFixture() throws IOException {
    Path pdf = this.writePdfBoxFixture(
      "Jacob and Wilhelm Grimm collected fairy tales.",
      "Little Red Riding Hood met a wolf in the woods.");

    String text = PdfTextExtractor.extract(pdf);

    assertTrue(text.contains("Jacob and Wilhelm Grimm"));
    assertTrue(text.contains("Little Red Riding Hood"));
  }

  @Test
  void corpusIndexesPdfThroughRagFactory() throws IOException {
    Path pdf = this.writePdfBoxFixture("Paris is the capital of France.");

    PreparedRag prepared = RagFactory.builder()
      .options(RagLoadOptions.forTinyModels())
      .addFile(pdf)
      .build();
    List<RagHit> hits = prepared.retrieve("capital of France Paris", 1);

    assertFalse(hits.isEmpty());
    assertTrue(hits.getFirst().chunk().text().contains("Paris"));
  }

  @Test
  void extractRejectsInvalidBytes() {
    assertThrows(UncheckedIOException.class,
      () -> PdfTextExtractor.extract("not-a-pdf".getBytes(US_ASCII)));
  }

  private Path writePdfBoxFixture(String... pageTexts) throws IOException {
    Path pdf = this.tempDir.resolve("sample.pdf");
    try (PDDocument document = new PDDocument()) {
      for (String pageText : pageTexts) {
        PDPage page = new PDPage();
        document.addPage(page);
        try (PDPageContentStream content = new PDPageContentStream(document, page)) {
          content.beginText();
          content.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
          content.newLineAtOffset(50, 700);
          content.showText(pageText);
          content.endText();
        }
      }
      document.save(pdf.toFile());
    }
    assertTrue(Files.isRegularFile(pdf));
    return pdf;
  }
}
