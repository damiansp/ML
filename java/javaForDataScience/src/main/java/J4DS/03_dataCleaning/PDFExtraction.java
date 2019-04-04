package j4ds.pdfextraction;

import java.io.File;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;


public class PDFExtraction {
  public static void main(String[] args) {
    try {
      PDDocument doc = PDDocument.load(new File("PDFFile.pdf"));
      PDFTextStripper tStripper = new PDFTextStripper();
      String docText = tStripper.getText(doc);
      System.out.println(docText);
    } catch (Exception e) { e.printStackTrace(); }
  }
}
