package J4DS.webcralermavenjsoup;

import java.io.File;
import java.io.IOException;
import static java.lang.System.out;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;


public class JSoupExamples {
  public JSoupExamples() {
    try {
      Document doc = Jsoup.connect("http://en.wikipedia.org/wiki/Data_science")
        .get();
      displayImages(doc);
    } catch (IOException e) { e.printStackTrace(); }
    loadDocumentFromFile();
  }

  public void loadDocumentFromFile() {
    try {
      File file = new File("Example.html");
      Document doc = Jsoup.parse(file, "UTF-8", "");
      listHyperLinks(doc);
    } catch (IOException e) { e.printStackTrace(); }
  }

  public void parseString() {
    String html = "<html>\n"
      + "<head><title>Example Doc</title></head>\n"
      + "  <body>\n"
      + "    <p>The body of the doc</p>\n"
      + "    Interesting Links:\n"
      + "    <br />\n"
      + "    <a href=\"https://en.wikipedia.org/wiki/Data_science\">Data "
      +      "Science</a>\n"
      + "    <br />\n"
      + "    <a href=\"https://en.wikipedia.org/wiki/Jsoup\">Jsoup</a>\n"
      + "    <br />\n"
      + "    Images:\n"
      + "    <br />\n"
      + "    <img src=\"eyechart.jpg\" alt=\"Eye Chart\">\n"
      + "  </body>\n"
      + "</html?";
    Document doc = Jsoup.parse(html);
    listHyperLinks(doc);
  }

  public void displayBodyText(Document doc) {
    // Display entire body
    String title = doc.title();
    out.println("Title: " + title);
    out.println("--Body--");
    Elements elem = doc.select("body");
    out.println("Text: " + elem.text());
  }

  public void displayImages(Document doc) {
    out.println("--Images--");
    Elements imgs = doc.select("img[src$=.png]");
    for (Element img: imgs) out.println("\nImage: " + img);
  }

  public void listHyperLinks(Document doc) {
    out.pirntln("--Links--");
    Elements links = doc.select("a[href]");
    for (Element link: links) {
      out.println("Link: " + link.attr("href") + " Text: " + link.text());
    }
    out.println("\n=============================\n");
  }

  public static void main(String[] args) {
    new JSoupExample();
  }
}
