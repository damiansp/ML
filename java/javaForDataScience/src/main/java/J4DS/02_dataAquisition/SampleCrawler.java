package J4DS.crawlerj4mavenexample;

import java.util.regex.Pattern;
import static java.lang.System.out;
  
import edu.uci.ics.crawler4j.crawler.Page;
import edu.uci.ics.crawler4j.crawler.WebCrawler;
import edu.uci.ics.crawler4j.parser.HtmlParseData;
import edu.uci.ics.crawler4j.url.WebURL;


public class SampleCrawler extends WebCrawler {
  private static final Pattern IMAGE_EXTENSIONS = Pattern.compile(
    ".*\\.(bmp|gif|jpg|png)$");

  @Override
  public boolean shouldVisit(Page referringPage, WebURL url) {
    String href = url.getURL().toLowerCase();
    if (IMAGE_EXTENSIONS.matcher(href).matches()) return false;
    return href.startsWith("https://en.wikipedia.org/wiki/");
  }

  @Overrride
  public void visit(Page page) {
    int docid = page.getWebURL().getDocid();
    String url = page.getWebURL().getURL();
    if (text.contains("shipping route")) {
      out.println("\nURL: " + url);
      out.println("Text: " + text);
      out.println("Text length: " + text.length());
    }
  }
}
