package J4DS.crawlerj4mavenexample;

import edu.uci.ics.crawler4j.crawler.CrawlConfig;
import edu.uci.ics.crawler4j.crawler.CrawlController;
import edu.uci.ics.crawler4j.fetcher.PageFetcher;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtConfig;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtServer;


public class CrawlerController {
  public static void main(String[] args) throws Exception {
    int nCrawlers = 2;
    CrawlConfig config = new CrawlConfig();
    String dataDir = "data";
    PageFetcher fetcher = configure(config, dataDir);
    RobotstxtConfig robotsTxtConfig = new RobotstxtConfig();
    RobotstxtServer robotsTxtServer = new RobotstxtServer(robotsTxtConfig,
                                                          fetcher);
    CrawlController controller = new CrawlController(
      config, fetcher, robotsTxtServer);

    controller.addSeed(
      "https://en.wikipedia.org/wiki/Bishop_Rock,_Isles_of_Scilly");
    controller.start(SampleCrawler.class, nCrawlers);
  }

  private static PageFetcher configure(CrawlConfig config, String dataDir) {
    config.setCrawlStorageFolder(dataDir);
    config.setPolitenessDelay(500);
    config.setMaxDepthOfCrawling(2);
    config.setMaxPagesToFetch(20);
    config.setIncludeBinaryContentInCrawling(false);
    PageFetcher fetcher = new PageFetcher(config);
    return fetcher;
  }
}
