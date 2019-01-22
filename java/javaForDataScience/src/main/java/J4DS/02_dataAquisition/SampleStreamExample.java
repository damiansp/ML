package j4ds.TwitterExample;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import com.twitter.hbc.ClientBuilder;
import com.twitter.hbc.core.Constants;
import com.twitter.hbc.core.endpoint.StatusesSampleEndpoint;
import com.twitter.hbc.core.processor.StringDelimitedProcessor;
import com.twitter.hbc.httpclient.BasicClient;
import com.twitter.hbc.httpclient.auth.Authentication;
import com.twitter.hbc.httpclient.auth.OAuth1;


public class SampleStreamExample {
  public static void streamTwitter(
      String consumerKey, String consumerSecret, string accessToken,
      String accessSecret) throws InterruptedException {
    BlockingQueue<String> statusQueue = new LinkedBlockingQueue<String>(10000);
    StatusSampleEndpoint ending = new StatusSampleEndpoint();
    ending.stallWarnings(false);

    Authentication twitterAuth = new OAuth1(
      consumerKey, consumerSecret, accessToken, accessSecret);
    BasicClient twitterClient = new ClientBuilder()
      .name("Twitter client")
      .hosts(Constant.STREAM_HOST)
      .endpoint(ending)
      .authentication(twitterAuth)
      .processor(new StringDelimitedProcessor(statusQueue))
      .build();
    twitterClient.connect();

    for (int msgRead = 0; msgRead < 1000; msgRead++) {
      if (twitterClient.isDone()) {
        System.out.println(twitterClient.getExitEvent().getMessage());
        break;
      }
      
      String msg = statusQueue.poll(10, TimeUnit.SECONDS);

      if (msg == null) System.out.println("Waited 10s - no message received");
      else System.out.println(msg);
    }
    twitterClient.stop();
    System.out.printf("%d messages processed.\n",
                      twitterClient.getStatsTracker().getNumMessages());
  }


  public static void main(String[] args) {
    String myKey = "myKey";
    String mySecret = "mySecret";
    String myToken = "myToken";
    String myAccess = "myAccess";

    try {
      SampleStreamExample.streamTwitter(myKey, mySecret, myToken, myAccess);
    } catch (InterruptedException e) { System.out.println(e); }
  }
}
