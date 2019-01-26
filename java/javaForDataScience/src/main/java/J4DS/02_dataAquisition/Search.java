package j4ds;

import java.io.IOException;
import java.util.List;
import static java.lang.System.out;

import com.google.api.client.googleapis.json.GoogleJsonResponseException;
import com.google.api.client.http.HttpRequest;
import com.google.api.client.http.HttpRequestInitializer;
import com.google.api.services.youtube.YouTube;
import com.google.api.services.youtube.model.ResourceId;
import com.google.api.services.youtube.model.SearchListResponse;
import com.google.api.services.youtube.model.SearchResult;
import com.google.api.services.youtube.model.Thumbnail;


// Adapted from
// https://developers.google.com/youtube/v3/code_samples/java#search_by_keyword
public class Search {
  public static void main(String[] args) {
    try {
      YouTube youtube = new YouTube.Builder(
        Auth.HTTP_TRANSPORT,
        Auth.JSON_FACTORY,
        new HttpRequestInitializer() {
          public void initialize(HttpRequest request) throws IOException {}
        })
        .setApplicationName("appName")
        .build();
      String queryTerm = "cats";
      YouTube.Search.List search = youtube.search().list("id,snippet");

      search.setType("video"); // "channel", "playlist" also allowed
      search.setFields("items(id/kind,id/videoId,snippet/title,"
                       + "snippet/description,snippet/thumbnails/default/url)");
      search.setMaxResults(10L);

      SearchListResponse searchResponse = search.execute();
      List<SearcResults> searchResultList = searchResponse.getItems();
      SearchResult video = searchResultList.iterator().next();
      Thumbnail thumbnail = video.getSnippet().getThumbnails().getDefault();

      out.println("Kind: " + video.getKind());
      out.println("Video ID: " + video.getId().getVideoId());
      out.println("Title: " + video.getSnippet().getTitle());
      out.println("Description: " + video.getSnippet().getDescription());
      out.println("Thumbnail: " + thumbnail.getUrl());
    } catch (GoogleJsonResponseException e) { e.printStackTrace(); }
    catch (IOException e) { e.printStackTrace(); }
  }
}
