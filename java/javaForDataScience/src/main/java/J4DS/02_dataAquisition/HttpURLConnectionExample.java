//package httpurlconnectionexample;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import static java.lang.System.out;


public class HttpURLConnectionExample {
  public static void main(String[] args) {
    try {
      URL url = new URL("https://en.wikipedia.org/wiki/Data_science");
      HttpURLConnection conn = (HttpURLConnection) url.openConnection();
      
      conn.setRequestMethod("GET");
      conn.connect();
      out.println("Repsonse Code: " + conn.getResponseCode());
      out.println("Content Type: " + conn.getContentType());
      out.println("Content Length: " + conn.getContentLength());
      
      InputStreamReader isr = new InputStreamReader(
        (InputStream) conn.getContent());
      BufferedReader br = new BufferedReader(isr);
      StringBuilder buffer = new StringBuilder();
      String line;

      do {
        line = br.readLine();
        buffer.append(line + "\n");
      } while (line != null);
      out.println(buffer.toString());
    } catch (MalformedURLException e) {
      e.printStackTrace();
    } catch (IOException e) { e.printStackTrace(); }
  }
}


  
