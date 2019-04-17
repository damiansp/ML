import java.util.StringTokenizer;
import static java.lang.System.out;


public class Tokenizer {
  public static void main(String[] args) {
    String rawText = (
      "Call me Ishmael. Some years ago- never mind how long precisely - "
      + "having little or no money in my purse, and nothing particular to "
      + "interest me on shore, I thought I would sail about a little and see "
      + "the watery part of the world.");
    StringTokenizer tokenizer = new StringTokenizer(rawText, " ");

    while (tokenizer.hasMoreTokens()) {
      out.print(tokenizer.nextToken() + "\t");
    }
  }
}
