import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;
import static java.lang.System.out;

import com.aliasi.tokenizer.EnglishStopTokenizerFactory;
import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;


public class SimpleStringCleaning {
  private final String STOPWORDS_FILE = "path/to/stopwords.txt";
  
  public static void main(String[] args) {
    String dirtyText = (
      "Call me Ishmael. Some years ago- never mind how long precisely - having "
      + "little or no money in my purse, and nothing particular to interest me "
      + "on shore, I thought I would sail about a little and see the watery "
      + "part of the world.");

    //simpleClean(dirtyText);
    //simpleCleanToArray(dirtyText);
    //cleanAndJoin(dirtyText);
    //removeStopWords(dirtyText);
    //removeStopWordsRemoveAll(dirtyText);
    removeStopWithLing(dirtyText);
  }


  public static String simpleClean(String text) {
    out.println("Raw text: " + text);
    text = text.toLowerCase();
    text = text.replaceAll("[\\d[^\\w\\s]]+", " ");
    text = text.trim();
    while (text.contains("  ")) { text = text.replaceAll("  ", " "); }
    out.println("Cleaned: " + text);
    return text
  }


  public static String[] simpleCleanToArray(String text) {
    out.println("Raw text: " + text);
    text = text.replaceAll("[\\d[^\\w\\s]]+", " ");
    String[] cleanText = text.toLowerCase().trim().split("[\\W\\d]+");

    out.print("Cleaned: ");
    for (String clean: cleanText) { out.print(clean + " "); }
    out.println();
    return cleanText;
  }


  public static String cleanAndJoin(String text) {
    out.println("Raw text: " + text);
    String[] words = text.toLowerCase().trim().split("[\\W\\d]+");
    String cleanText = String.join(" ", words);

    out.println("Cleaned: " + cleanText);
    return cleanText;
  }


  public static void removeStopWords(String text) {
    try {
      Scanner readStop = new Scanner(new File(STOPWORDS_FILE));
      ArrayList<String> words = new ArrayList<String>(
        Arrays.asList(simpleCleanToArray(text)));

      out.println("Original clean text: " + words.toString());
      ArrayList<String> foundWords = new ArrayList();

      while (readStop.hasNextLine()) {
        String stopWord = readStop.nextLine().toLowerCase();

        if (word.contains(stopWord)) { foundWords.add(stopWord); }
      }
      words.removeAll(foundWords);
      out.println("With stop words removed: " + words.toString());
    } catch (FileNotFoundException e) { e.printStackTrace(); }
  }


  public static void removeStopWordsRemoveAll(String text) {
    try {
      out.println(text);
      Scanner stopWordList = new Scanner(new File(STOPWORDS_FILE));
      TreeSet<String> stopWords = new TreeSet<String>();

      while (stopWordList.hasNextLine()) {
        stopWords.add(stopWordList.nextLine());
      }
      ArrayList<String> dirtyText = new ArrayList<String>(
        Arrays.asList(text.split(" ")));

      dirtyText.removeAll(stopWords);
      out.println("Clean words: ");
      for (String x: dirtyText) { out.print(x + " "); }
      out.println();
      stopWordList.close();
    } catch (FileNotFoundException e) { e.printStackTrace(); }
  }


  public static void remove StopWithLing(String text) {
    out.println(text);
    text = text.toLowerCase().trim();
    TokenizerFactory fact = IndoEuropeanTokenizerFactory.INSTANCE;

    fact = new EnglishStopTokenizerFactory(fact);
    Tokenizer tok = fact.tokenizer(text.toCharArray(), 0, text.length());
    for (String word: tok) { out.print(word + " "); }
  }
}
