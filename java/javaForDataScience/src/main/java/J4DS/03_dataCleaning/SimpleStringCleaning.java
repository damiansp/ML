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


  public static void removeStopWords(String text) {}
}
