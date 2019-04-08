import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import static java.lang.System.*;


public class SimpleSearch {
  public static void main(String[] args) {
    String toFind = "I";
    String replaceWith = "Ishmael";
    String dirtyText = (
      "Call me Ishmael. Some years ago- never mind how long precisely - having "
      + "little or no money in my purse,  and nothing particular to interest "
      + "me on shore, I thought I would sail about a little and see the watery "
      + "part of the world.");

    simpleFindReplace(dirtyText, toFind, replaceWith);
    try {
      Scanner textToClean = new Scanner(new File("path/to/file.txt"));

      while (textToClean.hasNext()) {
        //String dirtyText = textToClean.nextLine();
        //simpleSearch(dirtyText, toFind);
        //scannerSearch(dirtyText, toFind);
        //simpleFindReplace(dirtyText, toFind, replaceWith);
        ;
      }
      textToClean.close();
    } catch (FileNotFoundException e) { e.printStackTrace(); }
  }


  public static void simpleSearch(String text, String toFind) {
    int count = 0;

    text = text.toLowerCase().trim();
    toFind = toFind.toLowerCase().trim();
    if (text.contains(toFind)) {
      String[] words = text.split(" ");

      for (String word: words) {
        if (word.equals(toFind)) { count++; }
      }
      out.println("Found %s %d times in the text", toFind, count);
    }
  }


  public static void scannerSearch(String text, String toFind) {
    text = text.toLowerCase().trim();
    toFind = toFind.toLowerCase().trim();
    Scanner textLine = new Scanner(text);

    out.println("Found " + tetLine.findWithinHorizon(toFind, 10));
  }


  public static void simpleFindReplace(
      String text, String toFind, String replaceWith) {
    text = text.toLowerCase().trim();
    toFind = toFind.toLowerCase().trim();
    out.println(text);
    if (text.contains(toFind)) {
      text = text.replaceAll(toFind, replaceWith);
      out.println(text);
      //for (String word: textLine) { out.print(word + " "); }
    }
  }


  public static void searchWholeFile(String path, String toFind) {
    try {
      int line = 0;
      String textLine = "";
      BufferedReader textToClean = new BufferedReader(new FileReader(path));

      toFind = toFind.toLowerCase().trim();
      while ((textLine = textToCLean.readLine()) != null) {
        line++;
        if (textLine.toLowerCase().trim().contains(toFind)) {
          out.println("Found %s in %s", toFind, textLine);
          //out.println("Found %s on line %d", toFind, line);
          //String[] words = textLine.split(" ");
          //for (int i = 0; i < words.length; i++) {
          //  if(words[i].equals(toFind)) {
          //    out.println(
          //      "On line %d found %s at location %d", line, toFind, (i - 1));
          //}
        }
      }
      textToClean.close();
    } catch (FileNotFoundExcetion e) { e.printStackTrace(); }
    catch (IOException e) { e.printStackTrace(); }
  }
}
