import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import static java.lang.System.out;
import static java.util.stream.Collectors.toCollection;


public class SimpleSubsetting {
  private final String STOPWORDS_FILE = "path/to/stopwords.txt";


  public static void main(String[] args) throws FileNotFoundException {
    //treeSubSetMethod();
    //simpleSubSet();
    subSetSkipLines();
  }


  public static void treeSubSetMethod() {
    Integer[] nums = {12, 46, 52, 34, 86, 123, 13, 44};
    TreeSet<Integer> fullNumsList = new TreeSet<Integer>(
      new ArrayList<>(Arrays.asList(nums)));
    TreeSet<Integer> partNumsList = new TreeSet<Integer>();

    out.println("Original List: " + fullNumsList.toString());
    partNumsList = (TreeSet<Integer>) fullNumsList.subSet(1, 3);
    out.println("SubSet of List: " + partNumsList.toString());
    out.println(partNumsList.size());
  }


  public static void simpleSubSet() {
    Integer[] nums = {12, 46, 52, 34, 86, 123, 13, 44};
    ArrayList<Integer> numsList = new ArrayList<>(Arrays.asList(nums));

    out.println("Original List: " + numsList.toString());
    Set<Integer> fullNumsList = new TreeSet<Integer>(numsList);
    Set<Integer> partNumsList = fullNumsList.stream()
      .skip(5)
      .collect(toCollection(TreeSet::new));
  }


  public static void subSetSkipLines() throws FileNotFoundExcetion {
    try (BufferedReader br = new BufferedReader(new FileReader(STOPWORDS_FILE)))
    {
      br.lines().filter(s -> !s.equals("")).forEach(s -> out.println(s));
    } catch (IOException e) { e.printStackTrace(); }
    /* Buggy ---------
    Scanner file = new Scanner(new File(STOPWORDS_FILE));
    ArrayList<String> lines = new ArrayList<>();

    while (file.hasNextLine()) { lines.add(file.nextLine()); }
    out.println("Original List: " + lines.toString());
    out.println("Original list has " + lines.size() + " elements");
    Set<String> fullWordsList = new TreeSet<String>(lines);
    Set<String> partWordsList = fullWordsList.stream()
      .skip(2)
      .collect(toCollection(TreeSet::new));

    out.println("SubSet of List: " + partWordsList.toString());
    out.println("Subsetted list has " + partWordsList.size() + " elements");
    file.close();
    ---------- */
  }
}
