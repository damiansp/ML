import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import static java.lang.System.out.println;


public class SimpleSort {
  public static void main(String[] args) {
    basicSort();
    complexSort();
  }


  public static void basicSort() {
    String[] words = {"cat", "dog", "house", "boat", "zoo", "road"};
    ArrayList<String> wordList = new ArrayList<>(Arrays.asList(words));
    Integer[] nums = {12, 24, 34, 54, 39, 9, 38, 20, 54, 93, 8, 73, 48, 92};
    ArrayList<Integer> numList = new ArrayList<>(Arrays.asList(nums));

    println("Original Word List: " + wordList.toString());
    Collections.sort(wordList);
    println("Ascending Word List: " + wordList.toString());
    println("Original Number List: " + numList.toString());
    Collections.reverse(numList);
    println("Reversed Number List: " + numList.toString());
    Collections.sort(numList);
    println("Ascending Number List: " + numList.toString());
    Comparator<Integer> basicOrder = Integer::compare;
    Comparator<Integer> descendOrder = basicOrder.reversed();

    Collections.sort(numList, descendOrder);
    println("Descending Number List: " + numList.toString());
    Comparator<Integer> compareInts = (Integer first, Integer second) ->
      Integer.comapre(first, second);

    Collections.sort(numList, compareInts);
    println("Sorted Numbers using Lambda: " + numList.toString());
    Comparator<String> basicWords = String::compareTo;
    Comparator<String> descendWords = basicWords.reversed();

    Collections.sort(wordList, descendWords);
    println("Reversed Words using Comparator: " + wordList.toString());
    Comparator<String> compareWords = (String first, String second) ->
      first.compareTo(second);

    Collections.sort(wordList, compareWords);
    println("Sorted Words using Lambda: " + wordList.toString());
  }


  public static void complexSort() {
    ArrayList<Dogs> dogs = new ArrayList<Dogs>();

    println();
    dogs.add(new Dogs("Zoey", 8));
    dogs.add(new Dogs("Roxy", 10));
    dogs.add(new Dogs("Kyla", 7));
    dogs.add(new Dogs("Shorty", 12));
    dogs.add(new Dogs("Uppity", 7));
    dogs.add(new Dogs("Penny", 4));
    println("Name " + " Age");
    for (Dogs d: dogs) { println(d.getName() + " " + d.getAge()); }
    println();
    dogs.sort(Comparator.comparing(Dogs::getName).thenComparing(Dogs::getAge));
    println("Name " + " Age");
    for (Dogs d: dogs) { println(d.getName() + " " + d.getAge()); }
    println();
  }
}
