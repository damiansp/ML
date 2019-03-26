import java.util.ArrayList;
import java.util.Optional;
import static java.lang.System.out;


public class DataImputation {
  public static void main(String[] args) { tempExample(); }


  public static void tempExample() {
    double[] tempList = new double[365];

    for (int i = 0; i < tempList.length; i++) {
      tempList[i] = Math.random() * 100;
    }
    tempList[5] = 0;
    double sum = 0;

    for (double d: tempList) {
      // out.println(d);
      sum += d;
    }
    out.println(sum / 365);
    String useName = "";
    String[] nameList = {
      "Amy", "Bob", "Sally", "Sue", "Don", "Rick", null, "Betsy"};
    Optional<String> tempName;

    for (String name: nameList) {
      tempName = Optional.ofNullable(name);
      useName = tempName.orElse("Default");
      out.println("Name to use: " + useName);
    }
  }
}
