import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import static java.lang.System.out;


public class CSVScanner {
  public static void main(String[] args) {
    try {
      File demoFile = new File("../data/myFile.csv");
      Scanner getData = new Scanner(demoFile);

      while (getData.hasNext()) { out.println(getData.nextLine()); }
    } catch (FileNotFoundException e) { e.printStackTrace(); }
  }
}
