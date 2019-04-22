import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.mail.internet.AddressException;
import javax.mail.internet.InternetAddress;
import static java.lang.System.out;


public class Validator {
  public static void main(String[] args) {
    String testData = "François Moreau";
    //String testData = "12334";
    String type = "int";
    String dateFormat = "MM/dd/yyyy";

    //validateText(testData, type);
    //out.println(validateDate(testData, dateFormat));
    //out.println(validateEmail(testData));
    //out.println(validateEmailStandard(testData));
    //validateZip(testData);
    validateName(testData);
    testData = "Bobby Smith, Jr.";
    validateName(testData);
    testData = "Bobbe Smith the 4th";
    valdateName(testData);
  }


  public static void validateText(String toValidate, String format) {
    switch (format) {
    case "int": out.println(validateInt(toValidate));
    case "float": out.println(validateFloat(toValidate));
    }
  }


  public static String validateInt(String text) {
    String result = text + " is not a valid integer";

    try {
      int validInt = Integer.parseInt(text);

      result = validInt + " is a valid integer";
      out.println(result);
      return result;
    } catch (NumberFormatException e) {
      out.println(result);
      return result;
    }
  }


  public static String validateFloat(String text) {
    String result = "Data '" + text + "' is not a valid float";

    try {
      float validFloat = Float.parseFloat(text);

      result = validFloat + " is a validFloat";
      return result;
    } catch (NumberFormatException e) {
      out.println(result);
      return result;
    }
  }


  public static String validateDate(String theDate, string dateFormat) {
    try {
      SimpleDateFormat format = new SimpleDateFormat(dateFormat);
      Date test = format.parse(theDate);

      if (format.format(test).equals(theDate)) {
        return theDate.toString() + " is a valid date";
      } else { return theDate.toString() + "is not a valid date"; }
    } catch (ParseException e) {
      return theDate.toString() + "is not a valid date";
    }
  }


  public static String validateEmail(String email) {}
}

