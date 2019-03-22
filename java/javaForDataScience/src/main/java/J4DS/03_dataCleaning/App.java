package j4ds.java.apachecommons;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import static java.lang.System.*;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.text.StrSubstitutor;
import org.apache.commons.lang3.text.StrTokenizer;
import org.apache.commons.validator.EmailValidator;
import org.apache.commons.validator.routines.IntegerValidator;


public class App {
  public static void main(String[] args) {
    String dirtyText = (
      "Call me Ishmael.  Some years ago - never mind how long precisely - "
      + "having little or no money in my purse, and nothing particular to "
      + "interest me on shore, I thought I would sail about a little and "
      + "see the watery part of the world.");

    validateEmailApache(dirtyText);
    //out.println.(validateInt("1234"));
    //out.println(findReplaceApacheCommons(dirtyText, "me", "X"));
  }


  public static void apacheCommonsTokenizer(String text) {
    StrTokenizer tokenizer = new StrTokenizer(text, ",");

    while (tokenizer.hasNext()) { out.println(tokenizer.next()); }
  }


  public static String validateEmailApache(String email) {
    EmailValidator eValidator = EmailValidator.getInstance();
    
    email = email.trim();
    if (eValidator.isValid(email)) {
      return email + " is a valid email address.";
    } else { return email + " is not a valid email address."; }
  }


  public static String validateInt(String text) {
    IntegerValidator intValidator = IntegerValidator.getInstance();

    if (intValidator.isValid(text)) { return text + " is a valid int"; }
    else { return text + " is not a valid int"; }
  }


  public static String findReplaceApacheCommons(
      String text, String toFind, String replaceWith) {
    out.println(text);
    text = StringUtils.replacePattern(text, "\\W\\s", " ");
    out.println(text);
    //out.println(StringUtils.replace(text, " me ", "X"));
    return StringUtils.replace(text, " me ", "X");
  }
}
