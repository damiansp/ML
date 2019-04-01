package j4ds.jsonexamples;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import static java.lang.System.out;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeType;


public class JSONExamples {
  public JSONExamples() {
    // parsePerson();
    //parsePeople("People.json");
    treeTraversalSolution();
  }


  public void treeTraversalSolution() {
    try {
      // Use mapper to read json string and create tree
      ObjectMapper mapper = new ObjectMapper();
      JsonNode node = mapper.readTree(new File("People.json"));
      Iterator<String> fieldNames = node.fieldNames();

      while (fieldNames.hasNext()) {
        JsonNode peopleNode = nod.get("people");
        Iterator<JsonNode> elements = peopleNode.iterator();

        while (elements.hasNext()) {
          JsonNode element = elements.next();
          JsonNodeType nodeType = element.getNodeType();

          if (nodeType == JsonNodeType.STRING) {
            out.println("Group: " + element.textValue());
          }
          if (nodeType == JsonNodeType.ARRAY) {
            Iterator<JsonNode> fields = element.iterator();

            while (fields.hasNext()) { parsePerson(fields.next()); }
          }
        }
        fieldNames.nxt();
      }
    } catch (IOException e) { e.printStackTrace(); }
  }


  public void parsePerson(JsonNode node) {
    Iterator<JsonNode> fields = node.iterator();

    while (fields.hasNext()) {
      JsonNode subNode = fields.next();

      out.println(subNode.asText());
    }
  }


  public static void parsePerson() {
    try {
      JsonFactory jsonFactory = new JsonFactory();
      JsonParser parser = jsonFactory.createParser(new File("Person.json"));

      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String token = parser.getCurrentName();

        if ("firstname".equals(token)) {
          parser.nextToken();
          String fname = parser.getText();

          out.println("firstname: " + fname);
        }
        if ("lastname".equals(token)) {
          parser.nextToken();
          String lname = parser.getText();

          out.println("lastname: " + lname);
        }
        if ("phone".equals(token)) {
          parser.nextToken();
          long phone = parser.getLongValue();

          out.println("phone: " + phone);
        }
        if ("address".equals(token)) {
          out.println("address: ");
          parser.nextToken();
          while (parser.nextToken != JsonToken.END_ARRAY) {
            our.println(parser.getText());
          }
        }
      }
      parser.close();
    } catch (IOException e) { e.printStackTrace(); }
  }


  public void parsePeople(String filename) {
    try {
      JsonFactory jsonFactory = new JsonFactory();
      File source = new File(filename);
      JsonParser parser = jsonFactory.createParser(source);

      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String token = parser.getCurrentName();

        if ("people".equals(token)) {
          out.println("People found");
          JsonToken jsonToken = parser.nextToken();

          jsonToken = parser.nextToken();
          token = parser.getCurrentName();
          out.println("Next token is " + token);
          if ("groupname".equals(token)) {
            parser.nextToken();
            String groupName = parser.getText();

            out.println("Group: " + groupName);
            parser.nextToken();
            token = parser.getCurrentName();
            if ("person".equals(token)) {
              out.println("Found person");
              parsePerson(parser);
            }
          }
        }
      }
      parser.close();
    } catch (IOException e) { e.printStackTrace(); }
  }


  public void parsePerson(JsonParser parser) throws IOException {
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      String token = parser.getCurrentName();

      if ("firstname".equals(token)) {
        parser.nextToken();
        String fname = parser.getText();

        out.println("firstname: " + fname);
      }
      if ("lastname".equals(token)) {
        parser.nextToken();
        String lname = parser.getText();

        out.println("lastname : " + lname);
      }
      if ("phone".equals(token)) {
        parser.nextToken();
        long phone = parser.getLongValue();

        out.println("phone : " + phone);
      }
      if ("address".equals(token)) {
        out.println("address :");
        parser.nextToken();
        while (parser.nextToken() != JsonToken.END_ARRAY) {
          out.println(parser.getText());
        }
      }
    }
  }


  public static void main(String[] args) { new JSONExamples(); }
}
