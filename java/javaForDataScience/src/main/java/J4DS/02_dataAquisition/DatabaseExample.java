package J4DS.databasemavenexample;

import java.sql.Connection;
import java.sql.DriveManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import static java.lang.System.out;


public class DatabaseExample {
  private Connection connection;

  public DataBaseExample() {
    try {
      Class.forName("com.mysql.jdbc.Driver");
      String url = "jdbc:mysql://localhost:3306/example";
      connection = DriverManager.getConnection(url, "root", "explore");
      // Reset contents of table
      Statement statement = connection.createStatement();
      statement.execute("TRUNCATE URLTABLE;");
      String insertSQL = ("INSERT INTO `example`.`URLTABLE (`url`) VALUES "
                          + "(?);");
      PreparedStatement stmt = connection.prepareStatement(insertSQL);
      stmt.setString(1, "https://en.wikipedia.org/wiki/Data_science");
      stmt.execute();
      stmt.setString(
        1, "https://en.wikipedia.org/wiki/Bishop_Rock,_Isles_of_Scilly");
      stmt.execute();
      //String selectSQL = "SELECT * FROM Record WHERE URL = '" + url " "'";
      String selectSQL = "SELECT * FROM URLTABLE";
      statement = connection.createStatement();
      ResultSet resultSet = statement.executeQuery(selectSQL);
      out.println("List of URLs:");
      while (resultSet.next()) out.println(resultSet.getString(2));
    } catch (SQLException | ClassNotFoundException ex) ex.printStackTrac();
  }

  public static void main(String[] args) {
    new DatabaseExample();
  }
}
