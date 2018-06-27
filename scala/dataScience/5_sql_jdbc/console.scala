// To be run in the console
import java.sql._

Class.forName("com.mysql.jdbc.Driver")

val connection = DriverManager.getConnection(
	"jdbc:mysql://127.0.0.1:3306/test", "username", "password")

// Create table
val statementString = """
CREATE TABLE pysicists (
	id INT(11) AUTO_INCREMENT PRIMARY KEY,
	name VARCHAR(32) NOT NULL);"""

// Insert
val statement = connection.prepareStatement("""
	INSERT INTO physicists (name) VALUES ('Isaac Newton')""")

val physicistNames = List("Marie Curie", "Albert Einstein", "Paul Dirac")
val statement2 = connection.prepareStatement("""
	INSERT INTO physicists (name) VALUES (?)""")
statement2.setString(1, "Richard Feynman")
statement2.addBatch()
physicistNames.foreach { name => 
	statement2.setString(1 name)
	statement2.addBatch()
}
statement2.executeBatch

