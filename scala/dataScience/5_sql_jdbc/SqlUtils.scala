import java.sql._

object SqlUtils {
  /** Create an auto-closing connection using loan pattern */
  def useConnection[T](
      db:String, host: String="127.0.0.1", user: String="root", password: String="", port: Int=3306)
      (f: Connection => T): T = {
    // Create connection
    val Url = s"jdbc:mysql://$host:$port/$db"
    Class.forName("com.mysql.jdbc.Driver")
    val connection = DriverManager.getConnection(Url, user, password)

    // Give conn to client, through callable `f` passed as arg
    try {
      f(connection)
    } finally {
      // When client done, close conn
      connection.close()
    }
  }
}


SqlUtils.useConnection("test") { connection => println(connection) }
