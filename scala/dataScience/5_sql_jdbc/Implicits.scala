import java.sql._

// Implicit conversion methods typically stored in Implicits object:
object Implicits {
  implicit def pimpConnection(conn: Connection) = new RichConnection(conn)
  implicit def depimpConnection(conn: RichConnection) = conn.underlying
  implicit def pimpResultSet(results: ResultSet) = new RichResultSet(results)
}