import java.sql.{ResultSet, Connection}
import Implicits._


object PhysicistDao {
  private def readFromResultSet(results: ResultSet): Physicist = {
    Physicist(results.read[String]("name"), results.read[Gender.Value]("gender"))
  }


  def readAll(connection: Connection): Vector[Physicist] = {
    connection.withQuery("SELECT * FROM physicist") { results =>
      val resultStream = SqlUtils.stream(results)
      resultStream.map(readFromResultSet).toVector
    }
  }
}