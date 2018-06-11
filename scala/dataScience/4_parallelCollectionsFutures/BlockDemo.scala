import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

object BlockDemo extends App {
	val f = Future { Thread.sleep(10000) }
	f.onComplete { _ => println("future completed") }
	Await.result(f, 1 second)
}
