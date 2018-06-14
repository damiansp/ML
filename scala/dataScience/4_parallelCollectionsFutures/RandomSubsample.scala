import breeze.linalg._
import breeze.numerics._


class RandomSubsample(val nElements: Int, val nCV: Int) {
	type CVFunction = (Seq[Int], Seq[Int]) => Double

	require(nElements > nCV, "nCV, the number of test items, must be < nElements")

	private val indexList = DenseVector.range(0, nElements)

	def mapSamples(nShuffles: Int)(f: CVFunction): DenseVector[Double] = {
		val cvResults = (0 to nShuffles).par.map { i=>
			val shuffledIndices = breeze.linalg.shuffle(indexList)
			val Seq(testIndices, trainingIndices) = split(shuffledIndices, Seq(nCV))
			f(trainingIndices.toScalaVector, test.toScalaVector)
		}
		DenseVector(cvResults.toArray)
	}
}