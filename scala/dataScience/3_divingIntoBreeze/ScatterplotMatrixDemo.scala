/**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*/
import breeze.linalg._
import breeze.numerics._
import breeze.plot._

object ScatterplotMatrixDemo extends App {
	val data = HWData.load
	val fig = Figure("Scatterplot Matrix Demo")
	val m = new ScatterplotMatrix(fig)

	// Make matrix using height, weight and reported weight
	val featureMatrix = DenseMatrix.horzcat(data.heights.toDenseMatrix.t, 
		                                      data.weights.toDenseMatrix.t, 
		                                      data.reportedWeights.toDenseMatrix.t)
	m.plotFeatures(featureMatrix, List("height", "weight", "reportedWeights"))
	//fig.saveas("scatterplotDemo.png")
}