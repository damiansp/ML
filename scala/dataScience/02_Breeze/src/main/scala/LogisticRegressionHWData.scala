import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats._


object LogisticRegressionHWData extends App {
  def rescale(v: DenseVector[Double]): DenseVector[Double] = (v - mean(v)) / stddev(v)

  val data = HWData.load
  val rescaledHeights = rescale(data.heights)
  val rescaledWeights = rescale(data.weights)
  val rescaledHeightsM = rescaledHeights.toDenseMatrix.t
  val rescaledWeightsM = rescaledWeights.toDenseMatrix.t
  val X = DenseMatrix.horzcat(
    DenseMatrix.ones[Double](rescaledHeightsM.rows, 1),
    rescaledHeightsM,
    rescaledWeightsM)
  println(s"Feature Matrix (X) size: (${X.rows}, ${X.cols})")
  val target = data.genders.values.map{gender => if (gender == 'M') 1.0 else 0.0}

  def costFunction(parameters: DenseVector[Double]): Double = {
    val xBeta = X * parameters
    val expXBeta = exp(xBeta)
    -sum((target *:* xBeta) - log1p(expXBeta))
  }

  def costFunctionGradient(parameters: DenseVector[Double]): DenseVector[Double] = {
    val xBeta = X * parameters
    val probs = sigmoid(xBeta)
    X.t * (probs - target)
  }

  val f = new DiffFunction[DenseVector[Double]] {
    def calculate(parameters: DenseVector[Double]) =
      (costFunction(parameters), costFunctionGradient(parameters))
  }

  val optimalParameters = minimize(f, DenseVector(0.0, 0.0, 0.0))
  println("Optimal Parameters:" + optimalParameters)
}
