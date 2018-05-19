/**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*/
object LogisticRegressionDemo {
  def main(args: Array[String]) {
    val data = HWData.load
    val regressor = new LogisticRegression(data.featureMatrix, data.target)
    val coefficients = regressor.optimalCoefficients
    println("Optimal coefficients (on rescaled data):")
    println(coefficients)
  }
}
