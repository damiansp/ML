/**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*//**-----*/
import breeze.linalg._
import breeze.numerics._
import breeze.plot._
import breeze.stats.regression._
import org.jfree.chart.axis.NumberTickUnit
import org.jfree.chart.annotations.XYTextAnnotation
import org.jfree.chart.plot.ValueMarker

object BreezeDemo extends App {
  def simplePlot {
    val x = linspace(-4.0, 4.0, 100)
    val fig = Figure("Sigmoid")
    val plt = fig.subplot(0)

    plt += plot(x, sigmoid(x), name="f(x)")
    plt += plot(x, sigmoid(2.0 * x), name="f(2x)")
    plt += plot(x, sigmoid(10.0 * x), name="f(10x)")
    plt.yaxis.setTickUnit(new NumberTickUnit(0.1))
    plt.plot.addDomainMarker(new ValueMarker(0.0))
    plt.plot.addRangeMarker(new ValueMarker(1.0))
    plt.xlim = (-4.0, 4.0)
    plt.xlabel = "x"
    plt.ylabel = "sigmoid(x)"
    plt.legend = true
  }


  def scatterPlot {
  	val data = HWData.load
  	val heights = data.heights
  	val weights = data.weights
  	val leastSquaresResult = leastSquares(
  	  DenseMatrix.horzcat(DenseMatrix.ones[Double](data.nPoints, 1), heights.toDenseMatrix.t),
  	  weights)
  	val leastSquaresCoefficients = leastSquaresResult.coefficients
  	val label = (
	  f"weight = ${leastSquaresCoefficients(0)}%.4f + ${leastSquaresCoefficients(1)}%.4f * height")

  	println("Least Squares Result: ")
  	println(label)
  	println(s"residuals = ${leastSquaresResult.rSquared}")

  	val dummyHeights = linspace(heights.min, heights.max, 200)
  	val fitted = leastSquaresCoefficients(0) :+ (leastSquaresCoefficients(1) :* dummyHeights)
  	val fig = Figure("Height vs Weight")
  	val plt = fig.subplot(0)
  	plt += plot(heights, weights, '+', colorcode="black")
  	plt += plot(dummyHeights, fitted, colorcode="red")
  	plt.plot.addAnnotation(new XYTextAnnotation(label, 175.0, 105.0))
  }


  def advancedScatterExample {
  	val fig = new Figure("Advanced Scatterplot Example")
  	val plt = fig.subplot(0)
  	val xs = linspace(0.0, 1.0, 100)
  	val sizes = 0.025 * rand(100)
  	val colorPalette = new GradientPaintScale(0.0, 1.0, PaintScale.MaroonToGold)
  	val colors = DenseVector.rand(100).mapValues(colorPalette)

  	plt += scatter(xs, xs :^ 2.0, sizes.apply, colors.apply)
  }


  def subplotExample {
  	val data = HWData.load
  	val fig = new Figure("Subplot example")

  	// Upper subplot
  	var plt = fig.subplot(2, 1, 0)
  	plt += plot(data.heights, data.weights, '.')

  	// Lower
  	plt = fig.subplot(2, 1, 1)
  	plt += plot(data.heights, data.reportedHeights, '.', colorcode="black")
  }

	//simplePlot
	//scatterPlot
	//advancedScatterExample
	subplotExample
}

