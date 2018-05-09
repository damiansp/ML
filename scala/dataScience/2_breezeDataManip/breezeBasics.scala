import breeze.linalg._
import breeze.numerics._
import breeze.stats._

/** Vectors */
val v = DenseVector(1.0, 2.0, 3.0)
println(v(1)) // 2.0
println(v :* 2.0) // (2.0, 4.0, 6.0)
println(v :+ DenseVector(5.0, 7.0, 9.0)) // (6.0, 9.0, 12.0)
println(v :* 2) // error; will not coerce type
println(v :+ DenseVector(8.0, 9.0)) // error will not operate on mismatched vecs

val v2 = DenseVector(4.0, 5.0, 6.0)
println(v dot v2) // 32.0

// Classes Vector, SparseVector, and HashVector also available


/** Matrices */
val m = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
println(2.0 :* m) // 2.  4.  6.
                  // 8. 10. 12.


/** Building Vectors and Matrices */
val onesVec = DenseVector.ones[Double](5)
val zerosVec = DenseVector.zeros[Int](3)
val seq = linspace(0.0, 1.0, 10) // start, end, length
val multsOf5 = DenseVector.tabulate(4) { i => 5.0 * i } // 0.0, 5.0, 10.0, 15.0
val twoRowCol = DenseMatrix.tabulate[Int](2, 3) { (row, col) => 2*row + col }
val randVec = DenseVector.rand(2) // on [0, 1]
val randM = DenseMatrix.rand(2, 3)
val someVec = DenseVector(Array(2, 3, 4))
val someDigits = Seq(2, 3, 4)
val someDigitsVec = DenseVector(someDigits :_*) // for various scala collections


/** Advanced Indexing and Slicing */
val v = DenseVector.tabulate(5) { _.toDouble } // 0.0, 1.0, ..., 4.0
println(v(-1)) // last item (like Python)
println(v(1 to 3)) // 1.0, 2.0, 3.0
println(v(1 until 3)) // 1.0, 2.0
println(v(v.length - 1 to 0, by -1)) // reverse: 4.0, 3.0, ..., 0.0
val vBits = v(2, 4) // 2.0, 4.0; class = SliceVector (maps to original memory)
val vBitsVec = vBits.toDenseVector

val mask = DenseVector(true, false, false, true, true)
println(v(mask).toDenseVector) // 0.0, 3.0, 4.0
val filtered = v(v :< 3.0).toDenseVector // 0.0, 1.0, 2.0; :< element-wise <

println(m(1, 2)) // 6.0
println(m(-1, 1)) // 5.0
println(m(0 until 2, 0 until 2)) // 1.0 2.0
                                 // 4.0 5.0
println(m(::, 1)) // 2.0, 5.0; numpy m[:, 1]


/** Mutating Vectors and Matrices */
val v = DenseVector(1.0, 2.0, 3.0)
v(1) = 22.0
v(0 until 2) := DenseVector(50.0, 51.0)
v(0 until 2) := 0.0 // recycles vector

val v = DenseVector(1.0, 2.0, 3.0)
v :+= 4.0

val v = DenseVector.tabulate(6) { _.toDouble } // 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
val viewEvens = v(0 until v.length by 2) // 0.0, 2.0, 4.0
viewEvens := 10.0 // 10.0, 10.0, 10.0
v // 10.0, 1.0, 10.0, 3.0, 10.0, 5.0  !!
val copyEvens = v(0 until v.length by 2).copy


/** Matrix Multiplication, Transposition, and the Orientation of Vectors  */
val m1 = DenseMatrix((2.0, 3.0), (5.0, 6.0), (8.0, 9.0))
val m2 = DenseMatrix((10.0, 11.0), (12.0, 13.0))
m1 * m2 // matrix multiply

val v = DenseVector(1.0, 2.0)
m1 * v
val vTranspose = v.t
vTranspose * m1


/** Data Processing and Feature Engineering */
val data = HWData.load
data.genders
val maleVector = DenseVector.fill(data.genders.length)('M')
val isMale = (data.genders :== maleVector)
val maleHeights = data.heights(isMale)
maleHeights.toDenseVector
sum(I(isMale))
mean(data.heights)
mean(data.heights(isMale))
mean(data.heights(!isMale))

val discrepancy = (data.weights - data.reportedWeights) / data.weights
mean(discrepancy(isMale))
stddev(discrepancy(isMale))
val overReportMask = (data.reportedHeights :> data.heights).toDenseVector
sum(I(overReportMask :& isMale))
