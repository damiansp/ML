import breeze.linalg._

val v = DenseVector(1.0, 2.0, 3.0)
println(v(1)) // 2.0
println(v :* 2.0) // (2.0, 4.0, 6.0)
println(v :+ DenseVector(5.0, 7.0, 9.0)) // (6.0, 9.0, 12.0)
println(v :* 2) // error; will not coerce type
println(v :+ DenseVector(8.0, 9.0)) // error will not operate on mismatched vecs

val v2 = DenseVector(4.0, 5.0, 6.0)
print(v dot v2) // 32.0

// Classes Vector, SparseVector, and HashVector also available