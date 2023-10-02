package op

import tf "github.com/hdu-hh/tensorflow/tensorflow/go"

type NormFunc func(*Scope, tf.Output) tf.Output

// NormL1 returns the L1 norm of a tensor, i.e. the sum of its element values
func NormL1(s *Scope, x tf.Output) tf.Output {
	axis0 := Const(s, int32(0))
	return Mean(s, Flatten(s, x), axis0)
}

// NormAbsL1 returns the L1 norm of an absolute tensor, i.e. the sum of its absolute element values
func NormAbsL1(s *Scope, x tf.Output) tf.Output {
	axis0 := Const(s, int32(0))
	return Mean(s, Abs(s, Flatten(s, x)), axis0)
}

// NormL2 returns the L2 norm of a tensor, i.e. the sum of its squared element values
func NormL2(s *Scope, x tf.Output) tf.Output {
	axis0 := Const(s, int32(0))
	return Mean(s, Square(s, Flatten(s, x)), axis0)
}

// LayerNorm normalizes each batch element
func LayerNorm(s *Scope, x tf.Output) tf.Output {
	batches := x.Shape().Size(0)
	l := Reshape(s, x, Const(s, []int64{batches, -1}))
	axis1 := Const(s, int32(1))
	mean1 := Mean(s, l, axis1, MeanKeepDims(true))
	mean2 := Mean(s, Square(s, l), axis1, MeanKeepDims(true))
	eps := Const(s, float32(1e-6))
	rvari := Rsqrt(s, Add(s, eps, Sub(s, mean2, Square(s, mean1))))
	y := Mul(s, rvari, Sub(s, x, mean1))
	return y
}

// BatchNorm normalizes along the batch axis
func BatchNorm(s *Scope, x tf.Output) tf.Output {
	axis0 := Const(s, int32(0))
	mean1 := Mean(s, x, axis0)
	mean2 := Mean(s, Square(s, x), axis0)
	eps := Const(s, float32(1e-6))
	rvari := Rsqrt(s, Add(s, eps, Sub(s, mean2, Square(s, mean1))))
	y := Mul(s, rvari, Sub(s, x, mean1))
	return y
}
