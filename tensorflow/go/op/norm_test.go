package op

import (
	"fmt"

	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

func ExampleL1norm() {
	x := [][]float32{{1.1, 2.2, 3.3}, {9.1, 8.2, 7.3}, {1, 1, 1}}
	var (
		s       = NewScope()
		c       = Const(s, x)
		n       = NormL1(s, c)
		g, _    = s.Finalize()
		sess, _ = tf.NewSession(g, nil)
		f, _    = sess.Run(nil, []tf.Output{n}, nil)
	)
	fmt.Printf("%.2f", f[0].Value())
	// Output: 3.80
}

func ExampleL2norm() {
	x := [][]float32{{1.1, 2.2, 3.3}, {9.1, 8.2, 7.3}, {1, 1, 1}}
	var (
		s       = NewScope()
		c       = Const(s, x)
		n       = NormL2(s, c)
		g, _    = s.Finalize()
		sess, _ = tf.NewSession(g, nil)
		f, _    = sess.Run(nil, []tf.Output{n}, nil)
	)
	fmt.Printf("%.3f", f[0].Value())
	// Output: 24.809
}

func ExampleLayerNorm() {
	x := [][]float32{{1.1, 2.2, 3.3}, {9.7, 8.8, 7.9}, {1, 1, 1}}
	var (
		s       = NewScope()
		c       = Const(s, x)
		n       = LayerNorm(s, c)
		g, _    = s.Finalize()
		sess, _ = tf.NewSession(g, nil)
		f, _    = sess.Run(nil, []tf.Output{n}, nil)
	)
	v := f[0].Value().([][]float32)
	for b := range x {
		for i := range x[0] {
			fmt.Printf("%+.5f,", v[b][i])
		}
		fmt.Println()
	}
	// Output:
	// -1.22474,+0.00000,+1.22474,
	// +1.22474,+0.00000,-1.22474,
	// +0.00000,+0.00000,+0.00000,
}

func ExampleBatchNorm() {
	x := [][]float32{{1.1, 2.2, 3.3}, {9.7, 8.8, 7.9}, {1, 1, 1}}
	var (
		s       = NewScope()
		c       = Const(s, x)
		n       = BatchNorm(s, c)
		g, _    = s.Finalize()
		sess, _ = tf.NewSession(g, nil)
		f, _    = sess.Run(nil, []tf.Output{n}, nil)
	)
	v := f[0].Value().([][]float32)
	for b := range x {
		for i := range x[0] {
			fmt.Printf("%+.5f,", v[b][i])
		}
		fmt.Println()
	}
	// Output:
	// -0.69481,-0.52489,-0.26726,
	// +1.41414,+1.39971,+1.33631,
	// -0.71933,-0.87482,-1.06904,
}
