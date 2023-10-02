package op

import (
	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

// Linear implements a linear projection
// that by default is trainable with L2-norm weight decay and gets initialized with Xavier values.
// Use tags to select other behaviours.
func Linear(s *Scope, x tf.Output, outX int, tags ...VarTag) tf.Output {
	// prepare weights
	shape := tf.MakeShape(x.Shape().Size(-1), int64(outX)) // TODO: only last dim or add Dense-parm lastDims????
	dense := VariableV2(s, shape, x.DataType())
	// apply tags for the dense variable
	if len(tags) == 0 {
		tags = []VarTag{TagInitXavierNormal, TagTrainable, TagDecayL2}
	}
	s.tagVariable(dense, tags...)
	checked := CheckNumerics(s, dense, dense.Op.Name())
	return BatchMatMulV3(s, x, checked, x.DataType().DeRef())
}

// Bias implements a bias addition
// that by default is trainable with L1-decay and gets initialized with eps*uniform.
// Use tags to select other behaviours.
func Bias(s *Scope, x tf.Output, tags ...VarTag) tf.Output {
	// prepare biases
	bias := VariableV2(s, x.Shape(), x.DataType())
	if len(tags) == 0 {
		tags = []VarTag{TagInitEpsUniform, TagTrainable, TagDecayL1}
	}
	s.tagVariable(bias, tags...)
	checked := CheckNumerics(s, bias, bias.Op.Name())
	return Add(s, x, checked)
}

// MLP implements a multi layer perceptron
// (i.e. a linear projection eventually followed by a biased activation)
func MLP(s *Scope, x tf.Output, outX int, actFunc ActFunc, tags ...VarTag) tf.Output {
	y := Linear(s, x, outX, tags...)
	if actFunc != nil {
		y = Bias(s, y, tags...)
		y = actFunc(s, y)
	}
	return y
}
