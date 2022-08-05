/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package op defines functions for adding TensorFlow operations to a Graph.
//
// Functions for adding an operation to a graph take a Scope object as the
// first argument. The Scope object encapsulates a graph and a set of
// properties (such as a name prefix) for all operations being added
// to the graph.
//
// WARNING: The API in this package has not been finalized and can
// change without notice.
package op

import (
	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

// Const adds an operation to graph that produces value as output.
func Const(scope *Scope, value interface{}) (output tf.Output) {
	if scope.Err() != nil {
		return
	}
	t, ok := value.(*tf.Tensor)
	if !ok {
		var err error
		if t, err = tf.NewTensor(value); err != nil {
			scope.UpdateErr("Const", err)
			return
		}
	}
	return scope.AddOperation(tf.OpSpec{
		Type: "Const",
		Attrs: map[string]interface{}{
			"dtype": t.DataType(),
			"value": t,
		}}).Output(0)
}

// Func adds the tensorflow function to the graph
//
// The easiest way to get a [tf.Func] is to
//  * use [BuildFunc] or [BuildFuncPair] when building function graphs from scratch
//  * use the [tf.Graph.AsFunc] method when converting an existing graph
//  * use the [tf.ImportFunc] function when a FunctionDef protobuf is available
func Func(scope *Scope, fn *tf.Func, inputs ...tf.Input) []tf.Output {
	fnOp := scope.AddOperation(tf.OpSpec{
		Type:  fn.Name(),
		Input: inputs,
	})
	if scope.Err() != nil {
		return nil
	}
	return fnOp.Outputs()
}

// GoFunc is a function signature used for building a [tf.Func] from a go function.
// It returns the outputs, their names and a description string.
// Please see [BuildFunc] for an example.
type GoFunc func(s *Scope, inputs ...tf.Output) (outputs []tf.Output, outNames []string, desc string)

// BuildFunc returns a `tf.Func`` matching to a go function.
// The provided go function must have the GoFunc signature.
// e.g.
//	 goFunc := func(s *Scope, x ...tf.Output) (y []tf.Output, outNames []string, desc string) {
//		return []tf.Output{op.Add(s, x[0], x[1])}, nil, "just adding"
// 	 }
// 	 tfFunc := BuildFunc("adder", goFunc, tf.Float, tf.Float)
func BuildFunc(name string, goFunc GoFunc, dtypes ...tf.DataType) *tf.Func {
	// create placeholders for the inputs
	s := NewScope()
	var phs []tf.Output
	if l := len(dtypes); l > 0 {
		phs = make([]tf.Output, l)
		for i, dt := range dtypes {
			phs[i] = Placeholder(s, dt)
		}
	}
	// trace the go function
	outs, outNames, desc := goFunc(s, phs...)
	if err := s.Err(); err != nil {
		panic(err)
	}
	// build the graph
	g, err := s.Finalize()
	if err != nil {
		panic(err)
	}
	// get the function for the graph
	tfFunc, err := g.AsFunc(name, phs, outs, outNames, desc)
	if err != nil {
		panic(err)
	}
	return tfFunc
}

// BuildFuncPair returns two tf.Pair for functions sharing the signature.
// This simplifies handling op.While that requires such a pair.
func BuildFuncPair(name1, name2 string, goFn1, goFn2 GoFunc, dtypes ...tf.DataType) (*tf.Func, *tf.Func) {
	tfFn1 := BuildFunc(name1, goFn1, dtypes...)
	tfFn2 := BuildFunc(name2, goFn2, dtypes...)
	return tfFn1, tfFn2
}
