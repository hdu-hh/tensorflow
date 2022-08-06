/*
Copyright 2022 Herbert Dürr. All Rights Reserved.

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

package op

import (
	"math"
	"math/rand"
	"testing"

	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

func TestFuncCall1(t *testing.T) {
	mkFn1 := func(s *Scope, x ...tf.Output) (y []tf.Output, outNames []string, desc string) {
		y1 := Add(s, x[0], x[1])
		y2 := Pow(s, x[0], x[1])
		y2 = Cast(s, y2, tf.Int32)
		return []tf.Output{y1, y2}, nil, "just playing"
	}
	var (
		fn1     = BuildFunc("func1", mkFn1, tf.Float, tf.Float)
		s       = NewScope()
		_       = s.RegisterFunc(fn1, nil)
		c3      = Const(s, float32(3.0))
		y       = Func(s, fn1, c3, c3)
		g, _    = s.Finalize()
		sess, _ = tf.NewSession(g, nil)
		ft, _   = sess.Run(nil, y, nil)
	)
	if want, got := float32(6.0), ft[0].Value().(float32); got != want {
		t.Errorf("bad addition result: got %f, want %f", got, want)
	}
	if want, got := int32(27), ft[1].Value().(int32); got != want {
		t.Errorf("bad power result: got %d, want %d", got, want)
	}
}
func TestForOperation(t *testing.T) {
	mkBodyFn := func(s *Scope, x ...tf.Output) (outs []tf.Output, outNames []string, desc string) {
		y1 := Cast(s, x[0], tf.Double)
		y2 := Add(s, x[1], y1)
		return []tf.Output{y2}, nil, "dummy for-loop body"
	}

	inpVals := make([]float64, 10)
	for i := range inpVals {
		inpVals[i] = rand.Float64()
	}

	var (
		bodyFn = BuildFunc("bodyFn", mkBodyFn, tf.Int32, tf.Double)
		s      = NewScope()
		_      = s.RegisterFunc(bodyFn, nil)

		cStart   = Const(s, int32(3))
		cLimit   = Const(s, int32(9))
		cDelta   = Const(s, int32(5))
		cSamples = Const(s, inpVals)
		loopVals = For(s, cStart, cLimit, cDelta, []tf.Output{cSamples}, bodyFn)

		graph, _     = s.Finalize()
		sess, _      = tf.NewSession(graph, nil)
		fetched, err = sess.Run(nil, loopVals, nil)
	)
	if err != nil {
		t.Fatal(err)
	}
	gotVals := fetched[0].Value().([]float64)
	for i, got := range gotVals {
		if want := inpVals[i] + 3 + 8; math.Abs(got-want) > 1e-10 {
			t.Errorf("bad result at index %d: got %f, want %f, delta=%+e", i, got, want, got-want)
		}
	}
}

func TestWhile1(t *testing.T) {
	// get function for While condition
	mkCondFn := func(s *Scope, x ...tf.Output) (outs []tf.Output, outNames []string, desc string) {
		limit, value := x[0], x[1]
		var (
			compare   = Sub(s, value, limit)
			flatShape = Const(s, []int32{-1})
			flatVals  = Reshape(s, compare, flatShape)
			const0i   = Const(s, int32(0))
			flatMax   = Max(s, flatVals, const0i)
			const0f   = Const(s, float32(0))
			oneCond   = Less(s, flatMax, const0f)
		)
		return []tf.Output{oneCond}, nil, "get condition from inputs"
	}
	condFn := BuildFunc("condFn", mkCondFn, tf.Float, tf.Float, tf.Int64)

	// get function for While body
	mkBodyFn := func(s *Scope, x ...tf.Output) (outs []tf.Output, outNames []string, desc string) {
		limit, value, counter := x[0], x[1], x[2]
		var (
			cFactor  = Const(s, float32(1.01))
			outVal   = Mul(s, value, cFactor)
			const1   = Const(s, int64(1))
			outCount = Add(s, counter, const1)
		)
		return []tf.Output{limit, outVal, outCount}, nil, "get next loop values"
	}
	bodyFn := BuildFunc("bodyFn", mkBodyFn, tf.Float, tf.Float, tf.Int64)

	// get graph with While-op using the functions above
	var (
		s = NewScope()
		_ = s.RegisterFunc(condFn, nil)
		_ = s.RegisterFunc(bodyFn, nil)

		limitVal    = float32(100)
		cLimit      = Const(s, limitVal)
		initVals    = Const(s, []float32{+1, -2})
		initCounter = Const(s, int64(0))
		loopVals    = While(s, []tf.Output{cLimit, initVals, initCounter}, condFn, bodyFn)

		graph, _     = s.Finalize()
		sess, _      = tf.NewSession(graph, nil)
		fetched, err = sess.Run(nil, loopVals, nil)
	)
	if err != nil {
		t.Fatal(err)
	}
	gotVals := fetched[1].Value().([]float32)
	if gotVals[0] < limitVal {
		t.Errorf("expected first value>=%f, got %v", limitVal, gotVals)
	}
	gotCounter := fetched[2].Value().(int64)
	if want := int64(463); want != gotCounter {
		t.Errorf("expected %d iterations, got %v", want, gotCounter)
	}
}
