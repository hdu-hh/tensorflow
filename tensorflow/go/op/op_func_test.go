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
	"testing"

	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

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
	s := NewScope()
	s.RegisterFunc(condFn, nil)
	s.RegisterFunc(bodyFn, nil)

	var (
		limitVal     = float32(100)
		cLimit       = Const(s, limitVal)
		initVals     = Const(s, []float32{+1, -2})
		initCounter  = Const(s, int64(0))
		loopVals     = While(s, []tf.Output{cLimit, initVals, initCounter}, condFn, bodyFn)
		g3, _        = s.Finalize()
		sess, _      = tf.NewSession(g3, nil)
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
