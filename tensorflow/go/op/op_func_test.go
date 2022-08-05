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
	var (
		s1        = NewScope()
		limit1    = Placeholder(s1, tf.Float)
		value1    = Placeholder(s1, tf.Float)
		counter1  = Placeholder(s1, tf.Int64)
		compare   = Sub(s1, value1, limit1)
		flatShape = Const(s1, []int32{-1})
		flatVals  = Reshape(s1, compare, flatShape)
		const0i   = Const(s1, int32(0))
		flatMax   = Max(s1, flatVals, const0i)
		const0f   = Const(s1, float32(0))
		oneCond   = Less(s1, flatMax, const0f)
		g1, _     = s1.Finalize()
		condFn, _ = g1.AsFunc("condFn",
			[]tf.Output{limit1, value1, counter1},
			[]tf.Output{oneCond}, nil,
			"get condition from inputs",
		)
	)

	// get function for While body
	var (
		s2        = NewScope()
		limit2    = Placeholder(s2, tf.Float)
		value2    = Placeholder(s2, tf.Float)
		counter2  = Placeholder(s2, tf.Int64)
		cFactor   = Const(s2, float32(1.01))
		outVal    = Mul(s2, value2, cFactor)
		const1    = Const(s2, int64(1))
		outCount  = Add(s2, counter2, const1)
		g2, _     = s2.Finalize()
		bodyFn, _ = g2.AsFunc("bodyFn",
			[]tf.Output{limit2, value2, counter2},
			[]tf.Output{limit2, outVal, outCount}, nil,
			"get next loop values")
	)

	// get graph with While-op using the functions above
	s3 := NewScope()
	s3.RegisterFunc(condFn, nil)
	s3.RegisterFunc(bodyFn, nil)

	var (
		limitVal     = float32(100)
		cLimit       = Const(s3, limitVal)
		initVals     = Const(s3, []float32{+1, -2})
		initCounter  = Const(s3, int64(0))
		loopVals     = While(s3, []tf.Output{cLimit, initVals, initCounter}, condFn, bodyFn)
		g3, _        = s3.Finalize()
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
