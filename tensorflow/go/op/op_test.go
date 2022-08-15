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

// Tests for the generated code of some operations.

package op

import (
	"fmt"
	"testing"
	"time"

	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

func TestPlaceholder(t *testing.T) {
	s := NewScope()
	Placeholder(s.SubScope("x"), tf.Float, PlaceholderShape(tf.MakeShape(-1, 10)))
	Placeholder(s.SubScope("y"), tf.Float, PlaceholderShape(tf.ScalarShape()))
	Placeholder(s.SubScope("z"), tf.Float, PlaceholderShape(tf.Shape{}))
	if _, err := s.Finalize(); err != nil {
		t.Fatal(err)
	}
}

func TestAddOperationFailure(t *testing.T) {
	// prepare to recover from expected panic
	defer func() {
		if e := recover(); e == nil {
			t.Fatal("ResizeArea expects an int32 Tensor for size, should fail when an int64 is provided")
		}
	}()

	// Inspired from https://github.com/tensorflow/tensorflow/issues/9931
	s := NewScope()

	ResizeArea(s, Placeholder(s, tf.Float), Const(s, []int64{80, 80}))
}

func TestShapeAttribute(t *testing.T) {
	s := NewScope()
	x := Placeholder(s.SubScope("x"), tf.Int32, PlaceholderShape(tf.MakeShape(1)))
	y := Placeholder(s.SubScope("y"), tf.Int32, PlaceholderShape(tf.Shape{}))
	z := Add(s, x, y)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}

	value, err := tf.NewTensor([]int32{7})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[tf.Output]*tf.Tensor{
		x: value,
		y: value,
	}
	fetched, err := sess.Run(feeds, []tf.Output{z}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(fetched), 1; got != want {
		t.Fatalf("Fetched %d tensors, expected %d", got, want)
	}
	if got, want := fetched[0].Value().([]int32), []int32{14}; len(got) != len(want) || len(got) != 1 || got[0] != want[0] {
		t.Fatalf("Got %v, want %v", got, want)
	}
}

func ExampleVariable() {
	var (
		s       = NewScope()
		v       = Variable(s, tf.MakeShape(2, 3), tf.Float)
		ph      = Placeholder(s, tf.Float)
		wr      = Assign(s, v, ph)
		g, _    = s.Finalize()
		sess, _ = tf.NewSession(g, nil)
	)
	// assign tensor to variable
	t, _ := tf.NewTensor([][]float32{{1.3, 2.2, 3.1}, {4.6, 5.5, 6.4}})
	sess.Run(map[tf.Output]*tf.Tensor{ph: t}, nil, []*tf.Operation{wr.Op})
	// read tensor from variable
	f, _ := sess.Run(nil, []tf.Output{v}, nil)
	fmt.Println(f[0].Value())
	// Output: [[1.3 2.2 3.1] [4.6 5.5 6.4]]
}

func TestDataset(t *testing.T) {
	var (
		s = NewScope()

		// The use of a non-scalar here is inspired by
		// https://github.com/tensorflow/tensorflow/issues/14891
		c       = Const(s, []int32{21718, 31415})
		types   = []tf.DataType{c.DataType()}
		shapes  = []tf.Shape{c.Shape()}
		dataset = TensorDataset(s, []tf.Output{c}, shapes)

		iterator = Iterator(s, "", "", types, shapes)
		next     = IteratorGetNext(s, iterator, types, shapes)
		init     = MakeIterator(s, dataset, iterator)
	)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sess.Run(nil, nil, []*tf.Operation{init}); err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, next, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := results[0].Value().([]int32)
	if len(got) != 2 || got[0] != 21718 || got[1] != 31415 {
		t.Errorf("Got %v, want {21718, 31415}", got)
	}
	if _, err := sess.Run(nil, next, nil); err == nil {
		t.Errorf("Expected sess.Run() to fail since the iterator should have reached the end of the dataset")
	}
}

// use Optional's hasValue and getValue to drain a dataset
func TestDrainDataset(t *testing.T) {
	// define dataset contents
	var (
		types     = []tf.DataType{tf.Int64}
		shapes    = []tf.Shape{tf.ScalarShape()}
		loopLimit = 1000
	)

	var (
		s = NewScope()
		// prepare sum
		sumVar  = Variable(s, tf.ScalarShape(), tf.Int64)
		sumInit = Const(s, int64(0))
		wrSum0  = Assign(s, sumVar, sumInit)
		// simple RangeDataset for this test
		cOne    = Const(s, int64(1))
		cLimit  = Const(s, int64(loopLimit))
		dataset = RangeDataset(s, cOne, cLimit, cOne, types, shapes)
		// prepare iteration
		dsIter  = Iterator(s, "", "", types, shapes)
		mkIter  = MakeIterator(s, dataset, dsIter)
		nextOpt = IteratorGetNextAsOptional(s, dsIter, types, shapes)
		optVar  = Variable(s, tf.ScalarShape(), tf.Variant)
		wrOpt0  = Assign(s, optVar, nextOpt)
		hasVal  = OptionalHasValue(s, optVar)
		optVal  = OptionalGetValue(s, optVar, types, shapes)[0]
		addSum  = Add(s, sumVar, optVal)
		wrSumN  = Assign(s, sumVar, addSum)
		wrOptN  = Assign(s.WithControlDependencies(wrSumN.Op), optVar, nextOpt)
	)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	// connect the iterator to the dataset
	// and initialize the sum
	if _, err := sess.Run(nil, nil, []*tf.Operation{mkIter}); err != nil {
		t.Fatal(err)
	}
	// initialize the sum and the optional variable
	if _, err = sess.Run(nil, nil, []*tf.Operation{wrSum0.Op, wrOpt0.Op}); err != nil {
		t.Fatal(err)
	}
	// run the feed-fetch loop
	loopCount := 1
	startTime := time.Now()
	for ; loopCount <= 10*loopLimit; loopCount++ {
		fetched, _ := sess.Run(nil, []tf.Output{hasVal}, nil)
		if hasNext := fetched[0].Value().(bool); !hasNext {
			break
		}
		_, _ = sess.Run(nil, nil, []*tf.Operation{wrSumN.Op, wrOptN.Op})
	}
	dt := float64(time.Since(startTime).Nanoseconds())
	t.Logf("%d loops took %.1fms -> %.1fus/loop", loopLimit, dt*1e-6, dt*1e-3/float64(loopLimit))
	if loopCount != loopLimit {
		t.Errorf("looped %d times, wanted %d loops", loopCount, loopLimit)
	}
	fetched, _ := sess.Run(nil, []tf.Output{sumVar}, nil)
	if got, want := fetched[0].Value().(int64), int64((loopLimit-1)*loopLimit/2); want != got {
		t.Errorf("sum of while loop wrong: got %d, want %d", got, want)
	}
}
