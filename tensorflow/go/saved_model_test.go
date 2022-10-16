/*
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package tensorflow

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"
)

const savedModelSample = "testdata/saved_model/half_plus_two/00000123"

func TestSavedModelHalfPlusTwo(t *testing.T) {
	var (
		exportDir = savedModelSample
		tags      = []string{"serve"}
		options   = new(SessionOptions)
	)

	// Load saved model half_plus_two.
	m, err := LoadSavedModel(exportDir, tags, options)
	if err != nil {
		t.Fatalf("LoadSavedModel(): %v", err)
	}

	// Check that named operations x and y are present in the graph.
	if op := m.Graph.Operation("x"); op == nil {
		t.Fatalf("\"x\" not found in graph")
	}
	if op := m.Graph.Operation("y"); op == nil {
		t.Fatalf("\"y\" not found in graph")
	}

	// Define test cases for half plus two (y = 0.5 * x + 2).
	tests := []struct {
		name string
		X    float32
		Y    float32
	}{
		{"NegVal", -1, 1.5},
		{"PosVal", 1, 2.5},
		{"Zero", 0, 2.0},
		{"NegInf", float32(math.Inf(-1)), float32(math.Inf(-1))},
		{"PosInf", float32(math.Inf(1)), float32(math.Inf(1))},
	}

	// Run tests.
	for _, c := range tests {
		t.Run(c.name, func(t *testing.T) {
			x, err := NewTensor([]float32{c.X})
			if err != nil {
				t.Fatal(err)
			}

			y, err := m.Session.Run(
				map[Output]*Tensor{
					m.Graph.Operation("x").Output(0): x,
				},
				[]Output{
					m.Graph.Operation("y").Output(0),
				},
				nil,
			)
			if err != nil {
				t.Fatal(err)
			}

			got := y[0].Value().([]float32)[0]
			if got != c.Y {
				t.Fatalf("got: %#v, want: %#v", got, c.Y)
			}
		})
	}

	t.Logf("SavedModel: %+v", m)
	// TODO(jhseu): half_plus_two has a tf.Example proto dependency to run.
	// Add a more thorough test when the generated protobufs are available.
}

func TestSavedModelWithEmptyTags(t *testing.T) {
	var (
		exportDir = savedModelSample
		tags      = []string{}
		options   = new(SessionOptions)
	)

	_, err := LoadSavedModel(exportDir, tags, options)
	if err == nil {
		t.Fatalf("LoadSavedModel() should return an error if tags are empty")
	}
}

func TestSavedModelWithWrongTags(t *testing.T) {
	_, err := LoadSavedModel(savedModelSample, []string{"wrong"}, nil)
	if err == nil {
		t.Fatalf("LoadSavedModel() should return an error if tags don't match")
	}
	if got, want := err.Error(), "ListSavedModelDetails"; !strings.Contains(got, want) {
		t.Errorf("expected %q in %q error string", got, want)
	}
	t.Log(err)
}

func TestListSavedModelDetails(t *testing.T) {
	tags, allSigs, err := ListSavedModelDetails(savedModelSample)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := fmt.Sprintf("%v", tags), "[[serve]]"; got != want {
		t.Errorf("wrong tags: got %q, want %q", got, want)
	}
	var sigNames []string
	for _, graphSigs := range allSigs {
		for sigName := range graphSigs {
			sigNames = append(sigNames, sigName)
		}
	}
	sort.Strings(sigNames)
	want := "[classify_x2_to_y3 classify_x_to_y regress_x2_to_y3 regress_x_to_y regress_x_to_y2 serving_default]"
	if got := fmt.Sprintf("%v", sigNames); got != want {
		t.Errorf("wrong signature names:\n\tgot %q\n\twant %q", got, want)
	}
}
