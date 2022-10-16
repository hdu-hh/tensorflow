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
	"sort"
	"strings"
	"testing"
)

const savedModelSample = "../cc/saved_model/testdata/half_plus_two/00000123"

func TestSavedModel(t *testing.T) {
	tags := []string{"serve"}
	bundle, err := LoadSavedModel(savedModelSample, tags, nil)
	if err != nil {
		t.Fatalf("LoadSavedModel(): %v", err)
	}
	if op := bundle.Graph.Operation("y"); op == nil {
		t.Fatalf("\"y\" not found in graph")
	}
	t.Logf("SavedModel: %+v", bundle)
	// TODO(jhseu): half_plus_two has a tf.Example proto dependency to run. Add a
	// more thorough test when the generated protobufs are available.
}

func TestSavedModelWithEmptyTags(t *testing.T) {
	_, err := LoadSavedModel(savedModelSample, []string{}, nil)
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
