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

package tensorflow

func _Placeholder(g *Graph, name string, dt DataType) Output {
	op, err := g.AddOperation(OpSpec{
		Type: "Placeholder",
		Name: name,
		Attrs: map[string]interface{}{
			"dtype": dt,
		},
	})
	if err != nil {
		panic(err)
	}
	return op.Output(0)
}

func _Const(g *Graph, name string, value interface{}) Output {
	t, ok := value.(*Tensor)
	if !ok {
		var err error
		if t, err = NewTensor(value); err != nil {
			return Output{}
		}
	}
	op, err := g.AddOperation(OpSpec{
		Type: "Const",
		Name: name,
		Attrs: map[string]interface{}{
			"dtype": t.DataType(),
			"value": t,
		},
	})
	if err != nil {
		panic(err)
	}
	return op.Output(0)
}

func _Neg(g *Graph, name string, port Output) Output {
	op, err := g.AddOperation(OpSpec{
		Type:  "Neg",
		Name:  name,
		Input: []Input{port},
	})
	if err != nil {
		panic(err)
	}
	return op.Output(0)
}

func _Add(g *Graph, name string, x, y Output) Output {
	op, err := g.AddOperation(OpSpec{
		Type:  "Add",
		Name:  name,
		Input: []Input{x, y},
	})
	if err != nil {
		panic(err)
	}
	return op.Output(0)
}
