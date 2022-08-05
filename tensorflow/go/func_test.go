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

package tensorflow

import (
	"bytes"
	"testing"
)

// returns a test Func that returns the negative value of the input int8 value
func getNegFunc(t *testing.T, funcName string) *Func {
	g1 := NewGraph()
	x1, _ := Placeholder(g1, "x", Int8)
	y1, _ := Neg(g1, "neg", x1)
	fn, err := g1.AsFunc(funcName, []Output{x1}, []Output{y1}, []string{"y1"}, "TestFunc1")
	if err != nil {
		t.Fatal(err)
	}
	if fn == nil {
		t.Fatal("returned function is nil")
	}
	if gotName := fn.Name(); gotName != funcName {
		t.Errorf("function has wrong name: got %q, want %q", gotName, funcName)
	}
	return fn
}

func TestPlainFunc(t *testing.T) {
	const funcName = "funcOne"
	fn := getNegFunc(t, funcName)
	defer fn.Delete()

	// copy Func to another graph
	g2 := NewGraph()
	if err := g2.RegisterFunc(fn, nil); err != nil {
		t.Fatal(err)
	}
	if fns := g2.Functions(); len(fns) != 1 {
		t.Errorf("wrong number of functions: got %d, want 1", len(fns))
	} else if gotName := fns[0].Name(); gotName != funcName {
		t.Errorf("wrong function name: got %q, want %q", gotName, funcName)
	}

	// add Func operation to other graph
	x2Val := int8(+7)
	x2, _ := Const(g2, "x2", x2Val)
	funcOp, err := g2.addFunc(fn, "fn_1", x2)
	if err != nil {
		t.Fatal(err)
	}
	if want, got := 1, funcOp.NumOutputs(); want != got {
		t.Errorf("%q has wrong number of outputs: got %d, want %d", funcName, got, want)
	}

	// run Func in session
	sess, err := NewSession(g2, nil)
	if err != nil {
		t.Fatal(err)
	}
	fetched, err := sess.Run(nil, []Output{funcOp.Output(0)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want, got := -x2Val, fetched[0].Value().(int8); want != got {
		t.Errorf("%q calculated wrongly: got %d, want %d", funcName, got, want)
	}
}

func TestFuncImportExport(t *testing.T) {
	fn1 := getNegFunc(t, "xFunc")
	defer fn1.Delete()

	buf1 := bytes.Buffer{}
	n1, err := fn1.WriteTo(&buf1)
	if err != nil {
		t.Fatal(err)
	}
	if n1 != int64(buf1.Len()) {
		t.Errorf("exported sizes don't match: n=%d, len=%d", n1, buf1.Len())
	}

	fn2, err := ImportFuncFrom(buf1.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer fn2.Delete()
	buf2 := bytes.Buffer{}
	n2, err := fn2.WriteTo(&buf2)
	if err != nil {
		t.Fatal(err)
	}
	if n1 != n2 {
		t.Errorf("exported sizes don't match: got %d, want %d", n2, n1)
	}
	if !bytes.Equal(buf1.Bytes(), buf2.Bytes()) {
		t.Errorf("exported functions mismatch:\n%q\nvs\n%q", buf1.String(), buf2.String())
	}
}
