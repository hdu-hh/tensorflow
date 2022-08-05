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

// #include "tensorflow/c/c_api.h"
import "C"

import (
	"fmt"
	"io"
	"unsafe"
)

// Func represents a tensorflow function.
type Func struct {
	c *C.TF_Function
}

// Name returns the name of the tensorflow function
func (fn *Func) Name() string {
	cName := C.TF_FunctionName(fn.c)
	return C.GoString(cName)
}

// Delete the tensorflow function
// Deleting a function does not remove it from any graphs it was copied to.
func (fn *Func) Delete() {
	C.TF_DeleteFunction(fn.c)
}

// RegisterFunc copies and registers the tensorflow function
// and its eventual gradient into the graph.
// The function must not be nil.
func (g *Graph) RegisterFunc(fn, grad *Func) error {
	if fn == nil {
		return fmt.Errorf("cannot register a nil Func")
	}
	if grad == nil {
		grad = &Func{}
	}
	status := newStatus()
	C.TF_GraphCopyFunction(g.c, fn.c, grad.c, status.c)
	return status.Err()
}

// Functions returns the list of functions registered in the graph
func (g *Graph) Functions() []*Func {
	numFuncs := C.TF_GraphNumFunctions(g.c)
	if numFuncs <= 0 {
		return nil
	}
	cFuncs := make([]*C.TF_Function, numFuncs)
	status := newStatus()
	gotFuncs := C.TF_GraphGetFunctions(g.c, &cFuncs[0], numFuncs, status.c)
	if err := status.Err(); err != nil {
		err = fmt.Errorf("failed to get functions for graph: %w", err)
		panic(err)
	}
	goFuncs := make([]*Func, gotFuncs)
	for i := range goFuncs {
		goFuncs[i] = &Func{cFuncs[i]}
	}
	return goFuncs
}

// AsFunc returns the tensorflow function corresponding to the graph
func (g *Graph) AsFunc(name string, inputs, outputs []Output, outNames []string, desc string) (*Func, error) {
	if numOuts, numNames := len(outputs), len(outNames); numOuts != numNames && numNames != 0 {
		return nil, fmt.Errorf("mismatch of outputs and their names: %d vs %d", numOuts, numNames)
	}

	var pInputs *C.TF_Output
	if len(inputs) > 0 {
		cInputs := make([]C.TF_Output, len(inputs))
		for i, inp := range inputs {
			cInputs[i] = inp.c()
		}
		pInputs = &cInputs[0]
	}
	var pOutputs *C.TF_Output
	if len(outputs) > 0 {
		cOutputs := make([]C.TF_Output, len(outputs))
		for i, out := range outputs {
			cOutputs[i] = out.c()
		}
		pOutputs = &cOutputs[0]
	}
	var pOutNames **C.char
	if len(outNames) > 0 {
		cOutNames := make([]*C.char, len(outNames))
		for i, name := range outNames {
			cOutNames[i] = C.CString(name)
		}
		pOutNames = &cOutNames[0]
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cDesc := C.CString(desc)
	defer C.free(unsafe.Pointer(cDesc))
	cHashFnName := C.uchar(1) // name hashing enabled

	// TODO? select subset of ops needed for inputs to outputs
	cNumOpers := C.int(-1)      // all ops for now -> -1
	var pOpers **C.TF_Operation // all graph ops for now -> nil

	status := newStatus()
	fn := C.TF_GraphToFunction(
		g.c, cName, cHashFnName,
		cNumOpers, pOpers,
		C.int(len(inputs)), pInputs,
		C.int(len(outputs)), pOutputs,
		pOutNames,
		nil, cDesc, status.c)
	if err := status.Err(); err != nil {
		panic(err)
	}
	return &Func{fn}, nil
}

func (g *Graph) addFunc(fn *Func, name string, inputs ...Input) (*Operation, error) {
	return g.AddOperation(OpSpec{
		Type:  fn.Name(),
		Name:  name,
		Input: inputs,
	})
}

// WriteTo writes out a serialized representation.
//
// Implements the io.WriterTo interface.
func (fn *Func) WriteTo(w io.Writer) (int64, error) {
	buf := C.TF_NewBuffer()
	defer C.TF_DeleteBuffer(buf)
	status := newStatus()
	C.TF_FunctionToFunctionDef(fn.c, buf, status.c)
	if err := status.Err(); err != nil {
		return 0, err
	}
	proto, err := getBufferAsSlice(buf)
	if err != nil {
		return 0, err
	}
	n, err := w.Write(proto)
	return int64(n), err
}

// ImportFunc creates a Func from a serialized representation.
func ImportFuncFrom(proto []byte) (*Func, error) {
	if len(proto) == 0 {
		return nil, fmt.Errorf("cannot import from empty buffer")
	}
	status := newStatus()
	cFn := C.TF_FunctionImportFunctionDef(unsafe.Pointer(&proto[0]),
		C.size_t(len(proto)), status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}
	return &Func{cFn}, nil
}

// SetAttrProto sets the function attribute to the
// binary serialization of an AttrValue protocol buffer.
func (fn *Func) SetAttrProto(attrName string, proto []byte) error {
	cLen := C.size_t(len(proto))
	if len(proto) == 0 {
		proto = []byte{0}
	}
	cName := C.CString(attrName)
	defer C.free(unsafe.Pointer(cName))
	status := newStatus()
	C.TF_FunctionSetAttrValueProto(fn.c, cName,
		unsafe.Pointer(&proto[0]), cLen, status.c)
	return status.Err()
}

// WriteAttrTo writes out a serialized representation of the function attribute.
func (fn *Func) WriteAttrProto(attrName string, w io.Writer) (int64, error) {
	cName := C.CString(attrName)
	defer C.free(unsafe.Pointer(cName))
	buf := C.TF_NewBuffer()
	defer C.TF_DeleteBuffer(buf)

	status := newStatus()
	C.TF_FunctionGetAttrValueProto(fn.c, cName, buf, status.c)
	if err := status.Err(); err != nil {
		return 0, err
	}
	if buf.length == 0 {
		return 0, nil
	}
	proto, err := getBufferAsSlice(buf)
	if err != nil {
		return 0, err
	}
	n, err := w.Write(proto)
	return int64(n), err
}
