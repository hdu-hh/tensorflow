package tensorflow

import (
	"bytes"
	"fmt"
	"testing"
)

func TestSafeTensorLoader(t *testing.T) {
	// prepare input reader
	sample := map[string][]byte{"x": {123, 55, 17}}
	buf := bytes.NewBuffer([]byte{56, 0, 0, 0, 0, 0, 0, 0})
	buf.WriteString(`{"x":{"dtype":"I8","shape":[3,1],"data_offsets":[0,3]}} `)
	buf.Write(sample["x"])
	rd := bytes.NewReader(buf.Bytes())
	// create SafeTensorLoader
	srd, err := NewSafeTensorLoader(rd)
	if err != nil {
		t.Fatal(err)
	}
	// check tensor names
	allNames := srd.Names()
	if got, want := fmt.Sprintf("%v", allNames), "[x]"; got != want {
		t.Errorf("unexpected names: got %v, want %v", got, want)
	}
	// check tensor dtype and shape
	name := "x"
	dtype, shape := srd.Info(name)
	if got, want := dtype, Int8; got != want {
		t.Errorf("wrong dtype for %q, got %q, want %q", name, dtype, Int16)
	}
	if got, want := shape.String(), MakeShape(3, 1).String(); got != want {
		t.Errorf("wrong shape for %q, got %q, want %q", name, got, want)
	}
	// check value
	v, err := srd.LoadTensor(name)
	if err != nil {
		t.Error(err)
	} else if got, want := fmt.Sprintf("%v", v.Value()),
		fmt.Sprintf("%v", [][]int8{{123}, {55}, {17}}); got != want {
		t.Errorf("wrong value for %q, got %v, want %v", name, got, want)
	}
}

func TestSafeTensors_WriteAndLoad(t *testing.T) {
	// write as safe tensors to buffer
	wrMap := map[string]*Tensor{
		"bool": Bool.NewTensor(true),
		"u16":  Uint16.NewTensor([][]uint16{{123, 321}, {456, 654}}),
		"bf16": Bfloat16.NewTensor([]float32{321, 32.1, 3.21}),
	}
	wrNames := []string{"bool", "u16", "bf16"}
	writeBuf := bytes.Buffer{}
	err := WriteSafeTensors(&writeBuf, wrMap, wrNames)
	if err != nil {
		t.Fatal(err)
	}

	// read back from safe tensors buffer
	readBuf := bytes.NewReader(writeBuf.Bytes())
	loader, err := NewSafeTensorLoader(readBuf)
	if err != nil {
		t.Fatal(err)
	}
	rdNames := loader.Names()
	if got, want := len(rdNames), len(wrNames); got != want {
		t.Fatalf("wrong number of tensors: got %d, want %d for %v -> %v",
			got, want, wrNames, rdNames)
	}

	// compare written and loaded tensors
	for i, name := range rdNames {
		// check tensor order
		if got, want := name, wrNames[i]; got != want {
			t.Errorf("wrong tensor order at %d: got %q, want %q for %v -> %v",
				i, got, want, wrNames, rdNames)
		}
		// check dtype and shape
		dtype, shape := loader.Info(name)
		if got, want := dtype, wrMap[name].DataType(); got != want {
			t.Errorf("wrong data type for %q: got %q, want %q", name, got, want)
		}
		if got, want := shape.NumDimensions(), wrMap[name].Shape(); got != len(want) {
			t.Errorf("wrong dimmensions for %q: got %d, want %d", name, got, len(want))
		} else {
			for j, got := range shape.dims {
				if got != want[j] {
					t.Errorf("wrong shape dim %d for %q: got %d, want %d",
						j, name, got, want[j])
				}
			}
		}
	}
}
