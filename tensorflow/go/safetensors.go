package tensorflow

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

// safeTensorDict is the decoded tensor_dict of the safetensors data format
type safeTensorDict map[string]safeTensorItem

type safeTensorItem struct {
	DType   string
	Shape   []int64
	Offsets []int64 `json:"data_offsets"`
}

var dtype2safe = map[DataType]string{
	Bool:   "BOOL",
	Double: "F64", Float: "F32", Half: "F16", Bfloat16: "BF16",
	Int64: "I64", Int32: "I32", Int16: "I16", Int8: "I8",
	Uint64: "U64", Uint32: "U32", Uint16: "U16", Uint8: "U8",
}

var safe2dtype map[string]DataType

func init() {
	safe2dtype = make(map[string]DataType, len(dtype2safe))
	for k, v := range dtype2safe {
		safe2dtype[v] = k
	}
}

type SafeTensorLoader struct {
	reader io.ReadSeeker
	dict   safeTensorDict
	names  []string // sorted by offset for efficient reading
}

// NewSafeTensorLoader returns a SafeTensorLoader which can
// efficiently load data in the safetensors data interchange format.
func NewSafeTensorLoader(rd io.ReadSeeker) (*SafeTensorLoader, error) {
	var baseOfs int64
	err := binary.Read(rd, binary.LittleEndian, &baseOfs)
	if err != nil {
		return nil, err
	}
	srd := &SafeTensorLoader{reader: rd}
	dec := json.NewDecoder(io.LimitReader(rd, baseOfs))
	err = dec.Decode(&srd.dict)
	if err != nil {
		return srd, err
	}
	srd.names = make([]string, 0, len(srd.dict))
	baseOfs += 8
	for k, v := range srd.dict {
		srd.names = append(srd.names, k)
		v.Offsets[0] += baseOfs
		v.Offsets[1] += baseOfs
	}
	sort.Slice(srd.names, func(i, j int) bool {
		return srd.dict[srd.names[i]].Offsets[0] < srd.dict[srd.names[j]].Offsets[0]
	})
	return srd, nil
}

// Names returns the tensor names available.
// The names are sorted in the order most efficient for streaming.
func (loader *SafeTensorLoader) Names() []string { return loader.names }

// Info returns the data type and the shape of the named tensor
func (loader *SafeTensorLoader) Info(name string) (DataType, Shape) {
	x, ok := loader.dict[name]
	if !ok {
		panic(fmt.Errorf("safe tensor name %q not found", name))
	}
	dtype, ok := safe2dtype[x.DType]
	if !ok {
		panic(fmt.Errorf("unknown dtype %q for safe tensor name %q", x.DType, name))
	}
	return dtype, MakeShape(x.Shape...)
}

// LoadTensor returns the named tensor
func (loader *SafeTensorLoader) LoadTensor(name string) (*Tensor, error) {
	x, ok := loader.dict[name]
	if !ok {
		panic(fmt.Errorf("safe tensor name %q not found", name))
	}
	dtype, ok := safe2dtype[x.DType]
	if !ok {
		return nil, fmt.Errorf("unknown dtype %q for safe tensor name %q", x.DType, name)
	}
	if len(x.Offsets) != 2 {
		return nil, fmt.Errorf("cannot read safe tensor %q", name)
	}
	_, err := loader.reader.Seek(x.Offsets[0], io.SeekStart)
	if err != nil {
		return nil, fmt.Errorf("failed to read safe tensor %q", name)
	}
	t, err := ReadTensor(dtype, x.Shape, io.LimitReader(loader.reader, x.Offsets[1]-x.Offsets[0]))
	if err != nil {
		return nil, fmt.Errorf("failed to read safe tensor %q: %w", name, err)
	}
	return t, nil
}

// WriteSafeTensors writes the tensors in the map to the writer in safetensors format.
// If a names argument is provided the tensors are written in that order and tensors
// not mentioned in the name argument are skipped.
func WriteSafeTensors(wr io.Writer, dict map[string]*Tensor, names []string) error {
	// default to export all tensors in alphabetic order
	if names == nil {
		names = make([]string, 0, len(dict))
		for k := range dict {
			names = append(names, k)
		}
		sort.Strings(names)
	}

	// prepare tensor dictionary
	var offset int64
	safeDict := safeTensorDict{}
	for _, name := range names {
		t, ok := dict[name]
		if !ok {
			return fmt.Errorf("tensor %q not found for writing as save tensor", name)
		}
		safeType, ok := dtype2safe[t.DataType()]
		if !ok {
			return fmt.Errorf("dtype %q is not supported for safe tensor %q", t.DataType(), name)
		}
		numBytes := int64(dtype2bytesize[t.DataType()])
		numBytes *= numElements(t.Shape())
		safeDict[name] = safeTensorItem{
			DType:   safeType,
			Shape:   t.Shape(),
			Offsets: []int64{offset, offset + numBytes},
		}
		offset += numBytes
	}

	// export tensor dictionary as JSON
	jsonDict, _ := json.Marshal(&safeDict)
	jsonOfs := (len(jsonDict) + 7) & -8 // round up to 8 bytes
	binary.Write(wr, binary.LittleEndian, int64(jsonOfs))
	jsonDict = append(jsonDict, []byte("       ")...)
	wr.Write(jsonDict[:jsonOfs])
	// write tensor values
	for _, name := range names {
		t := dict[name]
		_, err := wr.Write(tensorData(t.c))
		if err != nil {
			return fmt.Errorf("failed to write safe tensor %q of %q and shape %v",
				name, t.DataType(), t.Shape())
		}
	}
	return nil
}
