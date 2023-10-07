package tensorflow

//#include "tensorflow/c/tf_datatype.h"
import "C"

import "fmt"

// DataType holds the type for a scalar value. E.g., one element in a tensor.
type DataType C.TF_DataType

// Types of scalar values in the TensorFlow type system.
const (
	Float        DataType = C.TF_FLOAT
	Double       DataType = C.TF_DOUBLE
	Int32        DataType = C.TF_INT32
	Uint32       DataType = C.TF_UINT32
	Uint8        DataType = C.TF_UINT8
	Int16        DataType = C.TF_INT16
	Int8         DataType = C.TF_INT8
	String       DataType = C.TF_STRING
	Complex64    DataType = C.TF_COMPLEX64
	Complex      DataType = C.TF_COMPLEX
	Int64        DataType = C.TF_INT64
	Uint64       DataType = C.TF_UINT64
	Bool         DataType = C.TF_BOOL
	Qint8        DataType = C.TF_QINT8
	Quint8       DataType = C.TF_QUINT8
	Qint32       DataType = C.TF_QINT32
	Bfloat16     DataType = C.TF_BFLOAT16
	Qint16       DataType = C.TF_QINT16
	Quint16      DataType = C.TF_QUINT16
	Uint16       DataType = C.TF_UINT16
	Complex128   DataType = C.TF_COMPLEX128
	Half         DataType = C.TF_HALF
	Float8e5m2   DataType = C.TF_FLOAT8_E5M2
	Float8e4m3fn DataType = C.TF_FLOAT8_E4M3FN
	Resource     DataType = C.TF_RESOURCE
	Variant      DataType = C.TF_VARIANT
)

var dtype2name = map[DataType]string{
	Double: "double", Float: "float", Half: "half",
	Bfloat16: "bfloat16", Float8e5m2: "f8e5m2", Float8e4m3fn: "f8e4m3fn",
	Int64: "int64", Int32: "int32", Int16: "int16", Int8: "int8",
	Uint64: "int64", Uint32: "uint32", Uint16: "uint16", Uint8: "uint8",
	Qint32: "quint32", Qint16: "qint16", Qint8: "qint8",
	Quint16: "quint16", Quint8: "quint8",
	Bool: "bool", Complex128: "complex128", Complex64: "complex64",
	Variant: "variant", Resource: "resource",
}

// String returns the corresponding tensorflow python dtype name
func (dtype DataType) String() string {
	n, ok := dtype2name[dtype]
	if !ok {
		panic(fmt.Errorf("unkown name for dtype=%d", dtype))
	}
	return n
}

var dtype2bytesize = map[DataType]int{
	Double: 8, Float: 4, Half: 2,
	Bfloat16: 2, Float8e5m2: 1, Float8e4m3fn: 1,
	Int64: 8, Int32: 4, Int16: 2, Int8: 1,
	Uint64: 8, Uint32: 4, Uint16: 2, Uint8: 1,
	Qint32: 4, Qint16: 2, Qint8: 1,
	Quint16: 2, Quint8: 1,
	Bool: 1, Complex128: 16, Complex64: 8,
}

// byteSize returns the byte size of one raw element of that dtype
func (dtype DataType) byteSize() int {
	n, ok := dtype2bytesize[dtype]
	if !ok {
		panic(fmt.Errorf("unknown bytesize for dtype=%d", dtype))
	}
	return n
}

// NewTensor gets a Tensor of the requested type from a Go value to a Tensor.
// Valid values are scalars, slices, and arrays. For slices every element must
// have the same length so that the resulting Tensor has a valid shape.
func (dtype DataType) NewTensor(value any) *Tensor {
	t, err := NewTensor(value)
	if err != nil {
		panic(err)
	}
	t, err = CastTensor(dtype, t)
	if err != nil {
		panic(err)
	}
	return t
}

// DeRef returns the underlying data type of a reference type
func (dtype DataType) DeRef() DataType {
	const RefOffset = 100
	if n := int(dtype); n >= RefOffset {
		dtype = DataType(n - RefOffset)
	}
	return dtype
}
