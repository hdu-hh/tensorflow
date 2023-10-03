package tensorflow

import (
	"math"
	"testing"
)

func TestCastWorker(t *testing.T) {
	const V = 123.456
	for _, sample := range []struct {
		name  string
		value float64
		dtype DataType
	}{
		{"bool", 1, Bool},
		{"int8", 123, Int8},
		{"int16", 123, Int16},
		{"int32", 123, Int32},
		{"int64", 123, Int64},
		{"uint8", 123, Uint8},
		{"uint16", 123, Uint16},
		{"uint32", 123, Uint32},
		{"uint64", 123, Uint64},
		{"qint8", 123, Qint8},
		{"qint16", 123, Qint16},
		{"uint32", 123, Qint32},
		{"quint8", 123, Quint8},
		{"quint16", 123, Quint16},
		{"float", 123.456, Float},
		{"double", 123.456, Double},
		{"bfloat16", 123.5, Bfloat16},
		{"half", 123.5, Half},
		{"f8_e5m2", 128, Float8e5m2},
		{"f8_e4m3", 120, Float8e4m3fn},
	} {
		t.Run(sample.name, func(t *testing.T) {
			x1 := sample.dtype.NewTensor(V)
			x2, err := CastTensor(Double, x1)
			if err != nil {
				t.Fatal(err)
			}
			if got := x1.DataType(); got != sample.dtype {
				t.Errorf("wrong dtype: got %T, want %s", got, sample.name)
			}
			if got := x2.Value().(float64); math.Abs(got-sample.value) > 0.1 {
				t.Errorf("wrong value: got %v, want %v", got, V)
			}
		})
	}
}
