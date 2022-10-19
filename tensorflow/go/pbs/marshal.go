package pbs

import (
	"google.golang.org/protobuf/proto"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
)

// MustMarshal returns the wire-format encoding of m. It panics if this is not possible.
func MustMarshal(m protoreflect.ProtoMessage) []byte {
	opts := proto.MarshalOptions{AllowPartial: true, Deterministic: true}
	b, err := opts.Marshal(m)
	if err != nil {
		panic(err)
	}
	return b
}

// MustUnmarshal parses the wire-format message in b into m and also returns it.
// It panics if unmarshalling is not possible.
// To get e.g. a ConfigProto from its byte encoding use
//
//	e.g. MustUnmarshal(b, &ConfigProto{})
func MustUnmarshal[T protoreflect.ProtoMessage](b []byte, m T) T {
	opts := proto.UnmarshalOptions{AllowPartial: true, DiscardUnknown: true}
	if err := opts.Unmarshal(b, m); err != nil {
		panic(err)
	}
	return m
}
