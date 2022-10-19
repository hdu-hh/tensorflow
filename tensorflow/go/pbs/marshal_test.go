package pbs

import (
	fmt "fmt"
	"strings"
)

func ExampleMustMarshal() {
	m := ConfigProto{
		AllowSoftPlacement:   true,
		OperationTimeoutInMs: 10000,
		GraphOptions: &GraphOptions{
			OptimizerOptions: &OptimizerOptions{
				OptLevel: OptimizerOptions_L1,
			},
		},
		Experimental: &ConfigProto_Experimental{},
	}
	b := MustMarshal(&m)
	fmt.Printf("[]byte(%q)", b)
	// Output: []byte("8\x01R\x02\x1a\x00X\x90N\x82\x01\x00")
}

func ExampleMustUnmarshal() {
	b := []byte("8\x01R\x02\x1a\x00X\x90N\x82\x01\x00")
	m := MustUnmarshal(b, &ConfigProto{})
	s := strings.ReplaceAll(m.String(), "  ", " ") // work around spaces intentionally injected by protobuf stringer
	fmt.Printf("%T:\n%v", m, s)
	// Output:
	// *pbs.ConfigProto:
	// allow_soft_placement:true graph_options:{optimizer_options:{}} operation_timeout_in_ms:10000 experimental:{}
}
