// worker is used e.g. for converting tensors
package tensorflow

import "fmt"

type (
	dtypePair struct{ inp, out DataType }
	opPair    struct{ inp, out *Operation }
	workerObj struct {
		graph   *Graph
		session *Session
		castMap map[dtypePair]opPair
	}
)

var worker workerObj

func CastTensor(dtype DataType, inp *Tensor) (out *Tensor, err error) {
	// short circuit if no casting is needed
	if inp.DataType() == dtype {
		return inp, nil
	}

	// ensure the worker session
	if worker.session == nil {
		worker.graph = NewGraph()
		worker.castMap = map[dtypePair]opPair{}
	}

	// get operations needed for type casting
	castKey := dtypePair{inp.DataType(), dtype}
	castOps, ok := worker.castMap[castKey]
	if !ok {
		castOps.inp, err = worker.graph.AddOperation(OpSpec{Type: "Const",
			Name:  fmt.Sprintf("castInp_%d_%d", castKey.inp, castKey.out),
			Attrs: map[string]any{"value": inp, "dtype": inp.DataType()},
		})
		if err != nil {
			return
		}
		castOps.out, err = worker.graph.AddOperation(OpSpec{Type: "Cast",
			Name:  fmt.Sprintf("castOut_%d_%d", castKey.inp, castKey.out),
			Input: []Input{castOps.inp.Output(0)},
			Attrs: map[string]interface{}{"DstT": dtype},
		})
		if err != nil {
			return
		}
		worker.castMap[castKey] = castOps
		worker.session, err = NewSession(worker.graph, nil)
		if err != nil {
			return
		}
	}

	// use the worker session to convert the tensor
	fetches, err := worker.session.Run(FeedMap{castOps.inp.Output(0): inp}, castOps.out.Outputs(), nil)
	if err != nil {
		return
	}
	return fetches[0], nil
}
