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

package op

import (
	"fmt"
	"runtime/debug"

	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

// Scope encapsulates common operation properties when building a [tf.Graph].
//
// A Scope object (and its derivatives, e.g., obtained from Scope.SubScope)
// act as a builder for graphs. They allow common properties (such as
// a name prefix) to be specified for multiple operations being added
// to the graph.
//
// A Scope object and all its derivatives (e.g., obtained from Scope.SubScope)
// are not safe for concurrent use by multiple goroutines.
type Scope struct {
	graph               *tf.Graph
	namemap             *opNameMap
	namespace           string
	controlDependencies []*tf.Operation
	device              string
	outTagMap           *outTagMap
	err                 *scopeErr
}

type opNameMap map[string]int
type outTagMap map[VarTag][]tf.Output

// scopeErr is used to share errors between all derivatives of a root scope.
type scopeErr struct {
	err error
}

// NewScope creates a Scope initialized with an empty graph
func NewScope() *Scope {
	return &Scope{
		graph:     tf.NewGraph(),
		namemap:   &opNameMap{},
		outTagMap: &outTagMap{},
		err:       new(scopeErr),
	}
}

// NewScopeWithGraph creates a Scope initialized with the graph thats passed in
func NewScopeWithGraph(graph *tf.Graph) *Scope {
	return &Scope{
		graph:     graph,
		namemap:   &opNameMap{},
		outTagMap: &outTagMap{},
		err:       new(scopeErr),
	}
}

// Finalize returns the [tf.Graph] on which this scope operates on and renders s
// unusable. If there was an error during graph construction, that error is
// returned instead.
func (s *Scope) Finalize() (*tf.Graph, error) {
	if err := s.Err(); err != nil {
		return nil, err
	}
	s.err.err = fmt.Errorf("Scope has been finalized and is no longer usable")
	return s.graph, nil
}

// AddOperation adds the operation to the [tf.Graph] managed by Scope s.
//
// If there is a name prefix associated with s (such as if s was created
// by a call to SubScope), then this prefix will be applied to the name
// of the operation being added. See also Graph.AddOperation.
func (s *Scope) AddOperation(args tf.OpSpec) *tf.Operation {
	if s.Err() != nil {
		return nil
	}
	if args.Name == "" {
		(*s.namemap)[args.Type]++
		args.Name += fmt.Sprintf("%s_%d", args.Type, (*s.namemap)[args.Type])
	}
	if s.namespace != "" {
		args.Name = s.namespace + "/" + args.Name
	}
	args.ControlDependencies = append(args.ControlDependencies, s.controlDependencies...)
	args.Device = s.device
	op, err := s.graph.AddOperation(args)
	if err != nil {
		s.UpdateErr(args.Type, err)
	}
	return op
}

// SubScope returns a new Scope which will cause all operations added to the
// graph to be namespaced with 'namespace'.  If namespace collides with an
// existing namespace within the scope, then a suffix will be added.
func (s *Scope) SubScope(namespace string) *Scope {
	namespace = s.uniqueName(namespace)
	if s.namespace != "" {
		namespace = s.namespace + "/" + namespace
	}
	return &Scope{
		graph:               s.graph,
		namemap:             &opNameMap{},
		namespace:           namespace,
		controlDependencies: s.controlDependencies,
		outTagMap:           s.outTagMap,
		device:              s.device,
		err:                 s.err,
	}
}

// WithControlDependencies returns a new Scope which will cause all operations
// added to the graph to execute only after all the provided operations have
// executed first (in addition to any other control dependencies in s).
func (s *Scope) WithControlDependencies(ops ...*tf.Operation) *Scope {
	// Force a copy of the control dependencies into a new underlying array on
	// every call.  We cannot alias the same underlying array as `ops`, otherwise
	// the user could modify that array after calling s.WithControlDependencies,
	// which would be confusing.  We cannot alias the same underlying array as the
	// original `s.controlDependencies`, since Scopes form a logical tree, and
	// other calls to s.WithControlDependencies could stomp on each other.
	deps := make([]*tf.Operation, 0, len(s.controlDependencies)+len(ops))
	deps = append(deps, s.controlDependencies...)
	deps = append(deps, ops...)
	return &Scope{
		graph:               s.graph,
		namemap:             s.namemap,
		namespace:           s.namespace,
		controlDependencies: deps,
		outTagMap:           s.outTagMap,
		device:              s.device,
		err:                 s.err,
	}
}

// WithDevice returns a new Scope which will cause all operations added to the
// graph to execute on devices that match the provided device specification.
//
// For example, WithDevice("/device:GPU:0") will cause operations added to
// the graph to execute on GPU #0.
//
// An empty string removes any device restrictions.
func (s *Scope) WithDevice(device string) *Scope {
	return &Scope{
		graph:               s.graph,
		namemap:             s.namemap,
		namespace:           s.namespace,
		controlDependencies: s.controlDependencies,
		outTagMap:           s.outTagMap,
		device:              device,
		err:                 s.err,
	}
}

// Err returns the error, if any, encountered during the construction
// of the [tf.Graph] managed by Scope s.
//
// Once Err returns a non-nil error, all future calls will do the same,
// indicating that the scope should be discarded as the graph could not
// be constructed.
func (s *Scope) Err() error {
	return s.err.err
}

// UpdateErr is used to notify Scope of any graph construction errors
// while creating the operation op.
func (s *Scope) UpdateErr(op string, err error) {
	if s.err.err == nil {
		s.err.err = fmt.Errorf("failed to add operation %q: %w (Stacktrace: %s)", op, err, debug.Stack())
	}
	// be vigilant
	panic(s.err.err)
}

func (s *Scope) uniqueName(name string) string {
	count := (*s.namemap)[name]
	(*s.namemap)[name]++
	return fmt.Sprint(name, "_", count)
}

// RegisterFunc copies and registers the [tf.Func] function and its eventual
// gradient into the graph and makes it available to be used in the graph.
// The function must not be nil.
func (s *Scope) RegisterFunc(fn, grad *tf.Func) error {
	if err := s.graph.RegisterFunc(fn, grad); err != nil {
		s.UpdateErr("RegisterFunc", err)
	}
	return nil
}

type VarTag string

const ( // parameter tags
	TagTrainable VarTag = "TagTrainable"
	TagDecayL1   VarTag = "TagDecayL1"
	TagDecayL2   VarTag = "TagDecayL2"
)

func (s *Scope) tagVariable(x tf.Output, tags ...VarTag) {
	for _, t := range tags {
		(*s.outTagMap)[t] = append((*s.outTagMap)[t], x)
	}
}

// GetParams returns all parameters in scope which have any of the requested tags.
// If no tags are provided then "TagTrainable" is assumed.
func (s *Scope) GetParams(tags ...VarTag) (params []tf.Output) {
	if len(tags) == 0 {
		tags = []VarTag{TagTrainable}
	}
	for _, t := range tags {
		params = append(params, (*s.outTagMap)[t]...)
	}
	return
}

// mustGetParams returns GetParams() and panics when no match is found
func (s *Scope) mustGetParams(tags ...VarTag) (params []tf.Output) {
	if params = s.GetParams(tags...); len(params) == 0 {
		panic(fmt.Errorf("no matching parameters found for tags: %v", tags))
	}
	return
}
