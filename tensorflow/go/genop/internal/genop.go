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

// Package internal generates Go source code with functions for TensorFlow operations.
//
// The basic outline of the generated API is as follows:
//
// - One function for each TensorFlow operation
// - The arguments to the function are the inputs and required attributes of the operation
// - The function returns the outputs
// - A function is also generated for each optional attribute of the operation.
//
// There is a possibility that there are name collisions between the functions
// generated for ops and the functions generated for optional attributes. For
// now, we ignore those, but will need to revisit if a collision is actually
// encountered.
package internal

/*
#include <stdlib.h>

#include "tensorflow/c/c_api.h"
*/
import "C"

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"text/template"
	"unicode"
	"unsafe"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"

	"github.com/hdu-hh/tensorflow/tensorflow/go/pbs"
)

// GenerateFunctionsForRegisteredOps writes a Go source code file to w
// containing functions for each TensorFlow operation registered in the address
// space of the calling process.
// apidefDirs should be a contain of directories containing api_def_*.pbtxt
// files to load.
func GenerateFunctionsForRegisteredOps(
	w io.Writer, apidefDirs []string) error {
	ops, apimap, err := registeredOps()
	if err != nil {
		return err
	}
	for _, dir := range apidefDirs {
		if err = updateAPIDefs(apimap, dir); err != nil {
			return err
		}
	}
	return generateFunctionsForOps(w, ops, apimap)
}

func registeredOps() (*pbs.OpList, *apiDefMap, error) {
	buf := C.TF_GetAllOpList()
	defer C.TF_DeleteBuffer(buf)
	var (
		list = new(pbs.OpList)
		size = int(buf.length)
		// A []byte backed by C memory.
		// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
		data = (*[1 << 30]byte)(unsafe.Pointer(buf.data))[:size:size]
		err  = proto.Unmarshal(data, list)
	)
	if err != nil {
		return nil, nil, err
	}
	// Sort ops by name
	sort.Slice(list.Op, func(i, j int) bool {
		return list.Op[i].Name < list.Op[j].Name
	})
	apimap, err := newAPIDefMap(list)
	return list, apimap, err
}

func updateAPIDefs(m *apiDefMap, dir string) error {
	files, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".pbtxt") {
			continue
		}
		data, err := os.ReadFile(path.Join(dir, file.Name()))
		if err != nil {
			return fmt.Errorf("failed to read %q: %w", file.Name(), err)
		}
		if err = m.Put(string(data)); err != nil {
			err := fmt.Errorf("failed to process %q in %q:\n\t%w",
				file.Name(), dir, err)
			fmt.Fprintln(os.Stderr, err.Error())
		}
	}
	return nil
}

func generateFunctionsForOps(w io.Writer, ops *pbs.OpList, apimap *apiDefMap) error {
	thisPackage := reflect.TypeOf(tmplArgs{}).PkgPath()
	if err := tmplHeader.Execute(w, thisPackage); err != nil {
		return err
	}
	denylist := map[string]bool{
		"Const":           true,
		"PyFunc":          true,
		"PyFuncStateless": true,
	}
	for _, op := range ops.Op {
		if denylist[op.Name] {
			continue
		}
		apidef, err := apimap.Get(op.Name)
		if err != nil {
			return err
		}
		if err := generateFunctionForOp(w, op, apidef); err != nil {
			return err
		}
	}
	return nil
}

func generateFunctionForOp(w io.Writer, op *pbs.OpDef, apidef *pbs.ApiDef) error {
	if strings.HasPrefix(op.Name, "_") { // Internal operation
		log.Printf("ignored op \"%s\": name with underline", op.Name)
		return nil
	}
	// Ignore operations where the Go types corresponding to the TensorFlow
	// type haven't been worked out (such as "variants"s).
	for _, a := range op.Attr {
		if _, err := goType(a.Type); err != nil {
			log.Printf("ignored op \"%s\": type \"%v\" not handled yet", op.Name, a.Type)
			return nil
		}
	}
	// Also, haven't figured out reference types yet, so ignore those too.
	for _, a := range op.InputArg {
		if a.IsRef {
			log.Printf("should ignore op \"%s\": ref-type input", op.Name)
			break
		}
	}
	for _, a := range op.OutputArg {
		if a.IsRef {
			log.Printf("should ignore op \"%s\": ref-type output", op.Name)
			break
		}
	}
	if apidef.Summary == "" {
		// Undocumented operation, perhaps a sign of not being ready to
		// export.
		log.Printf("should ignore op \"%s\": no summary yet", op.Name)
	}
	if apidef.Visibility >= 2 {
		log.Printf("op \"%s\": ignoring %s visibility", op.Name,
			map[int]string{2: "SKIP", 3: "HIDDEN"}[int(apidef.Visibility)])
	}
	if false && op.Deprecation != nil {
		// ignore deprecated ops
		log.Printf("ignoring op \"%s\" as it is deprecated in version %d: %s", op.Name,
			op.Deprecation.Version, op.Deprecation.Explanation)
		return nil
	}
	tmplArgs, err := newTmplArgs(op, apidef)
	if err != nil {
		return err
	}
	return tmplOp.Execute(w, tmplArgs)
}

var (
	// Go keywords that cannot be used as identifiers.
	// From https://golang.org/ref/spec#Keywords
	keywords = []string{
		"break", "default", "func", "interface", "select", "case",
		"defer", "go", "map", "struct", "chan", "else", "goto",
		"package", "switch", "const", "fallthrough", "if", "range",
		"type", "continue", "for", "import", "return", "var",
	}

	tmplHeader = template.Must(template.New("header").Parse(`// DO NOT EDIT
// This file was machine generated by {{.}}
//
// WARNING: This generation of wrapper function for TensorFlow ops is in an
// experimental state. The generated API can change without notice.

package op

import (
	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
	"math/rand"
	"strconv"
)

// optionalAttr is an intentionally un-exported type to hide
// details of how optional attributes to operations are implemented.
type optionalAttr map[string]interface{}

func makeOutputList(op *tf.Operation, start int, output string) ([]tf.Output, int, error) {
	size, err := op.OutputListSize(output)
	if err != nil {
		return nil, start, err
	}
	list := make([]tf.Output, size)
	for i := 0; i < size; i++ {
		list[i] = op.Output(start + i)
	}
	return list, start + size, nil
}
`))

	tmplOp = template.Must(template.New("op").Funcs(template.FuncMap{
		"MakeComment":         makeComment,
		"MakeOfsComment":      makeOfsComment,
		"MakeLowComment":      makeLowComment,
		"GoType":              goType,
		"CamelCase":           camelCase,
		"Identifier":          identifier,
		"IsListArg":           isListArg,
		"IsListAttr":          isListAttr,
		"MarshalProtoMessage": marshalProtoMessage,
	}).Parse(`
{{if .OptionalAttrs -}}
{{/* Type for specifying all optional attributes. */ -}}
// {{.Op.Name}}Attr is an optional argument to {{.Op.Name}}.
type {{.Op.Name}}Attr func(optionalAttr)

{{range .OptionalAttrs}}
// {{$.Op.Name}}{{CamelCase .RenameTo}} sets the optional {{.RenameTo}} attribute to value.
{{- if .Description}}
//	- value: {{MakeOfsComment 5 .Description}}
//	  If not specified, defaults to {{MarshalProtoMessage .DefaultValue}}
{{- else}}
//	- value: If not specified, defaults to {{MarshalProtoMessage .DefaultValue}}
{{- end}}
{{- if .HasMinimum}}
//
// {{if .IsListAttr }}REQUIRES: len(value) >= {{.Minimum}}{{else}}REQUIRES: value >= {{.Minimum}}{{end}}
{{- end}}
func {{$.Op.Name}}{{CamelCase .RenameTo}}(value {{GoType .Type}}) {{$.Op.Name}}Attr {
	return func(m optionalAttr) {
		m[{{printf "%q" .Name}}] = value
	}
}
{{end}}
{{end}}

{{- /* Create a godoc friendly comment. */ -}}

{{- if .APIDef.Summary}}
// {{MakeComment .APIDef.Summary}}
{{- else -}}
// {{MakeComment .Op.Name}} APIDef has no summary yet
{{- end -}}

{{- with .Op.Deprecation}}
//
// DEPRECATED at GraphDef version {{.Version}}: {{.Explanation}}
{{- end -}}

{{- with .APIDef.Description}}
//
// {{MakeComment .}}
{{- end -}}

{{- if .DescribeArguments}}
//
// Arguments:
{{- range .InArgsReordered}}{{if .Description}}
//	- {{Identifier .RenameTo}}: {{MakeOfsComment 5 .Description}}{{end}}
{{- end -}}
{{- range .RequiredAttrs}}{{if .Description}}
//	- {{Identifier .RenameTo}}: {{MakeOfsComment 5 .Description}}{{end}}
{{- end -}}
{{- end -}}
{{- if (not .Op.OutputArg) }}
//
// Returns the created operation.
{{- else }}
{{- if .DescribeOutputs}}
//
{{- if eq (len .OutArgs) 1 }}
// Returns {{range .OutArgs}}{{MakeLowComment .Description}}{{end}}
{{- else }}
// Returns:
{{- range .OutArgs}}
//	- {{Identifier .RenameTo}}{{if .Description}}: {{MakeOfsComment 5 .Description}}{{end}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- /*

  The function signature.
  Since OpDef.Name is in CamelCase, it cannot conflict with a reserved keyword in Golang
*/}}
func {{.Op.Name}}

{{- /*
  Fill in input arguments:
  (1) The Scope
  (2) All input arguments (which may be either []tf.Output or tf.Output)
  (3) All required attributes
  (4) Variadic list of optional attributes
*/ -}}

(scope *Scope
{{- range $i, $a := .InArgsReordered}}, {{Identifier $a.RenameTo}} {{if $a.IsListArg}}[]{{end}}tf.Output{{end -}}
{{range $i, $a := .RequiredAttrs}}, {{Identifier $a.RenameTo}} {{GoType $a.Type}}{{end -}}
{{if .OptionalAttrs}}, optional ...{{.Op.Name}}Attr{{end -}}
)

{{- /* Construct outputs: len(.OutArgs) or a *tf.Operation */ -}}

{{if .OutArgs -}}
({{range $i,$a := .OutArgs}}{{if $i}}, {{end}}{{Identifier $a.RenameTo}} {{if $a.IsListArg}}[]{{end}}tf.Output{{end -}})
{{- else -}}
(o *tf.Operation)
{{- end }} {
	if scope.Err() != nil {
		return
	}
	{{if .HasAttrs -}}
	attrs := map[string]interface{}{ {{- range .RequiredAttrs}}{{printf "%q" .Name}}: {{Identifier .RenameTo}},{{end}}}
	{{if .OptionalAttrs -}}
	for _, a := range optional {
		a(attrs)
	}
	{{end -}}
	{{end -}}
	{{if .HasSharedName -}}
	// default to unique shared_name attribute
	if _, ok := attrs["shared_name"]; !ok {
		attrs["shared_name"] = "$" + strconv.FormatInt(rand.Int63(), 36)
	}
	{{end -}}
	opspec := tf.OpSpec{
		Type: {{printf "%q" .Op.Name}},
		{{if .InArgs -}}
		Input: []tf.Input{
			{{range $i,$a := .InArgs}}{{if $a.IsListArg}}tf.OutputList({{Identifier $a.RenameTo}}){{else}}{{Identifier $a.RenameTo}}{{end}}, {{end}}
		},
		{{- end}}
		{{- if .HasAttrs}}
		Attrs: attrs,
		{{- end}}
	}
	{{- if .OutArgs}}
	{{- if .HasListOutput}}
	op := scope.AddOperation(opspec)
	if scope.Err() != nil {
		return
	}
	var idx int
	var err error
	{{- range $i, $a := .OutArgs}}
	{{- if $a.IsListArg}}
	if {{Identifier .RenameTo}}, _, err = makeOutputList(op, idx, {{printf "%q" .Name}}); err != nil {
		scope.UpdateErr({{printf "%q" $.Op.Name}}, err)
		return
	}
	{{- else }}
	{{Identifier .RenameTo}} = op.Output(idx)
	{{- end }}{{- /* if IsListArg */}}
	{{- end }}{{- /* range .OutArgs */}}
	return {{range $i, $a := .OutArgs}}{{if $i}}, {{end}}{{Identifier .RenameTo}}{{end}}
	{{- else }}
	op := scope.AddOperation(opspec)
	return {{range $i, $a := .OutArgs}}{{if $i}}, {{end}}op.Output({{$i}}){{end}}
	{{- end }}{{- /* if .HasListOutput */}}
	{{- else }}
	return scope.AddOperation(opspec)
	{{- end }}{{- /* if .OutArgs */}}
}
`))
)

type attrWrapper struct {
	op  *pbs.OpDef_AttrDef
	api *pbs.ApiDef_Attr
}

func (a *attrWrapper) Name() string              { return a.api.Name }
func (a *attrWrapper) RenameTo() string          { return a.api.RenameTo }
func (a *attrWrapper) Description() string       { return a.api.Description }
func (a *attrWrapper) Type() string              { return a.op.Type }
func (a *attrWrapper) IsListAttr() bool          { return isListAttr(a.op) }
func (a *attrWrapper) HasMinimum() bool          { return a.op.HasMinimum }
func (a *attrWrapper) Minimum() int64            { return a.op.Minimum }
func (a *attrWrapper) DefaultValue() interface{} { return a.api.DefaultValue }

type argWrapper struct {
	op  *pbs.OpDef_ArgDef
	api *pbs.ApiDef_Arg
}

func (a *argWrapper) Name() string        { return a.api.Name }
func (a *argWrapper) RenameTo() string    { return a.api.RenameTo }
func (a *argWrapper) Description() string { return a.api.Description }
func (a *argWrapper) IsListArg() bool     { return isListArg(a.op) }

type tmplArgs struct {
	Op     *pbs.OpDef
	APIDef *pbs.ApiDef
	// Op.Attr is split into two categories
	// (1) Required: These must be specified by the client and are thus
	//     included in the function signature.
	// (2) Optional: These need not be specified (as they have default
	//     values) and thus do not appear in the function signature.
	RequiredAttrs []*attrWrapper
	OptionalAttrs []*attrWrapper
	HasSharedName bool
	InArgs        []*argWrapper
	// Input arguments ordered based on arg_order field of ApiDef.
	InArgsReordered []*argWrapper
	OutArgs         []*argWrapper
}

func newTmplArgs(op *pbs.OpDef, apidef *pbs.ApiDef) (*tmplArgs, error) {
	ret := tmplArgs{Op: op, APIDef: apidef}

	// Setup InArgs field
	for i, in := range op.InputArg {
		argCombined := argWrapper{op: in, api: apidef.InArg[i]}
		ret.InArgs = append(ret.InArgs, &argCombined)
	}

	// Setup OutArgs field
	for i, out := range op.OutputArg {
		argCombined := argWrapper{op: out, api: apidef.OutArg[i]}
		ret.OutArgs = append(ret.OutArgs, &argCombined)
	}

	// Setup InArgsReordered field
	for _, argName := range apidef.ArgOrder {
		// Find the argument in op.InputArg
		argIndex := -1
		for i, in := range op.InputArg {
			if in.Name == argName {
				argIndex = i
				break
			}
		}
		if argIndex == -1 {
			return nil, fmt.Errorf(
				"couldn't find argument %s in ApiDef for op %s",
				argName, op.Name)
		}
		argCombined := argWrapper{
			op: op.InputArg[argIndex], api: apidef.InArg[argIndex]}
		ret.InArgsReordered = append(ret.InArgsReordered, &argCombined)
	}

	if len(op.Attr) == 0 {
		return &ret, nil
	}
	// Attributes related to the InputArg's type are inferred automatically
	// and are not exposed to the client.
	inferred := make(map[string]bool)
	for _, in := range op.InputArg {
		switch {
		case in.TypeAttr != "":
			inferred[in.TypeAttr] = true
		case in.TypeListAttr != "":
			inferred[in.TypeListAttr] = true
		}
		if in.NumberAttr != "" {
			inferred[in.NumberAttr] = true
		}
	}
	for i, attr := range op.Attr {
		if inferred[attr.Name] {
			continue
		}
		attrCombined := attrWrapper{op: attr, api: apidef.Attr[i]}
		if attr.DefaultValue == nil {
			ret.RequiredAttrs = append(ret.RequiredAttrs, &attrCombined)
		} else {
			ret.OptionalAttrs = append(ret.OptionalAttrs, &attrCombined)
			if attrCombined.Name() == "shared_name" {
				ret.HasSharedName = true
			}
		}
	}
	return &ret, nil
}

func (a *tmplArgs) HasAttrs() bool {
	return len(a.RequiredAttrs)+len(a.OptionalAttrs) > 0
}

func (a *tmplArgs) DescribeArguments() bool {
	for _, arg := range a.InArgs {
		if arg.Description() != "" {
			return true
		}
	}
	for _, attr := range a.RequiredAttrs {
		if attr.Description() != "" {
			return true
		}
	}
	return false

}
func (a *tmplArgs) DescribeOutputs() bool {
	for _, arg := range a.OutArgs {
		if arg.Description() != "" {
			return true
		}
	}
	return false
}
func (a *tmplArgs) HasListOutput() bool {
	for _, arg := range a.OutArgs {
		if arg.IsListArg() {
			return true
		}
	}
	return false
}

func makeComment(lines string) string {
	lines = fixupMultilineComments(lines)
	l := strings.SplitAfter(lines, "\n")
	return strings.Join(l, "// ")
}

// adapt lines to go structured comment
func makeOfsComment(ofs int, lines string) string {
	l := strings.SplitAfter(lines, "\n")
	s := strings.Join(l, "//\t   "[:ofs])
	return strings.TrimSpace(s)
}

// lowercase first rune in a comment
func makeLowComment(lines string) (out string) {
	out = makeComment(lines)
	if len(lines) > 0 {
		ci := rune(out[0]) // assuming ASCII
		co := unicode.ToLower(ci)
		if ci != co {
			var b bytes.Buffer
			b.WriteRune(co)
			b.WriteString(out[1:])
			out = b.String()
		}
	}
	return
}

var mlquoteRe = regexp.MustCompile("(?m)$\\s*```([a-z]*\\s*)$")

func fixupMultilineComments(lines string) string {
	found := mlquoteRe.FindAllStringSubmatchIndex(lines, -1)
	if found != nil {
		fixed := strings.Builder{}
		oldIdx := 0
		for i := 0; i < len(found); i += 2 {
			m := found[i]
			n := []int{len(lines), len(lines), len(lines), len(lines)}
			if i+1 < len(found) {
				n = found[i+1]
			}
			fixed.WriteString(lines[oldIdx:m[0]])
			oldIdx = n[1]
			quote := lines[m[1]:n[0]]
			indented := strings.ReplaceAll(quote, "\n", "\n\t")
			fixed.WriteString(indented)
		}
		fixed.WriteString(lines[oldIdx:])
		lines = fixed.String()
	}
	return lines
}

// goType converts a TensorFlow "type" ('string', 'int', 'list(string)' etc.)
// to the corresponding type in Go.
func goType(tfType string) (string, error) {
	list, tfType := parseTFType(tfType)
	var gotype string
	switch tfType {
	case "int":
		gotype = "int64"
	case "float":
		gotype = "float32"
	case "bool":
		gotype = "bool"
	case "type":
		gotype = "tf.DataType"
	case "shape":
		gotype = "tf.Shape"
	case "tensor":
		gotype = "tf.Tensor"
	case "string":
		gotype = "string"
	case "func":
		gotype = "*tf.Func"
	default:
		return "", fmt.Errorf("%q is not a recognized DataType", tfType)
	}
	if list {
		gotype = "[]" + gotype
	}
	return gotype, nil
}

func camelCase(snakeCase string) string {
	words := strings.Split(snakeCase, "_")
	for i, w := range words {
		words[i] = strings.ToUpper(string(w[0])) + w[1:]
	}
	return strings.Join(words, "")
}

// identifier creates an identifier for s usable in the generated Go source
// code.
//
// Avoids collisions with keywords and other identifiers used in the generated
// code.
func identifier(s string) string {
	// Identifiers used in the generated code.
	if s == "tf" || s == "scope" || s == "err" || s == "op" {
		return s + "_"
	}
	for _, k := range keywords {
		if s == k {
			// Alternatively, make the first letter upper case.
			return s + "_"
		}
	}
	return s
}

func isListArg(argdef *pbs.OpDef_ArgDef) bool {
	return argdef.TypeListAttr != "" || argdef.NumberAttr != ""
}

func isListAttr(attrdef *pbs.OpDef_AttrDef) bool {
	list, _ := parseTFType(attrdef.Type)
	return list
}

func marshalProtoMessage(m proto.Message) string {
	// Marshal proto message to string.
	o := prototext.MarshalOptions{Multiline: false}
	x := o.Format(m)

	// Remove superfluous whitespace, if present.
	//
	// Go protobuf output is purposefully unstable, randomly adding
	// whitespace.  See github.com/golang/protobuf/issues/1121
	x = strings.ReplaceAll(x, "  ", " ")

	// Remove the prefix of the string up to the first colon.
	//
	// This is useful when 's' corresponds to a "oneof" protocol buffer
	// message. For example, consider the protocol buffer message:
	//   oneof value { bool b = 1;  int64 i = 2; }
	// proto.CompactTextString) will print "b:true", or "i:7" etc. The
	// following strips out the leading "b:" or "i:".
	y := strings.SplitN(x, ":", 2)
	if len(y) < 2 {
		return x
	}
	return y[1]
}

func parseTFType(tfType string) (list bool, typ string) {
	const (
		listPrefix = "list("
		listSuffix = ")"
	)
	if strings.HasPrefix(tfType, listPrefix) && strings.HasSuffix(tfType, listSuffix) {
		return true, strings.TrimSuffix(strings.TrimPrefix(tfType, listPrefix), listSuffix)
	}
	return false, tfType
}
