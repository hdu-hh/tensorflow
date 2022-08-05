<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

# An experiment to revive Tensorflow with Go

## Background

Tensorflow with Go is a match made in heaven, considering the power and elegance of
both projects. This is a reasonable assessment until you try the upstream Golang
binding for more than running existing models.

The upstream Golang binding has many strong points but
* Tensorflow lists it as [archived or unsupported](https://www.tensorflow.org/api_docs)
* even seemingly trivial code can cause direct or subtle errors
* coding with it requires plenty of boilerplate code and manual error checking
* delayed error reporting makes it harder than needed to find problems
* many operands for training non-trivial graphs are missing
* etc.

This experimental fork tries
* to remedy the most striking problems mentioned above
* to add convenience that is only available with tensorflow's python binding now
* to allow running as "close to the metal" as desired

## Achievements so far

* auto-naming of graph operators reduces name conflicts and the need for sub-scopes drastically
* auto-naming of var-handles avoids the default behaviour of var-handles sharing the same variable
* problems when building graphs are reported directly now
* the new method [Operator.AttrMap()](https://pkg.go.dev/github.com/hdu-hh/tensorflow/tensorflow/go#Operation.AttrMap) allows a much more generic way to manipulate graphs
* the code has become sufficiently self-standing and could live outside a full tensorflow repository
* the go package has proper go module support
* many more operators are available to users of the go package
* [function nodes](https://pkg.go.dev/github.com/hdu-hh/tensorflow/tensorflow/go/op#Func) are supported now and allow e.g. [op.While](https://pkg.go.dev/github.com/hdu-hh/tensorflow/tensorflow/go/op#While) nodes for low-overhead training

## GoDocs for the provided packages

The API documentation of the packages can be read online at `go.pkg.dev`. Please see
* Tensorflow API: https://pkg.go.dev/github.com/hdu-hh/tensorflow/tensorflow/go
* Operations API: https://pkg.go.dev/github.com/hdu-hh/tensorflow/tensorflow/go/op

## Upstream

The [Tensorflow site](https://www.tensorflow.org/) and its [Github repository](https://github.com/tensorflow/tensorflo) are great resources. Please see
them for all aspects that are not directly related to this friendly fork.

## Install

This experimental fork focusses on the Golang binding and so it just
provides the Go packages. It depends on an installed tensorflow library though. The [official upstream library builds](https://github.com/tensorflow/tensorflow#official-builds) are the best
source of it. Please see the [TensorFlow install guide](https://www.tensorflow.org/install) for the [CPU or GPU library](https://github.com/tensorflow/build/tree/master/golang_install_guide) for more details.

## Hello World for Tensorflow Go

To get started and the Go packages of this fork installed and into action
write e.g. the lines below into a Go source file and `go run` it:

```go
import (
  "fmt"
  tf "github.com/hdu-hh/tensorflow/tensorflow/go"
  "github.com/hdu-hh/tensorflow/tensorflow/go/op"
)

func main() {
    var (
      s       = op.NewScope()
      t1, _   = tf.NewTensor("hello ")
      t2, _   = tf.NewTensor("world")
      c1      = op.Const(s, t1)
      c2      = op.Const(s, t2)
      both    = op.Add(s, c1, c2)
      g, _    = s.Finalize()
      sess, _ = tf.NewSession(g, nil)
      f, _    = sess.Run(nil, []tf.Output{both}, nil)
    )
    fmt.Println(f[0].Value())
  }
```

It looks a bit awkward even though error handling was skipped
for readability. But the experimental fork works on making it
much simpler. E.g. in the near future something like this may
work:

```go
  func main() {
    var (
      g    = tf.NewGraph()
      both = g.Const("hello ").Add("world")
      f, _ = g.Run(nil, both)
    )
    fmt.Println(f[0].Value())
  }
```

## FAQ

### Is it possible and reasonable to split off the Golang binding into an own project?

The library and the binding interact strongly but through clean interfaces.
In the experimental fork the strong source code dependencies have been removed,
so splitting the Golang binding off into a separate project has become feasible.
This would also reduce the size of the Golang packages considerably.

On the other hand there is no urgency for that and keeping up with upstream
and eventual merge requests into upstream are much simpler.

### Does it work with [Tensorboard](https://github.com/tensorflow/tensorboard)?

Working with Tensorboard for the graph aspects and a modern Go dev environment
for the code is just an awesome combination. Tensorflow summary operators
are the key for this interaction. They already work, but it is not as easy
or comfortable as it could be. It is planned to make it so.

## License

[Apache License 2.0](LICENSE)
