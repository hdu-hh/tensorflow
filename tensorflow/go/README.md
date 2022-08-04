# TensorFlow in Go

Construct and execute TensorFlow graphs in Go.

[![GoDoc](https://godoc.org/github.com/hdu-hh/tensorflow/tensorflow/go?status.svg)](https://godoc.org/github.com/hdu-hh/tensorflow/tensorflow/go)

> *WARNING*: The API defined in this package is not stable and can change
> without notice. The same goes for the package path:
> (`github.com/hdu-hh/tensorflow/tensorflow/go`).

# WARNING:

The TensorFlow team is not currently maintaning the Documentation for installing the Go bindings for TensorFlow.

The instructions has been maintained by the third party contributor: @wamuir

<<<<<<< HEAD
Please follow this [source](https://github.com/tensorflow/build/tree/master/golang_install_guide) by @wamuir for the installation of Golang with Tensorflow.
=======
-   [bazel](https://www.bazel.build/versions/master/docs/install.html)
-   Environment to build TensorFlow from source code
    ([Linux or macOS](https://www.tensorflow.org/install/source)). If you don't
    need GPU support, then try the following:

    ```sh
    sudo apt-get install python swig python-numpy # Linux
    brew install swig                             # OS X with homebrew
    ```
- [Protocol buffer compiler (protoc) 3.x](https://github.com/google/protobuf/releases/)

### Build

1.  Download the source code

    ```sh
    go get -d github.com/hdu-hh/tensorflow/tensorflow/go
    ```

2.  Build the TensorFlow C library:

    ```sh
    cd ${GOPATH}/src/github.com/hdu-hh/tensorflow
    ./configure
    bazel build -c opt //tensorflow:libtensorflow.so
    ```

    This can take a while (tens of minutes, more if also building for GPU).

3.  Make `libtensorflow.so` and `libtensorflow_framework.so` available to the
    linker. This can be done by either:

    a. Copying it to a system location, e.g.,

    ```sh
    sudo cp ${GOPATH}/src/github.com/hdu-hh/tensorflow/bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
    sudo cp ${GOPATH}/src/github.com/hdu-hh/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib
    ```

    OR

    b. Setting environment variables:

    ```sh
    export LIBRARY_PATH=${GOPATH}/src/github.com/hdu-hh/tensorflow/bazel-bin/tensorflow
    # Linux
    export LD_LIBRARY_PATH=${GOPATH}/src/github.com/hdu-hh/tensorflow/bazel-bin/tensorflow
    # OS X
    export DYLD_LIBRARY_PATH=${GOPATH}/src/github.com/hdu-hh/tensorflow/bazel-bin/tensorflow
    ```

4.  Build and test:

    ```sh
    go generate github.com/hdu-hh/tensorflow/tensorflow/go/op
    go test github.com/hdu-hh/tensorflow/tensorflow/go
    ```

### Generate wrapper functions for ops

Go functions corresponding to TensorFlow operations are generated in `op/wrappers.go`. To regenerate them:

Prerequisites:
- [Protocol buffer compiler (protoc) 3.x](https://github.com/google/protobuf/releases/)
- The TensorFlow repository under GOPATH

```sh
go generate github.com/hdu-hh/tensorflow/tensorflow/go/op
```

## Support

Use [Stack Overflow](http://stackoverflow.com/questions/tagged/tensorflow)
and/or [GitHub issues](https://github.com/tensorflow/tensorflow/issues).

## Contributions

Contributions are welcome. If making any signification changes, probably best to
discuss on a [GitHub issue](https://github.com/tensorflow/tensorflow/issues)
before investing too much time. GitHub pull requests are used for contributions.
>>>>>>> c90400d2248 (redirect upstream references to experimental fork)
