package op

import (
	"fmt"
	"math"

	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

const (
	// initializer tags
	TagInitAssign        VarTag = "TagInitAssign"
	TagInitZeros         VarTag = "TagInitZeros"
	TagInitOnes          VarTag = "TagInitOnes"
	TagInitUniform       VarTag = "TagInitUniform"       // limits=0...1
	TagInitEpsUniform    VarTag = "TagInitEpsNormal"     // mean=0, stddev=1e-4
	TagInitTruncNormal   VarTag = "TagInitTruncNormal"   // mean=0, stddev=1
	TagInitHeUniform     VarTag = "TagInitHeUniform"     // +-limit=sqrt(2/fan_in)
	TagInitHeNormal      VarTag = "TagInitHeNormal"      // mean=0, stddev=sqrt(2/fan_in), +-limit=2*stddev
	TagInitLecunUniform  VarTag = "TagInitLecunUniform"  // +-limit=sqrt(3/fan_in)
	TagInitLecunNormal   VarTag = "TagInitLecunNormal"   // mean=0, stddev=sqrt(1/fan_in)
	TagInitXavierUniform VarTag = "TagInitXavierUniform" // +-limit = sqrt(6 / (fan_in + fan_out))
	TagInitXavierNormal  VarTag = "TagInitXavierNormal"  // mean=0, stddev=sqrt(2/(fan_in+fan_out))
)

func (s *Scope) GetInitOp() *tf.Operation {
	var allInitOps []*tf.Operation
	for _, tag := range []VarTag{
		TagInitZeros, TagInitOnes,
		TagInitUniform, TagInitEpsUniform, TagInitTruncNormal,
		TagInitHeUniform, TagInitHeNormal,
		TagInitLecunUniform, TagInitLecunNormal,
		TagInitXavierUniform, TagInitXavierNormal} {
		for _, oneOp := range (*s.outTagMap)[tag] {
			allInitOps = append(allInitOps, addInitOp(s, oneOp, tag))
		}
	}
	for _, oneOp := range (*s.outTagMap)[TagInitAssign] {
		allInitOps = append(allInitOps, oneOp.Op)
	}
	return NoOp(s.WithControlDependencies(allInitOps...))
}

func addInitOp(s *Scope, x tf.Output, tag VarTag) *tf.Operation {
	xType, err := x.Op.Attr("dtype")
	if err != nil {
		panic(err)
	}
	dtype := xType.(tf.DataType)
	xShape, err := x.Op.Attr("shape")
	if err != nil {
		panic(err)
	}
	shape := xShape.(tf.Shape)
	shapeConst := Const(s, shape.MustSlice32())
	var y tf.Output
	switch tag {
	case TagInitZeros:
		y = ZerosLike(s, Empty(s, shapeConst, dtype))
	case TagInitOnes:
		y = OnesLike(s, Empty(s, shapeConst, dtype))
	case TagInitUniform:
		y = RandomUniform(s, shapeConst, dtype)
	case TagInitEpsUniform:
		y = RandomUniform(s, shapeConst, dtype)
		y = Mul(s, y, Const(s, float32(1e-4)))
	case TagInitTruncNormal:
		y = TruncatedNormal(s, shapeConst, dtype)
	case TagInitHeUniform:
		f := math.Sqrt2 / float32(shape.Size(0))
		y = RandomUniform(s, shapeConst, dtype)
		y = Mul(s, y, Const(s, f))
	case TagInitHeNormal:
		f := math.Sqrt2 / float32(shape.Size(0))
		avg, std, low, high := Const(s, float32(0)), Const(s, f), Const(s, -2*f), Const(s, +2*f)
		y = ParameterizedTruncatedNormal(s, shapeConst, avg, std, low, high)
	case TagInitLecunUniform:
		f := 1.73 / float32(shape.Size(0))
		y = RandomUniform(s, shapeConst, dtype)
		y = Mul(s, y, Const(s, f))
	case TagInitLecunNormal:
		f := 1 / float32(shape.Size(0))
		y = TruncatedNormal(s, shapeConst, dtype) // std LeCun is not truncated
		y = Mul(s, y, Const(s, f))
	case TagInitXavierUniform:
		f := 2.45 / float32(shape.Size(0)+shape.Size(-1))
		y = RandomUniform(s, shapeConst, dtype)
		y = Mul(s, y, Const(s, f))
	case TagInitXavierNormal:
		f := math.Sqrt2 / float32(shape.Size(0)+shape.Size(-1))
		y = TruncatedNormal(s, shapeConst, dtype) // std Xavier is not truncated
		y = Mul(s, y, Const(s, f))
	default:
		panic(fmt.Errorf("init tag %q not implemented yet", tag))
	}
	return Assign(s, x, y).Op
}
