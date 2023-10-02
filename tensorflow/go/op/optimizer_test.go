package op

import (
	"testing"

	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

func TestOptimizers(t *testing.T) {
	const B, W = 32, 16
	var ACT ActFunc = Identity
	var (
		s          = NewScope()
		x0         = RandomUniform(s, Const(s, []int32{B, W}), tf.Float)
		y0         = Mul(s, x0, Const(s, float32(1e3)))
		x1         = MLP(s, y0, W, ACT, TagInitXavierUniform)
		x2         = MLP(s, x1, W, ACT, TagInitXavierUniform)
		x3         = MLP(s, x2, W, ACT, TagInitXavierUniform)
		y1         = MLP(s, y0, W, ACT)
		y2         = MLP(s, y1, W, ACT)
		y3         = MLP(s, y2, W, ACT)
		axis0      = Const(s, int32(0))
		diff       = Flatten(s, Sub(s, x3, y3))
		losses     = []tf.Output{Mean(s, Square(s, diff), axis0)}
		optiSGD    = OptimizerSGD(s, losses, 5e-3)
		optiAdam   = OptimizerAdam(s, losses, 5e-4, 1e-3, 0.9, 0.999)
		optiAdamW  = OptimizerAdamW(s, losses, 1e-2, 5e-4, 1e-3, 0.9, 0.999)
		optiLayla  = OptimizerLayla(s, losses)
		optiLaylaW = OptimizerLaylaW(s, losses, 1e-2)
		initOp     = s.GetInitOp()
		graph, _   = s.Finalize()
		sess, _    = tf.NewSession(graph, nil)
	)
	for _, opti := range []Optimizer{optiSGD, optiAdam, optiAdamW, optiLayla, optiLaylaW} {
		t.Run(opti.getName(), func(t *testing.T) {
			sess.Run(nil, nil, []*tf.Operation{initOp})
			fetched, _ := sess.Run(nil, losses, nil)
			firstLoss := fetched[0].Value().(float32)
			fetches := append(losses, opti.getLearnRate())
			for step, loops := 0, int(1e3); step <= loops; step++ {
				f := opti.Step(sess, nil, fetches, nil)
				if step%(loops/10) == 0 {
					t.Logf("step=%v, loss=%v, lr=%v\n", step, f[0].Value(), f[1].Value())
				}
			}
			fetched, _ = sess.Run(nil, losses, nil)
			if finalLoss := fetched[0].Value().(float32); finalLoss >= 1e-2*firstLoss {
				t.Errorf("loss only improved by factor %.1f (from %.3f to %.3f)",
					firstLoss/finalLoss, firstLoss, finalLoss)
			}
		})
	}
}
