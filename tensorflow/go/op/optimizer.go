package op

import (
	tf "github.com/hdu-hh/tensorflow/tensorflow/go"
)

type Optimizer interface {
	Step(sess *tf.Session, feeds tf.FeedMap, fetches []tf.Output, targets []*tf.Operation) (fetched []*tf.Tensor)
	getName() string
	getParams() []tf.Output
	getLearnRate() tf.Output
}

type optimizerBase struct {
	name      string
	params    []tf.Output
	losses    []tf.Output
	learnRate tf.Output
	stepOp    *tf.Operation // operation performed in a step: usually a NoOp with control dependencies
}

func (base optimizerBase) getName() string         { return base.name }
func (base optimizerBase) getParams() []tf.Output  { return base.params }
func (base optimizerBase) getLearnRate() tf.Output { return base.learnRate }

func (base *optimizerBase) Step(sess *tf.Session, feeds tf.FeedMap, fetches []tf.Output, targets []*tf.Operation) []*tf.Tensor {
	targets = append(targets, base.stepOp)
	fetched, err := sess.Run(feeds, fetches, targets)
	if err != nil {
		panic(err)
	}
	return fetched
}

// OptimizerSGD is an optimizer with stochastic gradient descend
func OptimizerSGD(s *Scope, losses []tf.Output, learnRate float32, tags ...VarTag) Optimizer {
	params := s.mustGetParams(tags...)
	newGrads := Gradients(s, losses, params)
	// prepare update network
	lrConst := Const(s, learnRate)
	axis0 := Const(s, int32(0))
	maxLrLoss := Mul(s, lrConst, Max(s, Flatten(s, Pack(s, losses)), axis0))
	updateOps := make([]*tf.Operation, len(newGrads))
	for i, newGrad := range newGrads {
		newNorm := Sum(s, Square(s, Flatten(s, newGrad)), axis0)
		newGrad = Mul(s, newGrad, DivNoNan(s, maxLrLoss, newNorm))
		updateOps[i] = AssignSub(s, params[i], newGrad).Op
	}
	stepOp := NoOp(s.WithControlDependencies(updateOps...))
	return &optimizerBase{"SGD", params, losses, lrConst, stepOp}
}

// OptimizerAdam is an optimizer with adaptive momentum
// (based on https://arxiv.org/abs/1412.6980 article)
// Typical values are alpha=0.001, beta1=0.900, beta2=0.999
func OptimizerAdam(s *Scope, losses []tf.Output, learnRate, alpha, beta1, beta2 float32, tags ...VarTag) Optimizer {
	alpha = 1 // alpha forced to 1.0 for now while 1-beta**t terms are ignored
	params := s.mustGetParams(tags...)
	// initialize state variables
	moments1 := make([]tf.Output, len(params))
	moments2 := make([]tf.Output, len(params))
	for i, parm := range params {
		shape, dtype := parm.Shape(), parm.DataType().DeRef()
		moments1[i] = VariableV2(s, shape, dtype)
		moments2[i] = VariableV2(s, shape, dtype)
		s.tagVariable(moments1[i], TagInitZeros)
		s.tagVariable(moments2[i], TagInitZeros)
	}
	// get gradients
	grads := Gradients(s, losses, params)
	// prepare update network
	lrConst := Const(s, learnRate)
	axis0 := Const(s, int32(0))
	maxLrLoss := Mul(s, lrConst, Max(s, Flatten(s, Pack(s, losses)), axis0))
	alpConst := Const(s, float32(alpha))
	b1pConst := Const(s, float32(beta1))
	b1mConst := Const(s, float32(1-beta1))
	b2pConst := Const(s, float32(beta2))
	b2mConst := Const(s, float32(1-beta2))
	epsConst := Const(s, float32(1e-8))
	updateOps := make([]*tf.Operation, 0, 3*len(grads))
	for i, newGrad := range grads {
		newNorm := Sum(s, Square(s, Flatten(s, newGrad)), axis0)
		newGrad = Mul(s, newGrad, DivNoNan(s, maxLrLoss, newNorm))
		newMom1 := Add(s, Mul(s, b1pConst, moments1[i]), Mul(s, b1mConst, newGrad))
		newMom2 := Add(s, Mul(s, b2pConst, moments2[i]), Mul(s, b2mConst, Square(s, newGrad)))
		// TODO: apply 1/(1-beta**t) terms? after a few steps they converge to one anyway...
		newGrad = DivNoNan(s, Mul(s, alpConst, newMom1), Add(s, epsConst, Sqrt(s, newMom2)))
		mulGrad := Mul(s, lrConst, newGrad)
		scd := s.WithControlDependencies(mulGrad.Op)
		a1 := AssignSub(scd, params[i], mulGrad)
		a2 := Assign(scd, moments1[i], newMom1)
		a3 := Assign(scd, moments2[i], newMom2)
		updateOps = append(updateOps, a1.Op, a2.Op, a3.Op)
	}
	stepOp := NoOp(s.WithControlDependencies(updateOps...))
	return &optimizerBase{"Adam", params, losses, lrConst, stepOp}
}

// OptimizerLayla is an optimizer with layer adaptive exponential learning rate adaption
// (based on https://arxiv.org/abs/2309.06274 and extended with layer-specific learning rates)
func OptimizerLayla(s *Scope, losses []tf.Output, tags ...VarTag) Optimizer {
	params := s.mustGetParams(tags...)
	// initialize learning rates
	// TODO? learn rates per layer instead of per param?
	learnRates := VariableV2(s, tf.MakeShape(int64(len(params))), tf.Float)
	minLrConst := Const(s, float32(+1e-9))
	maxLrConst := Const(s, float32(+9e+12))
	lrInitOp := Assign(s, learnRates, Mul(s, minLrConst, OnesLike(s, learnRates)))
	s.tagVariable(lrInitOp, TagInitAssign)
	// initialize state of previous gradients
	oldGrads := make([]tf.Output, len(params))
	for i, parm := range params {
		shape, dtype := parm.Shape(), parm.DataType()
		oldGrads[i] = VariableV2(s, shape, dtype.DeRef())
		s.tagVariable(oldGrads[i], TagInitZeros)
	}
	// get gradients
	newGrads := Gradients(s, losses, params)
	// create the update network
	axis0 := Const(s, int32(0))
	f10Const := Const(s, float32(1.0))
	f05Const := Const(s, float32(0.5))
	updateOps := make([]*tf.Operation, 0, 2*len(params))
	splitLRs := Split(s, axis0, learnRates, int64(len(params)))
	for i, newGrad := range newGrads {
		// calculate cos=(A*B)/(|A||B|) between old and new gradients
		newL2Norm := Sum(s, Square(s, Flatten(s, newGrad)), axis0)
		mixedDot := Sum(s, Flatten(s, Mul(s, oldGrads[i], newGrad)), axis0)
		gradCos := Mul(s, mixedDot, Rsqrt(s, newL2Norm))
		// update learning rate
		newLr := Mul(s, splitLRs[i], Add(s, f10Const, Mul(s, f05Const, gradCos))) // lr *= 1.0 + 0.5*cos
		newLr = ClipByValue(s, newLr, minLrConst, maxLrConst)
		splitLRs[i] = newLr
		// update variables
		newGrad = Mul(s, newGrad, Rsqrt(s, newL2Norm))
		mulGrad := Mul(s, newLr, newGrad)
		scd := s.WithControlDependencies(mulGrad.Op)
		a1 := Assign(scd, oldGrads[i], newGrad)
		a2 := AssignSub(scd, params[i], mulGrad)
		updateOps = append(updateOps, a1.Op, a2.Op)
	}
	scd := s.WithControlDependencies(updateOps...)
	var lrAssign tf.Output
	if len(splitLRs) > 1 {
		lrAssign = Assign(scd, learnRates, ConcatV2(s, splitLRs, axis0))
	} else {
		lrAssign = Assign(scd, learnRates, splitLRs[0])
	}
	stepOp := NoOp(scd.WithControlDependencies(lrAssign.Op))
	return &optimizerBase{"Layla", params, losses, learnRates, stepOp}
}

// weightDecay is an optimizer which help generalization by reducing weights
type weightDecay struct {
	weightDecay optimizerBase
	lossDescend Optimizer
}

// WeightDecay allows adding decoupled weight decay to another optimizer.
// Weight decay helps the network to generalize its learnings.
func WeightDecay(s *Scope, refOpt Optimizer, decayRate float32) Optimizer {
	// find matching reference optimizer parameters
	refParmMap := map[tf.Output]int{}
	for i, p := range refOpt.getParams() {
		refParmMap[p] = i
	}
	paramMap := map[tf.Output]bool{}
	var losses []tf.Output
	axis0 := Const(s, int32(0))
	for _, p := range s.GetParams(TagDecayL1) {
		if _, ok := refParmMap[p]; ok {
			losses = append(losses, Mean(s, Flatten(s, p), axis0))
			paramMap[p] = true
		}
	}
	for _, p := range s.GetParams(TagDecayL2) {
		if _, ok := refParmMap[p]; ok {
			losses = append(losses, L2Loss(s, p))
			paramMap[p] = true
		}
	}
	var params []tf.Output
	for p := range paramMap {
		params = append(params, p)
	}
	// get gradients
	gradients := Gradients(s, losses, params)
	// create the update network
	updateOps := make([]*tf.Operation, len(gradients))
	decayConst := Const(s, decayRate)
	lrCount, splitLRs := 1, []tf.Output{refOpt.getLearnRate()}
	if shape := splitLRs[0].Shape(); shape.NumDimensions() > 0 {
		lrCount = int(shape.Size(-1))
		splitLRs = Split(s, axis0, refOpt.getLearnRate(), int64(lrCount))
	}
	for i, p := range params {
		lrIdx := 0
		if lrIdx >= lrCount {
			lrIdx = 0
		}
		rate := Mul(s, decayConst, splitLRs[lrIdx])
		updateOps[i] = AssignSub(s, p, Mul(s, rate, gradients[i])).Op
	}
	stepOp := NoOp(s.WithControlDependencies(updateOps...))
	return &weightDecay{
		lossDescend: refOpt,
		weightDecay: optimizerBase{"wdecay", params, losses, decayConst, stepOp},
	}
}

func (wd *weightDecay) getName() string         { return wd.lossDescend.getName() + "W" }
func (wd *weightDecay) getLearnRate() tf.Output { return wd.lossDescend.getLearnRate() }
func (wd *weightDecay) getParams() []tf.Output  { return wd.weightDecay.getParams() }

func (wd *weightDecay) Step(sess *tf.Session, feeds tf.FeedMap, fetches []tf.Output, targets []*tf.Operation) (fetched []*tf.Tensor) {
	fetched = wd.lossDescend.Step(sess, feeds, fetches, targets)
	wd.weightDecay.Step(sess, nil, nil, nil)
	return
}

// OptimizerAdamW is an optimizer with adaptive momentum and decoupled weight decay
// (based on https://arxiv.org/abs/1711.05101 article)
func OptimizerAdamW(s *Scope, losses []tf.Output, decayRate, learnRate, alpha, beta1, beta2 float32, tags ...VarTag) Optimizer {
	adam := OptimizerAdam(s.SubScope("adam"), losses, learnRate, alpha, beta1, beta2, tags...)
	return WeightDecay(s.SubScope("wdecay"), adam, decayRate)
}

// OptimizerLaylaW is an optimizer with layer specific exponential learning rates adaptions and decoupled weight decay
// (based on https://arxiv.org/abs/2309.06274 and https://arxiv.org/abs/1711.05101)
func OptimizerLaylaW(s *Scope, losses []tf.Output, decayRate float32, tags ...VarTag) Optimizer {
	layla := OptimizerLayla(s.SubScope("layla"), losses, tags...)
	return WeightDecay(s.SubScope("wdecay"), layla, decayRate)
}
