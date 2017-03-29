package sgdstore

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestNetApply(t *testing.T) {
	c := anyvec64.CurrentCreator()
	realNet, virtualNet := randomNetwork(c)

	inVec := c.MakeVector(12)
	anyvec.Rand(inVec, anyvec.Normal, nil)
	input := anydiff.NewVar(inVec)

	expectedOut := realNet.Apply(input, 4).Output()
	actualOut := virtualNet.Apply(input, 4).Output()

	diff := actualOut.Copy()
	diff.Sub(expectedOut)
	maxDiff := anyvec.AbsMax(diff).(float64)

	if maxDiff > 1e-4 {
		t.Errorf("expected %v but got %v", expectedOut.Data(), actualOut.Data())
	}
}

func TestNetTrain(t *testing.T) {
	c := anyvec64.CurrentCreator()
	realNet, virtualNet := randomNetwork(c)

	inVec := c.MakeVector(12)
	anyvec.Rand(inVec, anyvec.Normal, nil)
	input := anydiff.NewVar(inVec)

	targetVec := c.MakeVector(8)
	anyvec.Rand(targetVec, anyvec.Normal, nil)
	target := anydiff.NewVar(targetVec)

	stepSize := c.MakeVector(1)
	stepSize.AddScaler(c.MakeNumeric(0.1))

	trained := virtualNet.Train(input, target, anydiff.NewConst(stepSize), 4, 2)
	actual := trained.Parameters.Outputs()

	for i := 0; i < 2; i++ {
		out := realNet.Apply(input, 4)
		cost := anynet.MSE{}.Cost(target, out, 1)
		grad := anydiff.NewGrad(realNet.Parameters()...)
		cost.Propagate(stepSize.Copy(), grad)
		grad.Scale(c.MakeNumeric(-1))
		grad.AddToVars()
	}
	for i, a := range actual {
		x := realNet.Parameters()[i].Vector
		diff := x.Copy()
		diff.Sub(a)
		maxDiff := anyvec.AbsMax(diff).(float64)
		if maxDiff > 1e-4 {
			t.Error("bad value for layer", i)
		}
	}
}

func BenchmarkNetwork(b *testing.B) {
	c := anyvec32.CurrentCreator()
	realNet := anynet.Net{
		anynet.NewFC(c, 128, 256),
		anynet.NewFC(c, 256, 128),
	}
	var netParams []anydiff.Res
	for i, param := range realNet.Parameters() {
		if i%2 == 0 {
			anyvec.Rand(param.Vector, anyvec.Normal, nil)
		}
		netParams = append(netParams, param)
	}
	net := &Net{Parameters: anydiff.Fuse(netParams...)}

	inBatch := anydiff.NewVar(c.MakeVector(512))
	target := anydiff.NewVar(c.MakeVector(512))
	anyvec.Rand(inBatch.Vector, anyvec.Normal, nil)
	anyvec.Rand(target.Vector, anyvec.Normal, nil)

	stepSize := anydiff.NewConst(c.MakeVector(1))
	stepSize.Vector.AddScaler(float32(0.1))

	b.Run("Forward", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			net.Train(inBatch, target, stepSize, 4, 1)
		}
	})
	b.Run("Backward", func(b *testing.B) {
		grad := anydiff.NewGrad(append([]*anydiff.Var{inBatch, target},
			realNet.Parameters()...)...)
		upstream := make([]anyvec.Vector, len(netParams))
		for i, p := range netParams {
			upstream[i] = c.MakeVector(p.Output().Len())
			anyvec.Rand(upstream[i], anyvec.Normal, nil)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out := net.Train(inBatch, target, stepSize, 4, 1)
			out.Parameters.Propagate(upstream, grad)
		}
	})
}

func randomNetwork(c anyvec.Creator) (anynet.Net, *Net) {
	realNet := anynet.Net{
		anynet.NewFC(c, 3, 5),
		anynet.Tanh,
		anynet.NewFC(c, 5, 4),
		anynet.Tanh,
		anynet.NewFC(c, 4, 2),
		anynet.Tanh,
	}
	var netParams []anydiff.Res
	for i, param := range realNet.Parameters() {
		if i%2 == 0 {
			anyvec.Rand(param.Vector, anyvec.Normal, nil)
		}
		netParams = append(netParams, param)
	}
	return realNet, &Net{Parameters: anydiff.Fuse(netParams...)}
}
