package sgdstore

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestNetApply(t *testing.T) {
	c := anyvec64.CurrentCreator()
	realNet := anynet.Net{
		anynet.NewFC(c, 3, 5),
		anynet.Tanh,
		anynet.NewFC(c, 5, 4),
		anynet.Tanh,
		anynet.NewFC(c, 4, 2),
	}

	var netParams []anydiff.Res
	for i := 0; i < len(realNet); i += 2 {
		fc := realNet[i].(*anynet.FC)
		anyvec.Rand(fc.Biases.Vector, anyvec.Normal, nil)
		netParams = append(netParams, fc.Weights, fc.Biases)
	}
	virtualNet := &Net{Parameters: anydiff.Fuse(netParams...)}

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
