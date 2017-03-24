package sgdstore

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestBlockGradients(t *testing.T) {
	c := anyvec64.CurrentCreator()
	inSeq, inVars := randomTestSequence(3)
	block := &Block{
		InitParams: []*anydiff.Var{
			anydiff.NewVar(anyvec64.MakeVector(4 * 2)),
			anydiff.NewVar(anyvec64.MakeVector(2)),
		},
		TrainInput: anynet.NewFC(c, 3, 4),
		TrainTarget: anynet.Net{
			anynet.NewFC(c, 3, 4),
			anynet.Tanh,
		},
		StepSize: anynet.Net{
			anynet.NewFC(c, 3, 1),
			anynet.Exp,
		},
		Query: anynet.NewFC(c, 3, 4),
	}
	if len(block.Parameters()) != 10 {
		t.Errorf("expected 10 parameters, but got %d", len(block.Parameters()))
	}
	for _, param := range block.Parameters() {
		anyvec.Rand(param.Vector, anyvec.Normal, nil)
	}
	checker := &anydifftest.SeqChecker{
		F: func() anyseq.Seq {
			return anyrnn.Map(inSeq, block)
		},
		V: append(inVars, block.Parameters()...),
	}
	checker.FullCheck(t)
}

// randomTestSequence is borrowed from
// https://github.com/unixpickle/anynet/blob/6a8bd570b702861f3c1260a6916723beea6bf296/anyrnn/layer_test.go#L34
func randomTestSequence(inSize int) (anyseq.Seq, []*anydiff.Var) {
	inVars := []*anydiff.Var{}
	inBatches := []*anyseq.ResBatch{}

	presents := [][]bool{{true, true, true}, {true, false, true}}
	numPres := []int{3, 2}
	chunkLengths := []int{2, 3}

	for chunkIdx, pres := range presents {
		for i := 0; i < chunkLengths[chunkIdx]; i++ {
			vec := anyvec64.MakeVector(inSize * numPres[chunkIdx])
			anyvec.Rand(vec, anyvec.Normal, nil)
			v := anydiff.NewVar(vec)
			batch := &anyseq.ResBatch{
				Packed:  v,
				Present: pres,
			}
			inVars = append(inVars, v)
			inBatches = append(inBatches, batch)
		}
	}
	return anyseq.ResSeq(anyvec64.CurrentCreator(), inBatches), inVars
}
