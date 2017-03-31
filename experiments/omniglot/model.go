package main

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/sgdstore"
)

func NewModel(name string, sgdSteps, outCount int) anyrnn.Block {
	c := anyvec32.CurrentCreator()

	normInput := &anyrnn.LayerBlock{
		Layer: anynet.NewAffine(c, 4, -0.92*4),
	}

	switch name {
	case "sgdstore":
		return anyrnn.Stack{
			normInput,
			anyrnn.NewLSTM(c, 400+outCount, 384),
			sgdstore.LinearBlock(c, 384, 6, 2, sgdSteps, 0.2, 128, 256, 128),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 256, 128),
					anynet.Tanh,
					anynet.NewFC(c, 128, outCount),
					anynet.LogSoftmax,
				},
			},
		}
	case "lstm":
		return anyrnn.Stack{
			normInput,
			anyrnn.NewLSTM(c, 400+outCount, 384),
			anyrnn.NewLSTM(c, 384, 384).ScaleInWeights(c.MakeNumeric(2)),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 384, outCount),
					anynet.LogSoftmax,
				},
			},
		}
	default:
		essentials.Die("unknown model:", name)
		panic("unreachable")
	}
}

func CountParams(b anyrnn.Block) int {
	var res int
	for _, p := range anynet.AllParameters(b) {
		res += p.Vector.Len()
	}
	return res
}
