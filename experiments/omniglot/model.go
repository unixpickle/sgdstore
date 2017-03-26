package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/sgdstore"
)

func NewModel(name string, sgdSteps, outCount int) anyrnn.Block {
	c := anyvec32.CurrentCreator()

	inNet, err := anyconv.FromMarkup(c, `
		Input(w=20, h=20, d=1)
		Conv(w=3, h=3, n=8, sx=2, sy=2)
		BatchNorm
		ReLU
		Conv(w=3, h=3, n=16, sx=2, sy=2)
		BatchNorm
		ReLU
		FC(out=128)
		Tanh
	`)
	if err != nil {
		essentials.Die(err)
	}

	switch name {
	case "sgdstore":
		return anyrnn.Stack{
			&anyrnn.LayerBlock{Layer: inNet.(anynet.Layer)},
			&anyrnn.Feedback{
				InitOut: anydiff.NewVar(c.MakeVector(512)),
				Mixer:   anynet.ConcatMixer{},
				Block: anyrnn.Stack{
					&anyrnn.LayerBlock{
						Layer: anynet.Net{
							anynet.NewFC(c, 128+512, 512),
							anynet.Tanh,
							anynet.NewFC(c, 512, 512),
							anynet.Tanh,
						},
					},
					sgdstore.LinearBlock(c, 512, 4, 4, sgdSteps, 128, 256, 128),
				},
			},
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 512, outCount),
					anynet.LogSoftmax,
				},
			},
		}
	case "lstm":
		return anyrnn.Stack{
			&anyrnn.LayerBlock{Layer: inNet.(anynet.Layer)},
			anyrnn.NewLSTM(c, 128, 512),
			anyrnn.NewLSTM(c, 512, 512),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 512, outCount),
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
