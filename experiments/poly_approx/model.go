package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/sgdstore"
)

func NewModel(name string, sgdSteps int) anyrnn.Block {
	c := anyvec32.CurrentCreator()
	switch name {
	case "sgdstore":
		return anyrnn.Stack{
			&anyrnn.Feedback{
				InitOut: anydiff.NewVar(c.MakeVector(32)),
				Mixer:   anynet.ConcatMixer{},
				Block: anyrnn.Stack{
					&anyrnn.LayerBlock{
						Layer: anynet.Net{
							anynet.NewFC(c, 4+32, 32),
							anynet.Tanh,
							anynet.NewFC(c, 32, 32),
							anynet.Tanh,
						},
					},
					sgdstore.LinearBlock(c, 32, 2, 2, sgdSteps, 16, 32, 16),
				},
			},
			&anyrnn.LayerBlock{Layer: anynet.NewFC(c, 32, 1)},
		}
	case "lstm":
		return anyrnn.Stack{
			anyrnn.NewLSTM(c, 4, 64),
			anyrnn.NewLSTM(c, 64, 64),
			&anyrnn.LayerBlock{Layer: anynet.NewFC(c, 64, 1)},
		}
	default:
		essentials.Die("unknown model:", name)
		panic("unreachable")
	}
}
