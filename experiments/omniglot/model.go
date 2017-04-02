package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/sgdstore"
)

func NewModel(name string, sgdSteps, outCount int) anyrnn.Block {
	c := anyvec32.CurrentCreator()
	numPixels := ImageSize * ImageSize

	switch name {
	case "sgdstore":
		return anyrnn.Stack{
			normInputLayer(c, outCount, numPixels),
			anyrnn.NewVanilla(c, numPixels+outCount, 384, anynet.Tanh),
			anyrnn.NewVanilla(c, 384, 384, anynet.Tanh),
			sgdstore.LinearBlock(c, 384, 16, 2, sgdSteps, 0.2, 32, 256, 32),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 64, 64),
					anynet.Tanh,
					anynet.NewFC(c, 64, outCount),
					anynet.LogSoftmax,
				},
			},
		}
	case "parasgdstore":
		return anyrnn.Stack{
			normInputLayer(c, outCount, numPixels),
			anyrnn.NewVanilla(c, numPixels+outCount, 384, anynet.Tanh),
			anyrnn.NewVanilla(c, 384, 384, anynet.Tanh),
			&anyrnn.Parallel{
				Block1: &anyrnn.LayerBlock{Layer: anynet.Net{}},
				Block2: sgdstore.LinearBlock(c, 384, 16, 2, sgdSteps, 0.2, 32, 256, 32),
				Mixer: &anynet.AddMixer{
					In1: anynet.NewFC(c, 384, 64),
					In2: anynet.NewFC(c, 64, 64),
					Out: anynet.Tanh,
				},
			},
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 64, outCount),
					anynet.LogSoftmax,
				},
			},
		}
	case "lstm":
		return anyrnn.Stack{
			normInputLayer(c, outCount, numPixels),
			anyrnn.NewLSTM(c, numPixels+outCount, 384),
			anyrnn.NewLSTM(c, 384, 384).ScaleInWeights(c.MakeNumeric(2)),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 384, outCount),
					anynet.LogSoftmax,
				},
			},
		}
	case "vanilla":
		return anyrnn.Stack{
			normInputLayer(c, outCount, numPixels),
			anyrnn.NewVanilla(c, numPixels+outCount, 384, anynet.Tanh),
			anyrnn.NewVanilla(c, 384, 384, anynet.Tanh),
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

func normInputLayer(c anyvec.Creator, numOut, numPixels int) anyrnn.Block {
	affine := &anynet.Affine{
		Scalers: anydiff.NewVar(c.MakeVector(numPixels + numOut)),
		Biases:  anydiff.NewVar(c.MakeVector(numPixels + numOut)),
	}
	affine.Scalers.Vector.AddScaler(c.MakeNumeric(4))

	modified := affine.Scalers.Vector.Slice(numPixels, numPixels+numOut)
	modified.Scale(c.MakeNumeric(4))
	affine.Scalers.Vector.SetSlice(numPixels, modified)

	modified = affine.Biases.Vector.Slice(0, numPixels)
	modified.AddScaler(c.MakeNumeric(-4 * 0.92))
	affine.Biases.Vector.SetSlice(0, modified)

	return &anyrnn.LayerBlock{Layer: affine}
}
