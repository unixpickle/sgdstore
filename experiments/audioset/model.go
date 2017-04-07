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

func learnerBlock(name string, sgdSteps, numFeatures, numOut int) anyrnn.Block {
	c := anyvec32.CurrentCreator()

	inLayer := normInputLayer(c, numFeatures, numOut)

	switch name {
	case "sgdstore":
		return anyrnn.Stack{
			inLayer,
			anyrnn.NewVanilla(c, numFeatures+numOut, 384, anynet.Tanh),
			anyrnn.NewVanilla(c, 384, 384, anynet.Tanh),
			sgdstore.LinearBlock(c, 384, 16, 2, sgdSteps, 0.2, 32, 256, 32),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 64, 64),
					anynet.Tanh,
					anynet.NewFC(c, 64, numOut),
					anynet.LogSoftmax,
				},
			},
		}
	case "lstm":
		return anyrnn.Stack{
			inLayer,
			anyrnn.NewLSTM(c, numFeatures+numOut, 384),
			anyrnn.NewLSTM(c, 384, 384),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 384, numOut),
					anynet.LogSoftmax,
				},
			},
		}
	case "vanilla":
		return anyrnn.Stack{
			inLayer,
			anyrnn.NewVanilla(c, numFeatures+numOut, 384, anynet.Tanh),
			anyrnn.NewVanilla(c, 384, 384, anynet.Tanh),
			&anyrnn.LayerBlock{
				Layer: anynet.Net{
					anynet.NewFC(c, 384, numOut),
					anynet.LogSoftmax,
				},
			},
		}
	default:
		essentials.Die("unknown model:", name)
		panic("unreachable")
	}
}

func featureBlock(numFeatures, blockSize int) anyrnn.Block {
	c := anyvec32.CurrentCreator()
	return anyrnn.NewLSTM(c, blockSize, numFeatures).ScaleInWeights(c.MakeNumeric(7))
}

func normInputLayer(c anyvec.Creator, numFeatures, numOut int) anyrnn.Block {
	affine := &anynet.Affine{
		Scalers: anydiff.NewVar(c.MakeVector(numFeatures + numOut)),
		Biases:  anydiff.NewVar(c.MakeVector(numFeatures + numOut)),
	}

	// Scaling the one-hot vector for the last timestep
	// tends to improve performance.
	outScale := affine.Scalers.Vector.Slice(numFeatures, numFeatures+numOut)
	outScale.Scale(c.MakeNumeric(16))

	return &anyrnn.LayerBlock{Layer: affine}
}
