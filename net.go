package sgdstore

import (
	"fmt"

	"github.com/unixpickle/anydiff"
)

// Net is a dynamic feed-forward multi-layer perceptron.
//
// Each layer, except for the output layer, is implicitly
// followed by a tanh activation.
type Net struct {
	// Weight matrices for each layer.
	// Each matrix is row-major.
	Weights []anydiff.Res

	// Biases for each layer.
	Biases []anydiff.Res
}

// Apply applies the network to an input batch, producing
// an output batch.
func (n *Net) Apply(inBatch anydiff.Res, batchSize int) anydiff.Res {
	for i, weights := range n.Weights {
		biases := n.Biases[i]
		if i > 0 {
			inBatch = anydiff.Tanh(inBatch)
		}
		outSize := biases.Output().Len()
		inSize := weights.Output().Len() / outSize
		if inSize*batchSize != inBatch.Output().Len() {
			panic(fmt.Sprintf("layer %d: input size %d should be %d", i,
				inBatch.Output().Len()/batchSize, inSize))
		}
		inMat := &anydiff.Matrix{Data: inBatch, Rows: batchSize, Cols: inSize}
		weightMat := &anydiff.Matrix{Data: weights, Rows: outSize, Cols: inSize}
		inBatch = anydiff.MatMul(false, true, inMat, weightMat).Data
		inBatch = anydiff.AddRepeated(inBatch, biases)
	}
	return inBatch
}
