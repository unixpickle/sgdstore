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
	// Parameters stores the weights and biases of the
	// network.
	// Each even index corresponds to a weight matrix.
	// Each odd index corresponds to a bias vector.
	// Matrices are row-major.
	Parameters anydiff.MultiRes
}

// Apply applies the network to an input batch, producing
// an output batch.
func (n *Net) Apply(inBatch anydiff.Res, batchSize int) anydiff.Res {
	return anydiff.Unfuse(n.Parameters, func(params []anydiff.Res) anydiff.Res {
		if len(params)%2 != 0 {
			panic("mismatching bias and weight count")
		}
		for i := 0; i < len(params); i += 2 {
			weights := params[i]
			biases := params[i+1]
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
	})
}
