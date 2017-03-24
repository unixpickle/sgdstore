package sgdstore

import (
	"fmt"

	"github.com/unixpickle/anydiff"
)

// Net is a dynamic feed-forward multi-layer perceptron.
//
// Each layer is implicitly followed by a tanh.
type Net struct {
	// Parameters stores the weights and biases of the
	// network.
	// Each even index corresponds to a weight matrix.
	// Each odd index corresponds to a bias vector.
	// Matrices are row-major.
	//
	// This should not be empty.
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
			inBatch = applyLayer(params[i], params[i+1], inBatch, batchSize)
		}
		return inBatch
	})
}

// InSize calculates the input size of the network using
// the dimensions of the first layer.
func (n *Net) InSize() int {
	if len(n.Parameters.Outputs()) < 2 {
		panic("network cannot be empty")
	}
	return n.Parameters.Outputs()[0].Len() / n.Parameters.Outputs()[1].Len()
}

// Train performs SGD training on the batch.
//
// The input, target, and stepSize needn't be pooled by
// the caller.
func (n *Net) Train(inBatch, target, stepSize anydiff.Res, batchSize,
	numSteps int) *Net {
	ins := anydiff.Fuse(inBatch, target, stepSize)
	newParams := anydiff.PoolMulti(ins, func(s []anydiff.Res) anydiff.MultiRes {
		inBatch, target, stepSize := s[0], s[1], s[2]
		net := n
		for i := 0; i < numSteps; i++ {
			net = net.step(inBatch, target, stepSize, batchSize)
		}
		return net.Parameters
	})
	return &Net{Parameters: newParams}
}

// step performs a step of gradient descent and returns
// the new network.
//
// The input, target, and stepSize should be pooled by the
// caller.
func (n *Net) step(inBatch, target, stepSize anydiff.Res, batchSize int) *Net {
	newParams := anydiff.PoolMulti(n.Parameters, func(params []anydiff.Res) anydiff.MultiRes {
		grad := applyBackprop(params, inBatch, target, batchSize)
		return anydiff.PoolMulti(grad, func(grads []anydiff.Res) anydiff.MultiRes {
			var newParams []anydiff.Res
			for i, g := range grads[1:] {
				p := anydiff.Add(params[i], anydiff.ScaleRepeated(g, stepSize))
				newParams = append(newParams, p)
			}
			return anydiff.Fuse(newParams...)
		})
	})
	return &Net{Parameters: newParams}
}

// applyLayer applies a single layer.
func applyLayer(weights, biases, inBatch anydiff.Res, batchSize int) anydiff.Res {
	outSize := biases.Output().Len()
	inSize := weights.Output().Len() / outSize
	if inSize*batchSize != inBatch.Output().Len() {
		panic(fmt.Sprintf("input size %d should be %d",
			inBatch.Output().Len()/batchSize, inSize))
	}
	inMat := &anydiff.Matrix{Data: inBatch, Rows: batchSize, Cols: inSize}
	weightMat := &anydiff.Matrix{Data: weights, Rows: outSize, Cols: inSize}
	inBatch = anydiff.MatMul(false, true, inMat, weightMat).Data
	return anydiff.Tanh(anydiff.AddRepeated(inBatch, biases))
}

// applyBackprop applies the network and performs
// backward-propagation.
// The result is [inGrad, param1Grad, param2Grad, ...].
// The caller should pool the input parameters.
func applyBackprop(params []anydiff.Res, in, target anydiff.Res,
	batchSize int) anydiff.MultiRes {
	if len(params) == 0 {
		scaler := target.Output().Creator().MakeNumeric(
			2 / float64(target.Output().Len()),
		)
		if target.Output().Len() != in.Output().Len() {
			panic(fmt.Sprintf("target length %d (expected %d)", target.Output().Len(),
				in.Output().Len()))
		}
		return anydiff.Fuse(anydiff.Scale(anydiff.Sub(target, in), scaler))
	}
	weights, biases := params[0], params[1]
	outSize := biases.Output().Len()
	inSize := weights.Output().Len() / outSize
	if inSize*batchSize != in.Output().Len() {
		panic(fmt.Sprintf("input size %d should be %d",
			in.Output().Len()/batchSize, inSize))
	}
	inMat := &anydiff.Matrix{Data: in, Rows: batchSize, Cols: inSize}
	weightMat := &anydiff.Matrix{Data: weights, Rows: outSize, Cols: inSize}
	matOut := anydiff.MatMul(false, true, inMat, weightMat).Data
	biasOut := anydiff.AddRepeated(matOut, biases)
	tanhOut := anydiff.Tanh(biasOut)
	return anydiff.PoolFork(tanhOut, func(tanhOut anydiff.Res) anydiff.MultiRes {
		nextOut := applyBackprop(params[2:], tanhOut, target, batchSize)
		return anydiff.PoolMulti(nextOut, func(x []anydiff.Res) anydiff.MultiRes {
			outGrad := x[0]
			laterGrads := x[1:]
			pg := anydiff.Mul(anydiff.Complement(anydiff.Square(tanhOut)), outGrad)
			return anydiff.PoolFork(pg, func(pg anydiff.Res) anydiff.MultiRes {
				productGrad := &anydiff.Matrix{
					Data: pg,
					Rows: batchSize,
					Cols: outSize,
				}
				weightGrad := anydiff.MatMul(true, false, productGrad, inMat).Data
				biasGrad := anydiff.SumRows(productGrad)
				inGrad := anydiff.MatMul(false, false, productGrad, weightMat).Data
				ourGrad := []anydiff.Res{inGrad, weightGrad, biasGrad}
				return anydiff.Fuse(append(ourGrad, laterGrads...)...)
			})
		})
	})
}
