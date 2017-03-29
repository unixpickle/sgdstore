package sgdstore

import (
	"fmt"

	"github.com/unixpickle/anydiff"
)

// NetBatch is a batch of dynamic feed-forward multi-layer
// perceptrons.
//
// Each layer is implicitly followed by a tanh.
type Net struct {
	// Parameters stores the weights and biases of the
	// network.
	// Each even index corresponds to a batch of weight
	// matrices.
	// Each odd index corresponds to a batch of bias vectors.
	// Matrices are row-major.
	//
	// This should not be empty.
	Parameters anydiff.MultiRes

	// Num is the number of networks in the batch.
	Num int
}

// Apply applies the networks to a batch of input batches,
// producing a batch of output batches.
func (n *Net) Apply(inBatch anydiff.Res, batchSize int) anydiff.Res {
	return anydiff.Unfuse(n.Parameters, func(params []anydiff.Res) anydiff.Res {
		if len(params)%2 != 0 {
			panic("mismatching bias and weight count")
		}
		for i := 0; i < len(params); i += 2 {
			inBatch = applyLayer(params[i], params[i+1], inBatch, batchSize, n.Num)
		}
		return inBatch
	})
}

// InSize calculates the input size of the network using
// the dimensions of the first layer.
//
// This is invariant to n.Num.
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
	if stepSize.Output().Len() != n.Num {
		panic("invalid stepSize length")
	}
	ins := anydiff.Fuse(inBatch, target, stepSize)
	newParams := anydiff.PoolMulti(ins, func(s []anydiff.Res) anydiff.MultiRes {
		inBatch, target, stepSize := s[0], s[1], s[2]
		net := n
		for i := 0; i < numSteps; i++ {
			net = net.step(inBatch, target, stepSize, batchSize)
		}
		return net.Parameters
	})
	return &Net{Parameters: newParams, Num: n.Num}
}

// step performs a step of gradient descent and returns
// the new network.
//
// The input, target, and stepSize should be pooled by the
// caller.
func (n *Net) step(inBatch, target, stepSize anydiff.Res, batchSize int) *Net {
	newParams := anydiff.PoolMulti(n.Parameters, func(params []anydiff.Res) anydiff.MultiRes {
		grad := applyBackprop(params, inBatch, target, batchSize, n.Num)
		return anydiff.PoolMulti(grad, func(grads []anydiff.Res) anydiff.MultiRes {
			var newParams []anydiff.Res
			for i, g := range grads[1:] {
				gMat := &anydiff.Matrix{
					Data: g,
					Rows: n.Num,
					Cols: g.Output().Len() / n.Num,
				}
				p := anydiff.Add(params[i], anydiff.ScaleRows(gMat, stepSize).Data)
				newParams = append(newParams, p)
			}
			return anydiff.Fuse(newParams...)
		})
	})
	return &Net{Parameters: newParams, Num: n.Num}
}

// applyLayer applies a single layer.
func applyLayer(weights, biases, inBatch anydiff.Res, batchSize, numNets int) anydiff.Res {
	inMat, weightMat := layerMats(weights, biases, inBatch, batchSize, numNets)
	inBatch = anydiff.BatchedMatMul(false, true, inMat, weightMat).Data
	return anydiff.Tanh(batchedAddRepeated(inBatch, biases, numNets))
}

// applyBackprop applies the networks and performs
// backward-propagation.
// The result is [inGrad, param1Grad, param2Grad, ...].
// The caller should pool the input parameters.
func applyBackprop(params []anydiff.Res, in, target anydiff.Res,
	batchSize, numNets int) anydiff.MultiRes {
	if len(params) == 0 {
		scaler := target.Output().Creator().MakeNumeric(
			2 / float64(target.Output().Len()/numNets),
		)
		if target.Output().Len() != in.Output().Len() {
			panic(fmt.Sprintf("target length %d (expected %d)", target.Output().Len(),
				in.Output().Len()))
		}
		return anydiff.Fuse(anydiff.Scale(anydiff.Sub(target, in), scaler))
	}
	inMat, weightMat := layerMats(params[0], params[1], in, batchSize, numNets)
	matOut := anydiff.BatchedMatMul(false, true, inMat, weightMat).Data
	biasOut := batchedAddRepeated(matOut, params[1], numNets)
	tanhOut := anydiff.Tanh(biasOut)
	return anydiff.PoolFork(tanhOut, func(tanhOut anydiff.Res) anydiff.MultiRes {
		nextOut := applyBackprop(params[2:], tanhOut, target, batchSize, numNets)
		return anydiff.PoolMulti(nextOut, func(x []anydiff.Res) anydiff.MultiRes {
			outGrad := x[0]
			laterGrads := x[1:]
			pg := anydiff.Mul(anydiff.Complement(anydiff.Square(tanhOut)), outGrad)
			return anydiff.PoolFork(pg, func(pg anydiff.Res) anydiff.MultiRes {
				productGrad := &anydiff.MatrixBatch{
					Data: pg,
					Rows: batchSize,
					Cols: weightMat.Rows,
					Num:  numNets,
				}
				weightGrad := anydiff.BatchedMatMul(true, false, productGrad, inMat).Data
				biasGrad := batchedSumRows(productGrad)
				inGrad := anydiff.BatchedMatMul(false, false, productGrad, weightMat).Data
				ourGrad := []anydiff.Res{inGrad, weightGrad, biasGrad}
				return anydiff.Fuse(append(ourGrad, laterGrads...)...)
			})
		})
	})
}

func layerMats(weights, biases, inBatch anydiff.Res, batchSize, numNets int) (inMat,
	weightMat *anydiff.MatrixBatch) {
	outSize := biases.Output().Len() / numNets
	inSize := weights.Output().Len() / (outSize * numNets)
	if inSize*batchSize*numNets != inBatch.Output().Len() {
		panic(fmt.Sprintf("input size %d should be %d",
			inBatch.Output().Len()/(batchSize*numNets), inSize))
	}
	inMat = &anydiff.MatrixBatch{
		Data: inBatch,
		Rows: batchSize,
		Cols: inSize,
		Num:  numNets,
	}
	weightMat = &anydiff.MatrixBatch{
		Data: weights,
		Rows: outSize,
		Cols: inSize,
		Num:  numNets,
	}
	return
}

func batchedAddRepeated(vec, biases anydiff.Res, n int) anydiff.Res {
	return anydiff.Pool(vec, func(vec anydiff.Res) anydiff.Res {
		return anydiff.Pool(biases, func(biases anydiff.Res) anydiff.Res {
			biasVecs := splitVec(biases, n)
			var res []anydiff.Res
			for i, v := range splitVec(vec, n) {
				b := biasVecs[i]
				res = append(res, anydiff.AddRepeated(v, b))
			}
			return anydiff.Concat(res...)
		})
	})
}

func batchedSumRows(m *anydiff.MatrixBatch) anydiff.Res {
	return anydiff.Pool(m.Data, func(data anydiff.Res) anydiff.Res {
		var sums []anydiff.Res
		for _, matData := range splitVec(data, m.Num) {
			matrix := &anydiff.Matrix{Data: matData, Rows: m.Rows, Cols: m.Cols}
			sums = append(sums, anydiff.SumRows(matrix))
		}
		return anydiff.Concat(sums...)
	})
}

func splitVec(vec anydiff.Res, n int) []anydiff.Res {
	chunkSize := vec.Output().Len() / n
	var chunks []anydiff.Res
	for i := 0; i < n; i++ {
		chunks = append(chunks, anydiff.Slice(vec, i*chunkSize, (i+1)*chunkSize))
	}
	return chunks
}
