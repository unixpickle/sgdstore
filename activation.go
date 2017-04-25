package sgdstore

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
)

// Activation is an activation function.
type Activation int

// Supported activation functions.
const (
	Tanh Activation = iota
	ReLU
)

// Forward applies the activation function in the forward
// direction.
func (a Activation) Forward(in anydiff.Res) anydiff.Res {
	switch a {
	case Tanh:
		return anydiff.Tanh(in)
	case ReLU:
		return anydiff.ClipPos(in)
	}
	panic("unsupported activation")
}

// Backward applies backward propagation, given the output
// from the forward pass and the upstream vector.
func (a Activation) Backward(out, upstream anydiff.Res) anydiff.Res {
	switch a {
	case Tanh:
		return anydiff.Mul(anydiff.Complement(anydiff.Square(out)), upstream)
	case ReLU:
		mask := out.Output().Copy()
		anyvec.GreaterThan(mask, mask.Creator().MakeNumeric(0))
		return anydiff.Mul(upstream, anydiff.NewConst(mask))
	}
	panic("unsupported activation")
}

// Layer returns a compatible anynet.Layer.
func (a Activation) Layer() anynet.Layer {
	switch a {
	case Tanh:
		return anynet.Tanh
	case ReLU:
		return anynet.ReLU
	}
	panic("unsupported activation")
}
