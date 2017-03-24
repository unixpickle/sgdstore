package sgdstore

import (
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Block{}).SerializerType(), DeserializeBlock)
}

// Block is an RNN block that uses a Net as its memory.
type Block struct {
	InitParams []*anydiff.Var

	// Gates which transform the input into various vectors
	// used to train and query the current Net.
	TrainInput  anynet.Layer
	TrainTarget anynet.Layer
	StepSize    anynet.Layer
	Query       anynet.Layer
}

// DeserializeBlock deserializes a Block.
func DeserializeBlock(d []byte) (block *Block, err error) {
	defer essentials.AddCtxTo("deserialize sgdstore.Block", &err)
	var savedVecs []serializer.Serializer
	block = &Block{}
	err = serializer.DeserializeAny(d, &savedVecs, &block.TrainInput, &block.TrainTarget,
		&block.StepSize, &block.Query)
	if err != nil {
		return nil, err
	}
	for _, vecObj := range savedVecs {
		if vec, ok := vecObj.(*anyvecsave.S); ok {
			block.InitParams = append(block.InitParams, anydiff.NewVar(vec.Vector))
		} else {
			return nil, fmt.Errorf("expected vector but got %T", vecObj)
		}
	}
	return
}

// Start produces a start state.
func (b *Block) Start(n int) anyrnn.State {
	res := &State{
		Creator:    b.Parameters()[0].Vector.Creator(),
		ParamSizes: make([]int, len(b.Parameters())),
		Params:     make([][]anyvec.Vector, n),
	}
	for i, p := range b.InitParams {
		res.ParamSizes[i] = p.Vector.Len()
		for j := range res.Params {
			res.Params[j] = append(res.Params[j], p.Vector.Copy())
		}
	}
	return res
}

// PropagateStart propagates through the start state.
func (b *Block) PropagateStart(s anyrnn.StateGrad, g anydiff.Grad) {
	state := s.(*State)
	for paramNum, p := range b.InitParams {
		if grad, ok := g[p]; ok {
			for _, netGrad := range state.Params {
				grad.Add(netGrad[paramNum])
			}
		}
	}
}

// Parameters returns the block's parameters, including
// the parameters of the gates.
func (b *Block) Parameters() []*anydiff.Var {
	gateParams := anynet.AllParameters(b.TrainInput, b.TrainTarget, b.StepSize, b.Query)
	return append(gateParams, b.InitParams...)
}

// SerializerType returns the unique ID used to serialize
// a Block with the serializer package.
func (b *Block) SerializerType() string {
	return "github.com/unixpickle/sgdstore.Block"
}

// Serialize serializes the block.
func (b *Block) Serialize() ([]byte, error) {
	var savedVecs []serializer.Serializer
	for _, v := range b.InitParams {
		savedVecs = append(savedVecs, &anyvecsave.S{Vector: v.Vector})
	}
	return serializer.SerializeAny(
		savedVecs,
		b.TrainInput,
		b.TrainTarget,
		b.StepSize,
		b.Query,
	)
}

// State is the anyrnn.State and anyrnn.StateGrad type for
// a Block.
type State struct {
	Creator    anyvec.Creator
	ParamSizes []int
	Params     [][]anyvec.Vector
}

// Present returns the present sequence map.
func (s *State) Present() anyrnn.PresentMap {
	m := make(anyrnn.PresentMap, len(s.Params))
	for i, x := range s.Params {
		if x != nil {
			m[i] = true
		}
	}
	return m
}

// Reduce removes states.
func (s *State) Reduce(p anyrnn.PresentMap) anyrnn.State {
	reduced := make([][]anyvec.Vector, len(s.Params))
	for i, present := range p {
		if present {
			reduced[i] = s.Params[i]
		}
	}
	return &State{
		Creator:    s.Creator,
		ParamSizes: s.ParamSizes,
		Params:     reduced,
	}
}

// Expand inserts gradients.
func (s *State) Expand(p anyrnn.PresentMap) anyrnn.StateGrad {
	expanded := append([][]anyvec.Vector{}, s.Params...)
	curPres := s.Present()
	for i, present := range p {
		if present && !curPres[i] {
			for _, size := range s.ParamSizes {
				vec := s.Creator.MakeVector(size)
				expanded[i] = append(expanded[i], vec)
			}
		}
	}
	return &State{
		Creator:    s.Creator,
		ParamSizes: s.ParamSizes,
		Params:     expanded,
	}
}
