package sgdstore

import (
	"fmt"
	"math"

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

	// Steps is the number of SGD steps to take at each
	// timestep.
	Steps int
}

// LinearBlock creates a Block with linear gates.
//
// The blockIn argument specifies the input size for the
// block.
//
// The trainBatch and queryBatch arguments specify the
// batch sizes to use for training and querying,
// respectively.
//
// The numSteps argument specifies the number of training
// steps to take at each timestep.
// The lrBias argument specifies the approximate initial
// learning rate (i.e. step size).
//
// The layerSizes specify the sizes of the layers at every
// point in the network.
// The first layer corresponds to the network's input, and
// the last to the network's output.
// Thus, there must be at least two layer sizes.
//
// The block's output size can be computed as
//
//     queryBatch * layerSizes[len(layerSizes)-1]
//
func LinearBlock(c anyvec.Creator, blockIn, trainBatch, queryBatch, numSteps int,
	lrBias float64, layerSizes ...int) *Block {
	if len(layerSizes) < 2 {
		panic("not enough layer sizes")
	} else if trainBatch < 1 || queryBatch < 1 {
		panic("invalid batch size")
	}

	res := &Block{
		TrainInput: anynet.NewFC(c, blockIn, trainBatch*layerSizes[0]),
		TrainTarget: anynet.Net{
			anynet.NewFC(c, blockIn, trainBatch*layerSizes[len(layerSizes)-1]),
			anynet.Tanh,
		},
		StepSize: anynet.Net{
			anynet.NewFC(c, blockIn, 1).AddBias(c.MakeNumeric(math.Log(lrBias))),
			anynet.Exp,
		},
		Query: anynet.NewFC(c, blockIn, queryBatch*layerSizes[0]),
		Steps: numSteps,
	}

	layerSize := layerSizes[0]
	for _, size := range layerSizes[1:] {
		fc := anynet.NewFC(c, layerSize, size)
		layerSize = size
		res.InitParams = append(res.InitParams, fc.Parameters()...)
	}

	return res
}

// DeserializeBlock deserializes a Block.
func DeserializeBlock(d []byte) (block *Block, err error) {
	defer essentials.AddCtxTo("deserialize sgdstore.Block", &err)
	var vecData []byte
	block = &Block{}
	err = serializer.DeserializeAny(d, &vecData, &block.TrainInput, &block.TrainTarget,
		&block.StepSize, &block.Query, &block.Steps)
	if err != nil {
		return nil, err
	}
	savedVecs, err := serializer.DeserializeSlice(vecData)
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
	res := &State{Params: make([]*anyrnn.VecState, len(b.InitParams))}
	for i, p := range b.InitParams {
		res.Params[i] = anyrnn.NewVecState(p.Vector, n)
	}
	return res
}

// PropagateStart propagates through the start state.
func (b *Block) PropagateStart(s anyrnn.StateGrad, g anydiff.Grad) {
	state := s.(*State)
	for i, paramVar := range b.InitParams {
		state.Params[i].PropagateStart(paramVar, g)
	}
}

// Step evaluates the block.
func (b *Block) Step(s anyrnn.State, in anyvec.Vector) anyrnn.Res {
	state := s.(*State)
	inPool := anydiff.NewVar(in)
	netPool := state.pool()
	present := state.Present()
	n := present.NumPresent()
	gateOuts := b.applyGates(inPool, n)

	allRes := anydiff.PoolMulti(gateOuts, func(gateOuts []anydiff.Res) anydiff.MultiRes {
		poolReses := make([]anydiff.Res, len(netPool))
		for i, x := range netPool {
			poolReses[i] = x
		}
		net := &Net{
			Parameters: anydiff.Fuse(poolReses...),
			Num:        n,
		}
		batchSize := gateOuts[0].Output().Len() / (net.InSize() * n)
		newNet := net.Train(gateOuts[0], gateOuts[1], gateOuts[2], batchSize, b.Steps)
		return anydiff.PoolMulti(newNet.Parameters,
			func(newParams []anydiff.Res) anydiff.MultiRes {
				net1 := *newNet
				net1.Parameters = anydiff.Fuse(newParams...)
				batchSize := gateOuts[3].Output().Len() / (net.InSize() * n)
				applied := newNet.Apply(gateOuts[3], batchSize)
				newReses := append([]anydiff.Res{applied}, newParams...)
				return anydiff.Fuse(newReses...)
			})
	})

	newState := &State{
		Params: make([]*anyrnn.VecState, len(state.Params)),
	}
	for i, newVec := range allRes.Outputs()[1:] {
		newState.Params[i] = &anyrnn.VecState{
			PresentMap: state.Params[i].PresentMap,
			Vector:     newVec,
		}
	}
	v := anydiff.NewVarSet(b.Parameters()...)

	return &blockRes{
		InPool:   inPool,
		NetPools: netPool,
		OutVec:   allRes.Outputs()[0],
		OutState: newState,
		AllRes:   allRes,
		V:        v,
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
	vecData, err := serializer.SerializeSlice(savedVecs)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeAny(
		serializer.Bytes(vecData),
		b.TrainInput,
		b.TrainTarget,
		b.StepSize,
		b.Query,
		b.Steps,
	)
}

// applyGates returns a vector of the form:
//
//     [trainIn, trainTarget, step, query]
//
func (b *Block) applyGates(x anydiff.Res, n int) anydiff.MultiRes {
	gates := []anynet.Layer{b.TrainInput, b.TrainTarget, b.StepSize, b.Query}
	var outs []anydiff.Res
	for _, gate := range gates {
		outs = append(outs, gate.Apply(x, n))
	}
	return anydiff.Fuse(outs...)
}

// State is the anyrnn.State and anyrnn.StateGrad type for
// a Block.
type State struct {
	Params []*anyrnn.VecState
}

// Present returns the present sequence map.
func (s *State) Present() anyrnn.PresentMap {
	return s.Params[0].Present()
}

// Reduce removes states.
func (s *State) Reduce(p anyrnn.PresentMap) anyrnn.State {
	res := &State{Params: make([]*anyrnn.VecState, len(s.Params))}
	for i, param := range s.Params {
		res.Params[i] = param.Reduce(p).(*anyrnn.VecState)
	}
	return res
}

// Expand inserts gradients.
func (s *State) Expand(p anyrnn.PresentMap) anyrnn.StateGrad {
	res := &State{Params: make([]*anyrnn.VecState, len(s.Params))}
	for i, param := range s.Params {
		res.Params[i] = param.Expand(p).(*anyrnn.VecState)
	}
	return res
}

func (s *State) pool() []*anydiff.Var {
	res := make([]*anydiff.Var, len(s.Params))
	for i, packed := range s.Params {
		res[i] = anydiff.NewVar(packed.Vector)
	}
	return res
}

type blockRes struct {
	InPool   *anydiff.Var
	NetPools []*anydiff.Var
	OutVec   anyvec.Vector
	OutState *State
	AllRes   anydiff.MultiRes
	V        anydiff.VarSet
}

func (b *blockRes) State() anyrnn.State {
	return b.OutState
}

func (b *blockRes) Output() anyvec.Vector {
	return b.OutVec
}

func (b *blockRes) Vars() anydiff.VarSet {
	return b.V
}

func (b *blockRes) Propagate(u anyvec.Vector, s anyrnn.StateGrad,
	g anydiff.Grad) (anyvec.Vector, anyrnn.StateGrad) {
	allUpstream := make([]anyvec.Vector, len(b.AllRes.Outputs()))
	allUpstream[0] = u
	if s == nil {
		for i, x := range allUpstream {
			if x == nil {
				size := b.AllRes.Outputs()[i].Len()
				allUpstream[i] = u.Creator().MakeVector(size)
			}
		}
	} else {
		sg := s.(*State)
		for i, vecs := range sg.Params {
			allUpstream[i+1] = vecs.Vector
		}
	}

	for _, p := range b.pools() {
		g[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
		defer func(g anydiff.Grad, p *anydiff.Var) {
			delete(g, p)
		}(g, p)
	}

	b.AllRes.Propagate(allUpstream, g)

	paramGrad := make([]*anyrnn.VecState, len(b.NetPools))
	for i, netPool := range b.NetPools {
		paramGrad[i] = &anyrnn.VecState{
			Vector:     g[netPool],
			PresentMap: b.OutState.Present(),
		}
	}

	return g[b.InPool], &State{Params: paramGrad}
}

func (b *blockRes) pools() []*anydiff.Var {
	return append([]*anydiff.Var{b.InPool}, b.NetPools...)
}
