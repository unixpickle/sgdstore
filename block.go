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
	layerSizes ...int) *Block {
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
			anynet.NewFC(c, blockIn, 1),
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
	var savedVecs []serializer.Serializer
	block = &Block{}
	err = serializer.DeserializeAny(d, &savedVecs, &block.TrainInput, &block.TrainTarget,
		&block.StepSize, &block.Query, &block.Steps)
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
		ParamSizes: make([]int, len(b.InitParams)),
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

// Step evaluates the block.
func (b *Block) Step(s anyrnn.State, in anyvec.Vector) anyrnn.Res {
	state := s.(*State)
	inPool := anydiff.NewVar(in)
	netPool := state.pool()
	present := state.Present()
	n := present.NumPresent()
	gateOuts := b.applyGates(inPool, n)

	var outVecs []anyvec.Vector
	outParamVecs := make([][]anyvec.Vector, len(present))

	allRes := anydiff.PoolMulti(gateOuts, func(gateOuts []anydiff.Res) anydiff.MultiRes {
		var outParts []anydiff.MultiRes
		var i int
		for lane := 0; lane < len(present); lane++ {
			if !present[lane] {
				continue
			}
			trainIn, trainTarg := gateOuts[i], gateOuts[i+n]
			step, query := gateOuts[i+2*n], gateOuts[i+3*n]
			var reses []anydiff.Res
			for _, v := range netPool[lane] {
				reses = append(reses, v)
			}
			net := &Net{Parameters: anydiff.Fuse(reses...)}
			out := b.applyNet(trainIn, trainTarg, step, query, net)
			outParts = append(outParts, out)
			outVecs = append(outVecs, out.Outputs()[0])
			outParamVecs[lane] = out.Outputs()[1:]
			i++
		}
		return anydiff.FuseMulti(outParts...)
	})

	newState := &State{
		Creator:    state.Creator,
		ParamSizes: state.ParamSizes,
		Params:     outParamVecs,
	}
	v := anydiff.NewVarSet(b.Parameters()...)

	return &blockRes{
		InPool:   inPool,
		NetPools: netPool,
		OutVec:   state.Creator.Concat(outVecs...),
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
	return serializer.SerializeAny(
		savedVecs,
		b.TrainInput,
		b.TrainTarget,
		b.StepSize,
		b.Query,
		b.Steps,
	)
}

// applyGates returns a vector of the form:
//
//     [
//         trainIn1, ..., trainInN,
//         trainTarget1, ..., trainTargetN,
//         step1, ..., stepN,
//         query1, ..., queryN,
//     ]
//
func (b *Block) applyGates(x anydiff.Res, n int) anydiff.MultiRes {
	gates := []anynet.Layer{b.TrainInput, b.TrainTarget, b.StepSize, b.Query}
	var split []anydiff.MultiRes
	for _, gate := range gates {
		split = append(split, splitRes(gate.Apply(x, n), n))
	}
	return anydiff.FuseMulti(split...)
}

// applyNet trains and applies a network and returns a
// tuple of the form: [queryOut, newParam1, ...].
func (b *Block) applyNet(in, target, step, query anydiff.Res, net *Net) anydiff.MultiRes {
	batchSize := in.Output().Len() / net.InSize()
	newNet := net.Train(in, target, step, batchSize, b.Steps).Parameters
	return anydiff.PoolMulti(newNet, func(params []anydiff.Res) anydiff.MultiRes {
		newNet := &Net{Parameters: anydiff.Fuse(params...)}
		queryRes := newNet.Apply(query, query.Output().Len()/net.InSize())
		return anydiff.Fuse(append([]anydiff.Res{queryRes}, params...)...)
	})
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

func (s *State) pool() [][]*anydiff.Var {
	res := make([][]*anydiff.Var, len(s.Params))
	for i, vecs := range s.Params {
		if vecs == nil {
			continue
		}
		for _, vec := range vecs {
			p := anydiff.NewVar(vec)
			res[i] = append(res[i], p)
		}
	}
	return res
}

type blockRes struct {
	InPool   *anydiff.Var
	NetPools [][]*anydiff.Var
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
	n := b.OutState.Present().NumPresent()
	uSliceLen := u.Len() / n
	allUpstream := make([]anyvec.Vector, len(b.AllRes.Outputs()))
	vecsPerLane := len(allUpstream) / n
	for i := 0; i < n; i++ {
		uSlice := u.Slice(i*uSliceLen, (i+1)*uSliceLen)
		allUpstream[i*vecsPerLane] = uSlice
	}
	if s == nil {
		for i, x := range allUpstream {
			if x == nil {
				size := b.AllRes.Outputs()[i].Len()
				allUpstream[i] = u.Creator().MakeVector(size)
			}
		}
	} else {
		sg := s.(*State)
		present := sg.Present()
		var i int
		for lane, vecs := range sg.Params {
			if !present[lane] {
				continue
			}
			for j, vec := range vecs {
				allUpstream[1+i*vecsPerLane+j] = vec
			}
			i++
		}
	}

	for _, p := range b.pools() {
		g[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
		defer delete(g, p)
	}

	b.AllRes.Propagate(allUpstream, g)

	paramGrad := make([][]anyvec.Vector, len(b.NetPools))
	for i, netPool := range b.NetPools {
		if netPool == nil {
			continue
		}
		paramGrad[i] = make([]anyvec.Vector, len(netPool))
		for j, x := range netPool {
			paramGrad[i][j] = g[x]
		}
	}

	return g[b.InPool], &State{
		Creator:    b.OutState.Creator,
		ParamSizes: b.OutState.ParamSizes,
		Params:     paramGrad,
	}
}

func (b *blockRes) pools() []*anydiff.Var {
	res := []*anydiff.Var{b.InPool}
	for _, list := range b.NetPools {
		for _, v := range list {
			res = append(res, v)
		}
	}
	return res
}

func splitRes(r anydiff.Res, n int) anydiff.MultiRes {
	return anydiff.PoolFork(r, func(r anydiff.Res) anydiff.MultiRes {
		chunkSize := r.Output().Len() / n
		var reses []anydiff.Res
		for i := 0; i < n; i++ {
			reses = append(reses, anydiff.Slice(r, i*chunkSize, (i+1)*chunkSize))
		}
		return anydiff.Fuse(reses...)
	})
}
