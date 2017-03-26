package main

import (
	"math"
	"math/rand"

	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

type SampleList int

func (s SampleList) Len() int {
	return int(s)
}

func (s SampleList) Swap(i, j int) {
}

func (s SampleList) Slice(i, j int) anysgd.SampleList {
	return SampleList(j - i)
}

func (s SampleList) Creator() anyvec.Creator {
	return anyvec32.CurrentCreator()
}

func (s SampleList) GetSample(i int) (*anys2s.Sample, error) {
	poly := RandomPoly()

	var sample anys2s.Sample
	var lastValue float32

	for i := 0; i < EpisodeLen; i++ {
		inVec := make([]float32, 4)
		for j := range inVec[1:] {
			inVec[j+1] = float32(rand.NormFloat64())
		}
		inVec[0] = lastValue
		lastValue = poly.Eval(inVec[1:])
		outVec := []float32{lastValue}
		sample.Input = append(sample.Input, anyvec32.MakeVectorData(inVec))
		sample.Output = append(sample.Output, anyvec32.MakeVectorData(outVec))
	}

	return &sample, nil
}

type Term struct {
	X int
	Y int
	Z int

	Coeff float32
}

func (t Term) Eval(coord []float32) float32 {
	res := t.Coeff
	for i, pow := range []int{t.X, t.Y, t.Z} {
		res *= float32(math.Pow(float64(coord[i]), float64(pow)))
	}
	return res
}

type Poly []Term

func RandomPoly() Poly {
	var poly Poly
	for x := 0; x <= 3; x++ {
		for y := 0; y <= 3-x; y++ {
			for z := 0; z <= 3-(x+y); z++ {
				poly = append(poly, Term{
					X:     x,
					Y:     y,
					Z:     z,
					Coeff: float32(rand.NormFloat64()) / 5,
				})
			}
		}
	}
	return poly
}

func (p Poly) Eval(coord []float32) float32 {
	var sum float32
	for _, t := range p {
		sum += t.Eval(coord)
	}
	return sum
}
