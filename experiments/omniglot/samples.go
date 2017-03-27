package main

import (
	"math/rand"

	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/omniglot"
)

const ImageSize = 20

// Samples is a dummy anys2s.SampleList which generates
// random episodes.
type Samples struct {
	Length  int
	Sets    []omniglot.Set
	Augment bool

	NumClasses   int
	NumTimesteps int
}

func (s *Samples) Len() int {
	return s.Length
}

func (s *Samples) Swap(i, j int) {
}

func (s *Samples) Slice(i, j int) anysgd.SampleList {
	return &Samples{
		Length:       j - i,
		Sets:         s.Sets,
		Augment:      s.Augment,
		NumClasses:   s.NumClasses,
		NumTimesteps: s.NumTimesteps,
	}
}

func (s *Samples) Creator() anyvec.Creator {
	return anyvec32.CurrentCreator()
}

func (s *Samples) GetSample(i int) (*anys2s.Sample, error) {
	c := s.Creator()
	samples, classes := s.episode()
	oneHot := make([]float64, s.NumClasses)
	var inputs, outputs []anyvec.Vector
	for i, sample := range samples {
		img, err := sample.Image(s.Augment, ImageSize)
		if err != nil {
			return nil, err
		}
		class := classes[i]
		inVec := append(omniglot.Tensor(img), oneHot...)
		oneHot = make([]float64, s.NumClasses)
		oneHot[class] = 1
		inputs = append(inputs, c.MakeVectorData(c.MakeNumericList(inVec)))
		outputs = append(outputs, c.MakeVectorData(c.MakeNumericList(oneHot)))
	}
	return &anys2s.Sample{Input: inputs, Output: outputs}, nil
}

func (s *Samples) episode() (samples []*omniglot.AugSample, classes []int) {
	for class, setIdx := range rand.Perm(len(s.Sets))[:s.NumClasses] {
		set := s.Sets[setIdx]
		for _, x := range set {
			samples = append(samples, x)
			classes = append(classes, class)
		}
	}
	for i := 0; i < len(samples); i++ {
		idx := rand.Intn(len(samples)-i) + i
		samples[i], samples[idx] = samples[idx], samples[i]
		classes[i], classes[idx] = classes[idx], classes[i]
	}
	if len(samples) > s.NumTimesteps {
		samples = samples[:s.NumTimesteps]
		classes = classes[:s.NumTimesteps]
	}
	return
}
