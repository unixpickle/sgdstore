package main

import (
	"flag"
	"fmt"
	"math"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func Analysis(args []string) {
	var modelPath string
	fs := flag.NewFlagSet("analysis", flag.ExitOnError)
	fs.StringVar(&modelPath, "model", "", "path to model file")
	fs.Parse(args)

	if modelPath == "" {
		essentials.Die("Required flag: -model. See -help.")
	}

	var model anyrnn.Block
	if err := serializer.LoadAny(modelPath, &model); err != nil {
		essentials.Die(err)
	}

	for i, p := range model.(anynet.Parameterizer).Parameters() {
		v := p.Vector
		fmt.Printf("Param %d: mean=%f stddev=%f\n", i, computeMean(v),
			math.Sqrt(float64(computeVariance(v))))
	}
}

func computeMean(vec anyvec.Vector) float32 {
	return anyvec.Sum(vec).(float32) / float32(vec.Len())
}

func computeVariance(vec anyvec.Vector) float32 {
	mean := computeMean(vec)
	sq := vec.Copy()
	anyvec.Pow(sq, float32(2))
	moment2 := computeMean(sq)
	return moment2 - mean*mean
}
