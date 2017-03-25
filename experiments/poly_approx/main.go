package main

import (
	"flag"
	"log"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/rip"
)

const EpisodeLen = 64

func main() {
	var batchSize int
	var stepSize float64
	var modelName string
	var sgdSteps int

	flag.IntVar(&batchSize, "batch", 16, "SGD batch size")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.StringVar(&modelName, "model", "sgdstore", "RNN type (sgdstore or lstm)")
	flag.IntVar(&sgdSteps, "sgdsteps", 2, "SGD steps for sgdstore")

	flag.Parse()

	model := NewModel(modelName, sgdSteps)
	samples := SampleList(batchSize)

	trainer := &anys2s.Trainer{
		Func: func(s anyseq.Seq) anyseq.Seq {
			return anyrnn.Map(s, model)
		},
		Cost:    anynet.MSE{},
		Params:  model.(anynet.Parameterizer).Parameters(),
		Average: true,
	}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     trainer,
		Gradienter:  trainer,
		Transformer: &anysgd.Adam{},
		Samples:     samples,
		BatchSize:   batchSize,
		Rater:       anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iter, trainer.LastCost)
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())
}
