package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/omniglot"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func main() {
	var trainingPath string
	var testingPath string
	var modelPath string
	var modelType string
	var stepSize float64
	var sgdSteps int
	var batchSize int
	var numClasses int
	var episodeLen int

	flag.StringVar(&trainingPath, "training", "", "training data directory")
	flag.StringVar(&testingPath, "testing", "", "testing data directory")
	flag.StringVar(&modelPath, "out", "model_out", "model output path")
	flag.StringVar(&modelType, "model", "sgdstore", "model type (sgdstore or lstm)")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.IntVar(&sgdSteps, "steps", 1, "steps per sgdstore")
	flag.IntVar(&batchSize, "batch", 16, "SGD batch size")
	flag.IntVar(&numClasses, "classes", 5, "classes per episode")
	flag.IntVar(&episodeLen, "eplen", 50, "episode length")

	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	if trainingPath == "" || testingPath == "" {
		essentials.Die("Required flags: -training and -testing. See -help.")
	}

	var model anyrnn.Block
	if err := serializer.LoadAny(modelPath, &model); err != nil {
		log.Println("Creating new model.")
		model = NewModel(modelType, sgdSteps, numClasses)
	} else {
		log.Println("Loaded model.")
	}

	training, err := omniglot.ReadSet(trainingPath)
	if err != nil {
		essentials.Die(err)
	}
	training = training.Augment()

	testing, err := omniglot.ReadSet(testingPath)
	if err != nil {
		essentials.Die(err)
	}
	testing = testing.Augment()

	samples := &Samples{
		Length:       batchSize,
		Sets:         training.ByClass(),
		Augment:      true,
		NumClasses:   numClasses,
		NumTimesteps: episodeLen,
	}
	testSamples := *samples
	testSamples.Sets = testing.ByClass()
	trainer := &anys2s.Trainer{
		Func: func(s anyseq.Seq) anyseq.Seq {
			return anyrnn.Map(s, model)
		},
		Params:  model.(anynet.Parameterizer).Parameters(),
		Cost:    anynet.DotCost{},
		Average: true,
	}

	var iter int
	sgd := &anysgd.SGD{
		Gradienter:  trainer,
		Fetcher:     trainer,
		Transformer: &anysgd.Adam{},
		Rater:       anysgd.ConstRater(stepSize),
		Samples:     samples,
		BatchSize:   batchSize,
		StatusFunc: func(b anysgd.Batch) {
			if iter%4 == 0 {
				batch, err := trainer.Fetch(&testSamples)
				if err != nil {
					essentials.Die(err)
				}
				cost := trainer.TotalCost(batch)
				log.Printf("iter %d: cost=%v validation=%f", iter, trainer.LastCost,
					anyvec.Sum(cost.Output()))
			} else {
				log.Printf("iter %d: cost=%v", iter, trainer.LastCost)
			}
			iter++
		},
	}

	if err := sgd.Run(rip.NewRIP().Chan()); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}

	if err := serializer.SaveAny(modelPath, model); err != nil {
		essentials.Die(err)
	}
}
