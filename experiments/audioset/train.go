package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/audioset/metaset"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func Train(args []string) {
	var dataDir string
	var dataCSV string
	var evalClassPath string

	var learnerNetPath string
	var featureNetPath string
	var modelType string
	var stepSize float64
	var batchSize int
	var sgdSteps int
	var numClasses int
	var episodeLen int

	var audioFeatureSize int
	var pcmChunkSize int

	fs := flag.NewFlagSet("train", flag.ExitOnError)

	fs.StringVar(&dataDir, "datadir", "", "directory of AudioSet samples")
	fs.StringVar(&dataCSV, "datacsv", "", "path to AudioSet CSV file")
	fs.StringVar(&evalClassPath, "dataclasses", "eval_classes.txt", "path to eval classes")

	fs.StringVar(&learnerNetPath, "learner", "out_learner", "learner net output path")
	fs.StringVar(&featureNetPath, "features", "out_features", "feature net output path")
	fs.StringVar(&modelType, "model", "sgdstore",
		"model type (sgdstore, lstm, or vanilla)")
	fs.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	fs.IntVar(&batchSize, "batch", 16, "SGD batch size")
	fs.IntVar(&sgdSteps, "steps", 1, "steps per sgdstore")
	fs.IntVar(&numClasses, "classes", 5, "classes per episode")
	fs.IntVar(&episodeLen, "eplen", 50, "episode length")

	fs.IntVar(&audioFeatureSize, "audiofeats", 128, "audio feature vector size")
	fs.IntVar(&pcmChunkSize, "chunksize", 512, "PCM sample chunk size")

	fs.Parse(args)

	if dataDir == "" || dataCSV == "" {
		essentials.Die("Required flags: -datadir and -datacsv. See -help.")
	}

	var learner, features anyrnn.Block
	if err := serializer.LoadAny(learnerNetPath, &learner); err != nil {
		log.Println("Creating new learner...")
		learner = learnerBlock(modelType, sgdSteps, audioFeatureSize, numClasses)
	} else {
		log.Println("Loaded learner.")
	}
	if err := serializer.LoadAny(featureNetPath, &features); err != nil {
		log.Println("Creating new feature net...")
		features = featureBlock(audioFeatureSize, pcmChunkSize)
	} else {
		log.Println("Loaded feature net.")
	}

	allSamples, err := metaset.ReadSet(dataDir, dataCSV)
	if err != nil {
		essentials.Die(err)
	}

	training, eval := allSamples.Split(readEvalClasses(evalClassPath))

	log.Printf("Got %d samples: %d training, %d eval", len(allSamples), len(training),
		len(eval))

	trainer := &metaset.Trainer{
		Creator: anyvec32.CurrentCreator(),
		FeatureFunc: func(seq anyseq.Seq) anydiff.Res {
			return anyseq.Tail(anyrnn.Map(seq, features))
		},
		LearnerFunc: func(eps anyseq.Seq) anyseq.Seq {
			return anyrnn.Map(eps, learner)
		},
		Params:     anynet.AllParameters(learner, features),
		Set:        training,
		NumClasses: numClasses,
		NumSteps:   episodeLen,
		ChunkSize:  pcmChunkSize,
		Average:    true,
	}

	valBatches := fetchEvalBatches(*trainer, eval, batchSize)

	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     trainer,
		Gradienter:  trainer,
		Transformer: &anysgd.RMSProp{},
		Samples:     anysgd.LengthSampleList(batchSize),
		Rater:       anysgd.ConstRater(stepSize),
		BatchSize:   batchSize,
		StatusFunc: func(b anysgd.Batch) {
			if iter%4 == 0 {
				batch := <-valBatches
				valCost := anyvec.Sum(trainer.TotalCost(batch).Output())
				log.Printf("iter %d: cost=%v validation=%v", iter, trainer.LastCost,
					valCost)
			} else {
				log.Printf("iter %d: cost=%v", iter, trainer.LastCost)
			}
			iter++
		},
	}

	if err := sgd.Run(rip.NewRIP().Chan()); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}

	if err := serializer.SaveAny(learnerNetPath, learner); err != nil {
		essentials.Die(err)
	}
	if err := serializer.SaveAny(featureNetPath, features); err != nil {
		essentials.Die(err)
	}
}

func readEvalClasses(path string) []string {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		essentials.Die(err)
	}
	return strings.Fields(string(data))
}

func fetchEvalBatches(t metaset.Trainer, set metaset.Set, size int) <-chan anysgd.Batch {
	res := make(chan anysgd.Batch, 1)
	go func() {
		defer close(res)
		t.Set = set
		for {
			batch, err := t.Fetch(anysgd.LengthSampleList(size))
			if err != nil {
				essentials.Die(err)
			}
			res <- batch
		}
	}()
	return res
}
