package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/omniglot"
	"github.com/unixpickle/serializer"
)

func Accuracy(args []string) {
	var modelPath string
	var dataPath string
	var batchSize int
	var numClasses int
	var episodeLen int

	fs := flag.NewFlagSet("accuracy", flag.ExitOnError)
	fs.StringVar(&modelPath, "model", "model_out", "model path")
	fs.StringVar(&dataPath, "data", "", "path to evaluation data")
	fs.IntVar(&batchSize, "batch", 16, "")
	fs.IntVar(&numClasses, "classes", 5, "classes per episode")
	fs.IntVar(&episodeLen, "eplen", 50, "episode length")
	fs.Parse(args)

	if dataPath == "" {
		essentials.Die("Required flag: -data. See -help.")
	}

	var model anyrnn.Block
	if err := serializer.LoadAny(modelPath, &model); err != nil {
		essentials.Die(err)
	}

	data, err := omniglot.ReadSet(dataPath)
	if err != nil {
		essentials.Die(err)
	}
	data = data.Augment()

	tr := &anys2s.Trainer{}
	samples := &Samples{
		Length:       batchSize,
		Sets:         data.ByClass(),
		Augment:      false,
		NumClasses:   numClasses,
		NumTimesteps: episodeLen,
	}

	totalSeen := map[int]int{}
	totalCorrect := map[int]int{}

	for {
		batch, err := tr.Fetch(samples)
		if err != nil {
			essentials.Die(err)
		}
		b := batch.(*anys2s.Batch)
		actual := anyseq.SeparateSeqs(anyrnn.Map(b.Inputs, model).Output())
		expected := anyseq.SeparateSeqs(b.Outputs.Output())

		for i, actualSeq := range actual {
			expectedSeq := expected[i]
			seenCounts := map[int]int{}
			for j, a := range actualSeq {
				predicted := anyvec.MaxIndex(a)
				correct := anyvec.MaxIndex(expectedSeq[j])
				seen := seenCounts[correct]
				if correct == predicted {
					totalCorrect[seen]++
				}
				totalSeen[seen]++
				seenCounts[correct]++
			}
		}

		printAccuracies(totalSeen, totalCorrect)
	}
}

func printAccuracies(seen, correct map[int]int) {
	for i := 0; i <= 10; i++ {
		percent := 100 * float64(correct[i]) / float64(seen[i])
		fmt.Printf("Instance %d: %.2f%%\n", i, percent)
	}
}
