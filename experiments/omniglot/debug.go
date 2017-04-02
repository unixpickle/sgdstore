package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgdstore"
)

func Debug(args []string) {
	var inPath, outPath string
	var logStepSize bool
	var logTrainIn bool
	var logTrainTarget bool
	fs := flag.NewFlagSet("debug", flag.ExitOnError)
	fs.StringVar(&inPath, "in", "model_out", "input model path")
	fs.StringVar(&outPath, "out", "", "output model path")
	fs.BoolVar(&logStepSize, "logstep", false, "log step size")
	fs.BoolVar(&logTrainIn, "logtrain", false, "log training inputs")
	fs.BoolVar(&logTrainTarget, "logtarget", false, "log training targets")
	fs.Parse(args)

	if inPath == "" || outPath == "" {
		essentials.Die("Required flag: -in and -out. See -help.")
	}

	if !logStepSize && !logTrainIn && !logTrainTarget {
		fmt.Fprintln(os.Stderr, "Warning: no new debug layers. See -help.")
	}

	var model anyrnn.Block
	if err := serializer.LoadAny(inPath, &model); err != nil {
		essentials.Die(err)
	}

	stack := model.(anyrnn.Stack)
	for _, layer := range stack {
		if par, ok := layer.(*anyrnn.Parallel); ok {
			layer = par.Block2
		}
		if block, ok := layer.(*sgdstore.Block); ok {
			if logStepSize {
				net := block.StepSize.(anynet.Net)
				block.StepSize = append(net, debugLayer("step"))
			}
			if logTrainIn {
				net := block.TrainInput
				block.TrainInput = anynet.Net{net, debugLayer("train")}
			}
			if logTrainTarget {
				net := block.TrainTarget.(anynet.Net)
				block.TrainTarget = append(net, debugLayer("target"))
			}
		}
	}

	if err := serializer.SaveAny(outPath, model); err != nil {
		essentials.Die(err)
	}
}

func debugLayer(id string) anynet.Layer {
	return &anynet.Debug{ID: id, PrintRaw: true}
}
