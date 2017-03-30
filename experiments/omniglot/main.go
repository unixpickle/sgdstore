package main

import (
	"fmt"
	"os"

	"github.com/unixpickle/essentials"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, os.Args[0], "<sub-command> [args | -help]")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Subcommands:")
		fmt.Fprintln(os.Stderr, " train     train a new or existing model")
		fmt.Fprintln(os.Stderr, " analysis  dump weight statistics")
		fmt.Fprintln(os.Stderr, " accuracy  evaluate model")
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}
	switch os.Args[1] {
	case "train":
		Train(os.Args[2:])
	case "analysis":
		Analysis(os.Args[2:])
	case "accuracy":
		Accuracy(os.Args[2:])
	default:
		essentials.Die("unknown sub-command:", os.Args[1])
	}
}
