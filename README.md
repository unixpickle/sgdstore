# sgdstore

This is just a crazy idea I had: what if you augment RNN memory by allowing the RNN to train a neural net at every time-step. Specifically, the RNN would output training samples for the network and a step size. It would also output a batch of samples to run through the network (acting as a memory query).
