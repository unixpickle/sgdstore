# sgdstore

This is a memory-augmented neural network that uses a neural network as a storage device. Particularly, a *controller network* provides training examples for a *storage network* at every timestep. The *storage network* is trained on these samples with SGD at every timestep (in a differentiable manner)`. The controller can then query the storage network by feeding it inputs and seeing the corresponding outputs. The end result is that the storage network serves as a memory bank which is "written to" via SGD.

# Hypotheses

Neural networks seem to provide a lot of desirable properties as memory modules:

 * They can store a lot of information.
 * They can compress information.
 * They can generalize to new information.
 * They can interpolate between training samples.

In a sense, a neural network can be seen as a key-value store which tries to generalize to new keys. This seems like the perfect memory structure for a memory-augmented neural network.

# Results

Preliminary results on polynomial approximation looked promising. See [experiments/poly_approx](experiments/poly_approx). After seeing those results, I decided to scale up to a harder meta-learning task.

On the Omniglot handwriting dataset, the model (controlled by a vanilla RNN) outperformed an LSTM in training time (measured in epochs) and accuracy. See [experiments/omniglot](experiments/omniglot).
