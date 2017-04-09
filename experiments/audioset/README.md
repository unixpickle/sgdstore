# AudioSet

In this meta-learning experiment, I adapt the [AudioSet dataset](https://research.google.com/audioset/) to the domain of meta-learning. There are 527 sound classes in AudioSet; I split this into 477 training classes and [50 evaluation classes](eval_classes.txt). The task is very similar to the task from [Omniglot](../omniglot): the model is presented with sample after sample and has to predict the randomly assigned class of each sample.

Unlike for Omniglot, there are two RNN components in the model for this experiment. There is the meta-learning, which is also present in Omniglot. There is also the feature RNN, which takes variable-length audio segments and converts them to fixed-length vectors.

# Initial results

Initial results do not look good. Random guessing would have a loss of `ln(1/5)=1.609`. Training loss gets down to about 1.54. Evaluation loss stays at about 1.615, which is worse than random. Thus, the model is overfitting, and even then it's barely fitting the training data.

The above results were with roughly 18k samples, all taken from the official evaluation subset of AudioSet. I have just downloaded another 19k samples from the test set, so that extra data may prove very helpful.

Besides using more data, I will also experiment with higher-capacity models. More classes may be necessary for the model not to overfit, in which case data augmentation may be necessary. For example, I might double the number of classes by reversing samples, speeding them up, or overlaying them.
