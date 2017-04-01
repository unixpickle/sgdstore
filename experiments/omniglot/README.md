# Omniglot

In an episode of this experiment, the model is presented with a sequence of 50 images. It knows beforehand that there are five different kinds of characters with five different labels. However, it does not know what the labels are, nor has it seen the exact characters before. How well can a model learn to do this task, if you train it with enough background knowledge?

The answer is: very well. The sgdstore model achieves the following levels of accuracy on the n-th instance of a character:

<table>
  <tr>
    <th>Instance 1</th>
    <th>Instance 2</th>
    <th>Instance 3</th>
    <th>Instance 4</th>
    <th>Instance 10</th>
  </tr>
  <tr>
    <td>35.76%</td>
    <td>89.38%</td>
    <td>93.72%</td>
    <td>95.25%</td>
    <td>96.74%</td>
  </tr>
</table>

Note that these results look worse than the results in [Santoro et al.](https://arxiv.org/abs/1605.06065). This is likely due to the fact that I use a different training/evaluation split. In the aforementioned paper, the network is meta-trained on more data and tested on less data. The models I trained did indeed overfit slightly, indicating that more training data would be helpful. I am using the original background/evaluation split from [Lake et al.](http://science.sciencemag.org/content/350/6266/1332).

The following graph shows, for three different models, the validation error over time during meta-training. The sgdstore model clearly does the best, but the LSTM catches up after way more training. The vanilla RNN (which is used as a controller for sgdstore), does terribly on its own. I will update the graph after I have run the sgdstore model for longer.

![plot/plot.png](plot/plot.png)

I have yet to run experiments with the 15-class task. I suspect that the LSTM will have more difficulty on that task. That is where the vanilla RNN will truly show its colors.

I have found that I can get much better LSTM results than the ones reported in *Santoro et al.*. They stop training after 100,000 episodes, which seems arbitrary (almost like it was chosen to make their model look good, since it learns faster). I don't want to confuse learning speed with model capacity, which *Santoro et al.* seems to do.
