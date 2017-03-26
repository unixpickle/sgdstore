#!/bin/bash

TO_MATLAB="./log_to_matlab.sh"
$TO_MATLAB ../log/sgdstore_1step.txt >plot_file.m
$TO_MATLAB ../log/sgdstore_2step.txt >>plot_file.m
$TO_MATLAB ../log/sgdstore_3step.txt >>plot_file.m
$TO_MATLAB ../log/lstm.txt >>plot_file.m
echo -n 'plot(' >>plot_file.m
for i in {1..3}
do
  echo -n "sgdstore_${i}step_x, smooth_data(sgdstore_${i}step_y), " >>plot_file.m
  echo -n "'linewidth', 2, ';SS-${i};', " >>plot_file.m
done
echo "lstm_x(1:10500), smooth_data(lstm_y(1:10500)), 'linewidth', 2, ';LSTM;');" >>plot_file.m
