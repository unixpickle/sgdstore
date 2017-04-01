#!/bin/bash

echo 'Creating plot_file.m ...'

TO_MATLAB="./log_to_matlab.sh"
$TO_MATLAB ../log/lstm_log.txt >plot_file.m
$TO_MATLAB ../log/sgdstore_log.txt >>plot_file.m
$TO_MATLAB ../log/vanilla_log.txt >>plot_file.m
echo -n 'plot(' >>plot_file.m
first=1
for name in lstm sgdstore vanilla
do
  if [ $first -eq 1 ]; then
    first=0
  else
    echo -n ', ' >>plot_file.m
  fi
  echo -n "${name}_log_x, smooth_data(${name}_log_y), " >>plot_file.m
  echo -n "'linewidth', 2, ';${name};'" >>plot_file.m
done
echo ");" >>plot_file.m
