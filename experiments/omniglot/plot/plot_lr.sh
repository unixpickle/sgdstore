#!/bin/bash

echo 'Creating plot_file.m ...'

TO_MATLAB="./log_to_matlab.sh"
$TO_MATLAB ../log/lr_search/sgdstore_0001_log.txt 64 >plot_file.m
$TO_MATLAB ../log/lr_search/sgdstore_001_log.txt 64 >>plot_file.m
$TO_MATLAB ../log/lr_search/sgdstore_0003_log.txt 64 >>plot_file.m
echo -n 'plot(' >>plot_file.m
first=1
for name in sgdstore_0001 sgdstore_0003 sgdstore_001
do
  cleanName=$(echo $name | tr _ ' ')
  if [ $first -eq 1 ]; then
    first=0
  else
    echo -n ', ' >>plot_file.m
  fi
  echo -n "${name}_log_x, smooth_data(${name}_log_y), " >>plot_file.m
  echo -n "'linewidth', 2, ';${cleanName};'" >>plot_file.m
done
echo ");" >>plot_file.m
