#!/bin/bash

echo 'Creating plot_file.m ...'

TO_MATLAB="./log_to_matlab.sh"
$TO_MATLAB ../log/mem_search/sgdstore_4rh_log.txt 64 >plot_file.m
$TO_MATLAB ../log/mem_search/sgdstore_deep_log.txt 64 >>plot_file.m
$TO_MATLAB ../log/15_classes/sgdstore_log.txt 64 >>plot_file.m
echo -n 'plot(' >>plot_file.m
first=1
for name in sgdstore_4rh sgdstore_deep sgdstore
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
