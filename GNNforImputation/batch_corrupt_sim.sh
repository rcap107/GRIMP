#!/bin/bash

#error_f=0.5
for error_f in 0.02 0.1 0.2  #0.4 0.6
do
for clean_file in data/new-hard/*.csv
do
  for run in {1..1}
  do
  python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_all_columns \
  --tag_error_frac --save_folder data/new-hard/dirty
  done
done
done