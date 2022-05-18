#!/bin/bash

clean_file=data/adult/adult_nonulls.csv

# Increasing error fraction
for error_f in 0.2 0.4 0.6 0.8
do
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns education --tag_error_frac
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns education-num --tag_error_frac
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns workclass --tag_error_frac
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns marital-status --tag_error_frac
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns race --tag_error_frac
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns relationship --tag_error_frac
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns occupation --tag_error_frac

# Two columns
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns workclass education --tag_error_frac
#Two columns same row
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns workclass education --tag_error_frac --keep_mask --tag "samerow"

# Four columns
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns workclass education education-num marital-status --tag_error_frac
# Four columns same row
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns workclass education education-num marital-status --tag_error_frac --keep_mask --tag "samerow"

# Six columns
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns workclass education education-num marital-status occupation relationship --tag_error_frac
# Six columns same row
python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns workclass education education-num marital-status occupation relationship --tag_error_frac --keep_mask --tag "samerow"

done




## Two dirty columns
#python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns occupation education --tag_error_frac
#python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns occupation workclass --tag_error_frac
#python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns race marital-status --tag_error_frac
#python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns race education --tag_error_frac
#
## Two dirty columns, errors on the same row
#python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns occupation workclass --keep_mask --tag "samerow" --tag_error_frac
#python main_corruption.py -i $clean_file --method simple --error_fraction $error_f --target_columns race marital-status --keep_mask --tag "samerow" --tag_error_frac

